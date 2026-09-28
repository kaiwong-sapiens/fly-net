"""
Fast whole-brain leaky integrate-and-fire (LIF) simulator for a fly connectome.

Re-implements the model of Shiu et al. 2024 (Nature, "A Drosophila computational
brain model reveals sensorimotor processing"), which runs the FlyWire connectome
in Brian2, as a numba kernel so it can be driven in closed loop by a game.

Every neuron is identical; the connectome alone sets who talks to whom and how
strongly: weight = (#synapses) x (sign from predicted neurotransmitter) x w_syn.
    dv/dt = (v0 - v + g) / t_mbr        (unless refractory)
    dg/dt = -g / tau                    (unless refractory)
    spike when v > v_th  ->  v = v_rst, g = 0, refractory t_rfc
    presynaptic spike -> g_post += w after a 1.8 ms delay
                         (discarded if the target is refractory -- Brian2 does this
                          automatically for variables flagged "unless refractory")
Stimulated ("sensory") neurons receive Poisson kicks that each trigger one spike,
exactly as the original model's PoissonInput with f_poi = 250.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

# Parameters from Shiu et al. 2024 (model.py default_params); units: ms, mV
PARAMS = dict(
    v_0=-52.0,     # resting potential
    v_rst=-52.0,   # reset potential
    v_th=-45.0,    # spike threshold
    t_mbr=20.0,    # membrane time constant
    tau=5.0,       # synaptic time constant
    t_rfc=2.2,     # refractory period
    t_dly=1.8,     # synaptic delay
    w_syn=0.275,   # mV per synapse
    f_poi=250.0,   # Poisson kick = f_poi * w_syn (always causes a spike)
    dt=0.1,        # integration step (Brian2 default)
)


@dataclass
class Connectome:
    n: int
    indptr: np.ndarray      # CSR over presynaptic neurons
    indices: np.ndarray     # postsynaptic neuron index
    weights: np.ndarray     # signed synapse count (float32) -- multiplied by w_syn in the kernel
    root_ids: np.ndarray    # neuron id per model index (FlyWire root id / neuPrint bodyId)
    meta: pd.DataFrame = field(default=None, repr=False)  # annotations aligned to model index
    voxel_nm: tuple = (4.0, 4.0, 40.0)                    # size of one coordinate unit in meta pos_*/soma_*

    def idx(self, root_ids):
        m = {r: i for i, r in enumerate(self.root_ids)}
        return np.array([m[r] for r in root_ids], dtype=np.int64)

    def by_type(self, cell_type, side=None):
        m = self.meta
        sel = m.cell_type == cell_type
        if side is not None:
            sel &= m.side == side
        return np.flatnonzero(sel.values)


def load_flywire(data_dir: str | Path, version: str = "783") -> Connectome:
    data_dir = Path(data_dir)
    files = {"783": ("Completeness_783.csv", "Connectivity_783.parquet"),
             "630": ("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet")}
    f_comp, f_con = files[version]
    comp = pd.read_csv(data_dir / "Drosophila_brain_model" / f_comp, index_col=0)
    con = pd.read_parquet(
        data_dir / "Drosophila_brain_model" / f_con,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    n = len(comp)
    pre = con["Presynaptic_Index"].to_numpy(np.int64)
    post = con["Postsynaptic_Index"].to_numpy(np.int32)
    w = con["Excitatory x Connectivity"].to_numpy(np.float32)
    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, pre + 1, 1)
    indptr = np.cumsum(indptr)
    ann = pd.read_csv(
        data_dir / "flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        sep="\t", low_memory=False,
    )
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    meta = ann.reindex(comp.index)
    meta.index.name = "root_id"
    meta = meta.reset_index()
    return Connectome(n=n, indptr=indptr, indices=post, weights=w,
                      root_ids=comp.index.to_numpy(), meta=meta)


@njit(cache=True)
def _run(indptr, indices, weights, n, n_steps, stim_idx, stim_rate_hz, silenced,
         rec_idx, bin_steps, seed, p):
    """Core loop. stim_rate_hz is (n_rows, n_stim): row b applies to time bin b (last row repeats)."""
    np.random.seed(seed)
    v0, vth, tmbr, tau, trfc, tdly, wsyn, fpoi, dt = p
    am = math.exp(-dt / tmbr)
    as_ = math.exp(-dt / tau)
    c = tau / (tau - tmbr) * (as_ - am)
    thr = vth - v0
    kick = fpoi * wsyn
    D = int(round(tdly / dt))
    R = int(round(trfc / dt))
    u = np.zeros(n, np.float32)
    g = np.zeros(n, np.float32)
    # Brian2 refractoriness semantics (verified against the published model):
    # a neuron that spikes at step t is refractory for steps t .. t+R-1; while refractory
    # its v and g are frozen AND synaptic input addressed to it is discarded.
    # Poisson-driven neurons have t_rfc = 0 -> refractory only in their spike step.
    ref_until = np.zeros(n, np.int64)
    is_stim = np.zeros(n, np.bool_)
    for k in range(stim_idx.shape[0]):
        is_stim[stim_idx[k]] = True
    buf = np.zeros((D, n), np.float32)
    counts = np.zeros(n, np.int32)
    n_bins = n_steps // bin_steps
    rec = np.zeros((n_bins, rec_idx.shape[0]), np.int32)
    rec_pos = -np.ones(n, np.int32)
    for k in range(rec_idx.shape[0]):
        rec_pos[rec_idx[k]] = k
    spk = np.empty(n, np.int32)
    n_rate_rows = stim_rate_hz.shape[0]
    for t in range(n_steps):
        b = t // bin_steps
        # 1) integrate (exact solution of the linear ODE over dt), skipped while refractory
        for i in range(n):
            if t < ref_until[i]:
                continue
            gi = g[i]
            if gi != 0.0 or u[i] != 0.0:
                u[i] = u[i] * am + gi * c
                g[i] = gi * as_
        # 2) threshold
        ns = 0
        for i in range(n):
            if u[i] > thr and t >= ref_until[i]:
                spk[ns] = i
                ns += 1
                ref_until[i] = t + (1 if is_stim[i] else R)
        # 3) synapses: deliver spikes emitted D steps ago (dropped if target refractory), Poisson kicks
        slot = t % D
        row = buf[slot]
        for i in range(n):
            if row[i] != 0.0:
                if t >= ref_until[i]:
                    g[i] += row[i]
                row[i] = 0.0
        for k in range(stim_idx.shape[0]):
            r = stim_rate_hz[min(b, n_rate_rows - 1), k]
            if r > 0.0 and np.random.random() < r * dt * 1e-3:
                u[stim_idx[k]] += kick
        # 4) reset + schedule outgoing spikes
        for s in range(ns):
            i = spk[s]
            u[i] = 0.0
            g[i] = 0.0
            counts[i] += 1
            if rec_pos[i] >= 0 and b < n_bins:
                rec[b, rec_pos[i]] += 1
            if silenced[i]:
                continue
            for e in range(indptr[i], indptr[i + 1]):
                buf[slot, indices[e]] += weights[e] * wsyn
    return counts, rec


def run(con: Connectome, stim_idx, rate_hz=150.0, t_ms=1000.0, silence=None,
        rec_idx=None, bin_ms=10.0, seed=0, params=PARAMS, weights=None):
    """Simulate; returns (spike counts per neuron, binned spikes for rec_idx).
    `weights` optionally overrides con.weights (signed synapse counts, float32, CSR order)."""
    p = PARAMS if params is None else params
    stim_idx = np.asarray(stim_idx, np.int64)
    if np.isscalar(rate_hz):
        rates = np.full((1, len(stim_idx)), float(rate_hz))
    else:
        rates = np.atleast_2d(np.asarray(rate_hz, np.float64))
    silenced = np.zeros(con.n, np.bool_)
    if silence is not None:
        silenced[np.asarray(silence, np.int64)] = True
    rec_idx = np.zeros(0, np.int64) if rec_idx is None else np.asarray(rec_idx, np.int64)
    n_steps = int(round(t_ms / p["dt"]))
    bin_steps = int(round(bin_ms / p["dt"]))
    ptuple = (p["v_0"], p["v_th"], p["t_mbr"], p["tau"], p["t_rfc"], p["t_dly"],
              p["w_syn"], p["f_poi"], p["dt"])
    w = con.weights if weights is None else np.asarray(weights, np.float32)
    return _run(con.indptr, con.indices, w.astype(np.float32), con.n, n_steps, stim_idx, rates,
                silenced, rec_idx, bin_steps, seed, ptuple)
