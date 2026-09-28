"""
Load the male central nervous system connectome (Janelia / Google / Cambridge, Cell 2026)
from neuPrint into the same structure as the FlyWire loader, so the Fly Pong pipeline can
run on the male brain.

UNTESTED: neuPrint was not reachable from the sandbox this was written in. Property names
below follow neuprint-python conventions and the male-CNS preprint; check them against
`client.fetch_datasets()` and one `fetch_neurons()` call before trusting the output.

What it fetches (not the whole 166k-neuron CNS, which the model does not need):
  * every neuron within HOPS synaptic steps downstream of LC10a (>= MIN_W synapses per step),
    plus LC10a, AOTU019/025, DNa01/02, and all connections among them;
  * LC10a input-synapse locations, to place each LC10a's dendrite on the eye map directly.

setup:  pip install neuprint-python ; export NEUPRINT_TOKEN=...   (token: neuprint.janelia.org -> Account)
        export MALECNS_DATASET=male-cns:v0.9   (or whatever fetch_datasets() lists)
then:   python pipeline/build_subnet.py --dataset malecns   (and validate/export as for FlyWire)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import flysim

SERVER = os.environ.get("NEUPRINT_SERVER", "neuprint.janelia.org")
DATASET = os.environ.get("MALECNS_DATASET", "male-cns:v0.9")
HOPS = int(os.environ.get("MALECNS_HOPS", "3"))
MIN_W = int(os.environ.get("MALECNS_MIN_WEIGHT", "5"))
VOXEL_NM = 8.0
INHIBITORY = {"gaba", "glutamate"}      # same sign rule as the FlyWire model (Shiu et al. 2024)
KEEP_TYPES = ["LC10a", "AOTU019", "AOTU025", "DNa01", "DNa02"]
COLUMNAR = r"^(Tm|TmY|Li|Y|T2|T3|Mi)\d"   # eye-column neuron types (used for sanity checks)


def _client():
    from neuprint import Client
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise SystemExit("Set NEUPRINT_TOKEN (neuprint.janelia.org -> Account -> Auth token).")
    return Client(SERVER, dataset=DATASET, token=token)


def _first(df, names, default=np.nan):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index)


def load(cache: Path | None = None) -> flysim.Connectome:
    from neuprint import NeuronCriteria as NC, fetch_adjacencies, fetch_neurons, fetch_synapse_connections
    cache = Path(cache) if cache else None
    if cache and (cache / "neurons.parquet").exists():
        neurons = pd.read_parquet(cache / "neurons.parquet")
        conn = pd.read_parquet(cache / "conn.parquet")
        syn = pd.read_parquet(cache / "lc10a_inputs.parquet")
    else:
        _client()
        seed, _ = fetch_neurons(NC(type=KEEP_TYPES))
        ids = set(seed.bodyId)
        frontier = set(seed.bodyId[seed.type == "LC10a"])
        for hop in range(HOPS):                       # expand downstream of LC10a
            _, c = fetch_adjacencies(NC(bodyId=sorted(frontier)), None, min_total_weight=MIN_W)
            new = set(c.bodyId_post) - ids
            print(f"hop {hop + 1}: +{len(new)} neurons")
            ids |= new
            frontier = new
        neurons, _ = fetch_neurons(NC(bodyId=sorted(ids)))
        _, conn = fetch_adjacencies(NC(bodyId=sorted(ids)), NC(bodyId=sorted(ids)), min_total_weight=1)
        conn = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False).weight.sum()
        syn = fetch_synapse_connections(None, NC(type="LC10a"))
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            neurons.to_parquet(cache / "neurons.parquet"); conn.to_parquet(cache / "conn.parquet")
            syn.to_parquet(cache / "lc10a_inputs.parquet")

    neurons = neurons.drop_duplicates("bodyId").reset_index(drop=True)
    idx = pd.Series(np.arange(len(neurons)), index=neurons.bodyId.values)
    n = len(neurons)
    nt = _first(neurons, ["consensusNt", "predictedNt", "celltypePredictedNt"]).fillna("unknown").astype(str).str.lower()
    sign = np.where(nt.isin(INHIBITORY), -1.0, 1.0)
    pre = idx[conn.bodyId_pre.values].to_numpy(); post = idx[conn.bodyId_post.values].to_numpy()
    w = conn.weight.to_numpy(float) * sign[pre]
    order = np.lexsort((post, pre)); pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(n + 1, np.int64); np.add.at(indptr, pre + 1, 1); indptr = np.cumsum(indptr)

    side = _first(neurons, ["somaSide", "rootSide"]).map({"L": "left", "R": "right"})
    inst_side = neurons.get("instance", pd.Series("", index=neurons.index)).astype(str).str.extract(r"_([LR])$")[0].map({"L": "left", "R": "right"})
    side = side.fillna(inst_side)
    soma = neurons.get("somaLocation", pd.Series([None] * n))
    sx = np.array([s[0] if isinstance(s, (list, tuple, np.ndarray)) else np.nan for s in soma], float)
    sy = np.array([s[1] if isinstance(s, (list, tuple, np.ndarray)) else np.nan for s in soma], float)
    sz = np.array([s[2] if isinstance(s, (list, tuple, np.ndarray)) else np.nan for s in soma], float)
    superclass = _first(neurons, ["superclass", "class"]).fillna("unknown").astype(str)
    superclass = superclass.where(~superclass.str.contains("descending", case=False), "descending")
    meta = pd.DataFrame(dict(root_id=neurons.bodyId.values, cell_type=neurons.type.values, side=side.values,
                             super_class=superclass.values, top_nt=nt.values,
                             pos_x=sx, pos_y=sy, pos_z=sz, soma_x=sx, soma_y=sy, soma_z=sz))

    con = flysim.Connectome(n=n, indptr=indptr, indices=post.astype(np.int32), weights=w.astype(np.float32),
                            root_ids=neurons.bodyId.to_numpy(), meta=meta, voxel_nm=(VOXEL_NM,) * 3)
    # LC10a dendrite centroids straight from their input-synapse locations in the lobula (in um)
    if "roi_post" in syn.columns:
        syn = syn[syn.roi_post.astype(str).str.startswith("LO")]
    xyz = syn[["x_post", "y_post", "z_post"]].to_numpy(float) * VOXEL_NM / 1000.0
    cen = pd.DataFrame(xyz, columns=list("xyz")).groupby(syn.bodyId_post.values).mean()
    centroids = {int(idx[b]): cen.loc[b].to_numpy() for b in cen.index if b in idx.index}
    # Which axis is dorsal in neuPrint's male-CNS space is not verified here; -y matches FlyWire and the hemibrain.
    con.retina_kwargs = dict(centroids=centroids, dorsal=(0.0, -1.0, 0.0))
    con.source = f"Male CNS connectome ({DATASET}, neuPrint), {HOPS} hops downstream of LC10a, min {MIN_W} synapses per hop"
    return con
