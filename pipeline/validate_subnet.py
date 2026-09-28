"""
Step 2 - check the sub-circuit against the whole brain.

Random, time-varying ball trajectories and random LC10a synapse multipliers (0.3-3x) are
fed to both the whole connectome and the extracted sub-circuit with the same random seed.
If the sub-circuit is complete, the steering neurons' spike trains are identical.

usage: python pipeline/validate_subnet.py [--dataset flywire|malecns] [--trials 6]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import flysim  # noqa: E402
import retina  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="flywire")
ap.add_argument("--trials", type=int, default=6)
args = ap.parse_args()
wd = config.WORK / args.dataset

con = config.load(args.dataset)
rf = pd.read_csv(wd / "lc10a_rf.csv")
sub = np.load(wd / "sub_idx.npy")
stim = rf.idx.to_numpy()
pos = -np.ones(con.n, np.int64); pos[sub] = np.arange(len(sub))
pre_of = np.repeat(np.arange(con.n), np.diff(con.indptr))
keep = (pos[pre_of] >= 0) & (pos[con.indices] >= 0)
pre_l, post_l = pos[pre_of[keep]], pos[con.indices[keep]]
order = np.argsort(pre_l, kind="stable")
indptr = np.zeros(len(sub) + 1, np.int64); np.add.at(indptr, pre_l + 1, 1); indptr = np.cumsum(indptr)


def subcon(wfull):
    return flysim.Connectome(n=len(sub), indptr=indptr, indices=post_l[order].astype(np.int32),
                             weights=wfull[keep][order].astype(np.float32), root_ids=con.root_ids[sub])


is_lc = np.zeros(con.n, bool); is_lc[stim] = True
plastic = is_lc[pre_of]
dn = np.array([con.by_type(t, s)[0] for t, s in [("DNa02", "left"), ("DNa02", "right"), ("DNa01", "left"), ("DNa01", "right")]])
rng = np.random.default_rng(42)
rows = []
for trial in range(args.trials):
    T = 1500; nb = T // 10
    az = 80 * np.tanh(np.cumsum(rng.normal(0, 6, nb)) / 60)
    size = 4 + 41 * (0.5 + 0.5 * np.sin(np.linspace(0, rng.uniform(2, 6), nb) + rng.uniform(0, 6)))
    gain = rng.uniform(60, 150)
    rates = np.stack([retina.lc10a_rates(rf, az[b], 0.0, size[b], gain_hz=gain) for b in range(nb)])
    mult = np.ones(len(con.weights), np.float32); mult[plastic] = rng.uniform(0.3, 3.0, plastic.sum())
    wfull = con.weights * mult
    cF, recF = flysim.run(con, stim, rate_hz=rates, t_ms=T, rec_idx=dn, bin_ms=10, seed=trial, weights=wfull)
    cS, recS = flysim.run(subcon(wfull), pos[stim], rate_hz=rates, t_ms=T, rec_idx=pos[dn], bin_ms=10, seed=trial)
    rows.append(dict(trial=trial, gain=round(gain), whole_brain_active=int((cF > 0).sum()),
                     spiking_outside_sub=len(np.setdiff1d(np.flatnonzero(cF > 0), sub)),
                     steering_spikes_whole=recF.sum(0).tolist(), steering_spikes_sub=recS.sum(0).tolist(),
                     identical=bool(np.array_equal(recF, recS))))
df = pd.DataFrame(rows)
print(df.to_string())
print("ALL IDENTICAL" if df.identical.all() else "MISMATCH")
