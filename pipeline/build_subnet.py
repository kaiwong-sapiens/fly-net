"""
Step 1 - find the part of the brain Fly Pong can ever activate.

In this LIF model a neuron that never spikes has no effect on anything, so the set of
neurons that spike under *any* game condition, plus their mutual synapses, reproduces the
whole-brain dynamics exactly for those conditions. We sweep the game's whole input space
(ball direction x apparent size, at maximum eye gain) twice: with connectome weights, and
with every LC10a output synapse at the strongest value training can reach (x W_MAX).

usage: python pipeline/build_subnet.py [--dataset flywire|malecns]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import flysim  # noqa: E402
import retina  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="flywire")
args = ap.parse_args()
out = config.WORK / args.dataset
out.mkdir(parents=True, exist_ok=True)

con = config.load(args.dataset)
m = con.meta
rf_kwargs = getattr(con, "retina_kwargs", {})
rf = retina.lc10a_receptive_fields(con, **rf_kwargs)
stim = rf.idx.to_numpy()
is_lc = np.zeros(con.n, bool); is_lc[stim] = True
pre_of = np.repeat(np.arange(con.n), np.diff(con.indptr))
plastic = is_lc[pre_of]
w_trained = con.weights.copy(); w_trained[plastic] *= config.W_MAX
print("plastic synapses (LC10a outputs):", int(plastic.sum()))

active = np.zeros(con.n, np.int64)
t0 = time.time()
k = 0
for wset in (con.weights, w_trained):
    for size in (4.0, 15.0, 45.0):
        for az in np.arange(-95, 96, 5):
            rates = retina.lc10a_rates(rf, az, 0.0, size, gain_hz=config.GAIN_MAX)
            c, _ = flysim.run(con, stim, rate_hz=rates, t_ms=300, seed=k, weights=wset)
            active += c
            k += 1
    print(f"  sweep done ({time.time() - t0:.0f}s), neurons active so far: {(active > 0).sum()}", flush=True)

must = np.flatnonzero(m.cell_type.isin(["DNa02", "DNa01", "AOTU019", "AOTU025"]).values)
sub = np.union1d(np.flatnonzero(active > 0), np.union1d(stim, must))
print("sub-circuit neurons:", len(sub))
np.save(out / "sub_idx.npy", sub)
rf.to_csv(out / "lc10a_rf.csv", index=False)
