"""
Check flysim.py against the original Brian2 model of Shiu et al. 2024.

1. Always: stimulate 21 right-hemisphere sugar-sensing neurons at 100 Hz on FlyWire v630
   (the published example) and compare firing rates with the Brian2 results shipped in the
   Drosophila_brain_model repository (results/example/sugarR_100Hz.parquet, 30 trials).
2. With --brian2 PYTHON: also run the original Brian2 equations on the active sub-network
   with that interpreter (Brian2 needs numpy<2, so use a separate venv) and compare.

Brian2 detail that matters: synaptic input addressed to a neuron during its 2.2 ms
refractory period is discarded (Brian2 does this automatically for variables flagged
"unless refractory"); flysim.py copies that behaviour.
"""
import argparse
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import flysim  # noqa: E402

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
         720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
         720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]

BRIAN2_SCRIPT = r'''
import sys, pickle, numpy as np
sys.path.insert(0, sys.argv[2])
from model import default_params as P
from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network, mV, ms, Hz, prefs
prefs.codegen.target = "numpy"
d = pickle.load(open(sys.argv[1], "rb"))
P = dict(P); P["r_poi"] = d["rate"] * Hz
counts = np.zeros(d["n"])
for r in range(d["reps"]):
    neu = NeuronGroup(N=d["n"], model=P["eqs"], method="linear", threshold=P["eq_th"], reset=P["eq_rst"], refractory="rfc", namespace=P)
    neu.v = P["v_0"]; neu.g = 0; neu.rfc = P["t_rfc"]
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=P["t_dly"])
    syn.connect(i=d["pre"], j=d["post"]); syn.w = d["w"] * P["w_syn"]
    pois = []
    for i in d["stim"]:
        pois.append(PoissonInput(target=neu[i], target_var="v", N=1, rate=P["r_poi"], weight=P["w_syn"] * P["f_poi"]))
        neu[i].rfc = 0 * ms
    mon = SpikeMonitor(neu)
    Network(neu, syn, mon, *pois).run(1000 * ms)
    counts += np.bincount(np.asarray(mon.i), minlength=d["n"])
np.save(sys.argv[3], counts / d["reps"])
'''

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--brian2", help="python interpreter with brian2 + numpy<2 installed")
args = ap.parse_args()

con = flysim.load_flywire(config.DATA, "630")
stim = con.idx(SUGAR)
ref = pd.read_parquet(config.DATA / "Drosophila_brain_model/results/example/sugarR_100Hz.parquet")
ref_rate = ref.groupby("flywire_id").size() / ref.trial.nunique()

counts = np.zeros(con.n)
for s in range(args.reps):
    c, _ = flysim.run(con, stim, rate_hz=100.0, t_ms=1000, seed=100 + s)
    counts += c
ours = pd.Series(counts / args.reps, index=con.root_ids)
j = pd.concat([ref_rate.rename("brian2_published"), ours[ours > 0].rename("flysim")], axis=1).fillna(0)
ns = j.drop(index=[i for i in SUGAR if i in j.index])
print(f"active neurons: published {int((j.brian2_published > 0).sum())}, flysim {int((j.flysim > 0).sum())}")
print(f"non-stimulated neurons: r = {np.corrcoef(ns.brian2_published, ns.flysim)[0, 1]:.4f}, "
      f"mean |diff| = {np.abs(ns.brian2_published - ns.flysim).mean():.2f} Hz")

if args.brian2:
    sub = np.flatnonzero(counts > 0)
    sub = np.union1d(sub, con.idx([i for i in ref_rate.index if i in set(con.root_ids)]))
    p = -np.ones(con.n, np.int64); p[sub] = np.arange(len(sub))
    pre_of = np.repeat(np.arange(con.n), np.diff(con.indptr))
    keep = (p[pre_of] >= 0) & (p[con.indices] >= 0)
    d = dict(n=len(sub), pre=p[pre_of[keep]], post=p[con.indices[keep]], w=con.weights[keep].astype(float),
             stim=p[stim], rate=100.0, reps=args.reps)
    with tempfile.TemporaryDirectory() as td:
        pk, sc, outp = Path(td) / "net.pkl", Path(td) / "b2.py", Path(td) / "b2.npy"
        pickle.dump(d, open(pk, "wb")); sc.write_text(BRIAN2_SCRIPT)
        subprocess.run([args.brian2, str(sc), str(pk), str(config.DATA / "Drosophila_brain_model"), str(outp)], check=True)
        b2 = np.load(outp)
    mine = counts[sub] / args.reps
    nsl = np.setdiff1d(np.arange(len(sub)), d["stim"])
    print(f"Brian2 on the same sub-network: r = {np.corrcoef(b2[nsl], mine[nsl])[0, 1]:.4f}, "
          f"mean |diff| = {np.abs(b2 - mine)[nsl].mean():.2f} Hz")
