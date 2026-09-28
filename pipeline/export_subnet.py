"""
Step 3 - write the sub-circuit for the game (game/data/flypong_circuit.js) and for
pong_sim.py (work/flypong_circuit.json).

usage: python pipeline/export_subnet.py [--dataset flywire|malecns]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import flysim  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="flywire")
args = ap.parse_args()
wd = config.WORK / args.dataset

con = config.load(args.dataset)
m = con.meta
sub = np.load(wd / "sub_idx.npy")
rf = pd.read_csv(wd / "lc10a_rf.csv")
vox = np.asarray(getattr(con, "voxel_nm", (4, 4, 40)), float) / 1000.0

pos = -np.ones(con.n, np.int64); pos[sub] = np.arange(len(sub))
pre_of = np.repeat(np.arange(con.n), np.diff(con.indptr))
keep = (pos[pre_of] >= 0) & (pos[con.indices] >= 0)
pre_l, post_l, w_l = pos[pre_of[keep]], pos[con.indices[keep]], con.weights[keep]
order = np.lexsort((post_l, pre_l))
pre_l, post_l, w_l = pre_l[order], post_l[order], w_l[order]

um = lambda cols: m.loc[sub, cols].to_numpy(float) * vox  # noqa: E731
anchor, soma = um(["pos_x", "pos_y", "pos_z"]), um(["soma_x", "soma_y", "soma_z"])
xy = np.where(np.isnan(soma), anchor, soma)[:, :2]      # view from behind: x (fly's left -> right), y (dorsal -> ventral)

ct = m.cell_type.fillna("untyped").values[sub].astype(str)
sc = m.super_class.fillna("unknown").values[sub].astype(str)
role = np.array(["other"] * len(sub), dtype=object)
role[sc == "descending"] = "dn"
role[np.isin(ct, ["AOTU019", "AOTU025"])] = "aotu"
role[np.isin(ct, ["DNa02", "DNa01"])] = "steer"
role[ct == "LC10a"] = "lc10a"

rf_local = pos[rf.idx.to_numpy()]
xy[rf_local] = rf[["cx", "cy"]].to_numpy()               # LC10a drawn at their dendrites (retinotopic)

neurons = pd.DataFrame(dict(
    i=np.arange(len(sub)), root_id=con.root_ids[sub].astype(str), type=ct,
    side=m.side.fillna("unknown").values[sub].astype(str), cls=sc,
    nt=m.top_nt.fillna("unknown").values[sub].astype(str), x=xy[:, 0].round(1), y=xy[:, 1].round(1), role=role,
))

# brain silhouette: density of every neuron's anchor point in the loaded dataset
P = m[["pos_x", "pos_y"]].to_numpy(float) * vox[:2]
P = P[~np.isnan(P).any(1)]
xr, yr = (P[:, 0].min() - 10, P[:, 0].max() + 10), (P[:, 1].min() - 10, P[:, 1].max() + 10)
H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=(120, 64), range=(xr, yr))
H = np.log1p(H); H = (255 * H / H.max()).astype(int)

data = dict(
    source=getattr(con, "source", "FlyWire FAFB v783 (female whole brain); connectivity and transmitter signs as used by Shiu et al. 2024"),
    n=int(len(sub)),
    neurons={c: neurons[c].tolist() for c in neurons.columns},
    edges=dict(pre=pre_l.tolist(), post=post_l.tolist(), w=w_l.astype(int).tolist()),
    lc10a=dict(i=rf_local.tolist(), azimuth=rf.azimuth.round(1).tolist(),
               elevation=rf.elevation.round(1).tolist(), side=rf.side.tolist()),
    steer={f"{t}_{s[0].upper()}": int(pos[con.by_type(t, s)[0]]) for t in ["DNa02", "DNa01"] for s in ["left", "right"]},
    aotu={f"{t}_{s[0].upper()}": int(pos[con.by_type(t, s)[0]]) for t in ["AOTU019", "AOTU025"] for s in ["left", "right"]},
    silhouette=dict(x0=xr[0], x1=xr[1], y0=yr[0], y1=yr[1], nx=120, ny=64, v=H.T.flatten().tolist()),
    params=flysim.PARAMS,
)
blob = json.dumps(data, separators=(",", ":"), allow_nan=False)
config.WORK.mkdir(parents=True, exist_ok=True)
config.GAME_DATA.mkdir(parents=True, exist_ok=True)
(config.WORK / "flypong_circuit.json").write_text(blob)
(config.GAME_DATA / "flypong_circuit.js").write_text("window.FLYPONG_CIRCUIT=" + blob + ";\n")
print(f"{len(sub)} neurons, {len(pre_l)} connections -> {config.GAME_DATA / 'flypong_circuit.js'} ({len(blob)/1e6:.2f} MB)")
print(neurons.role.value_counts().to_dict())
