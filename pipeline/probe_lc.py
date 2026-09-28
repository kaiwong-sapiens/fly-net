"""
Which visual projection neuron types drive which descending neurons?

Stimulates each LC/LPLC type on one side (100 Hz Poisson, 500 ms, 3 seeds) in the whole
brain and lists the most active descending neurons. This is how the LC10a -> DNa02
pursuit pathway (and LPLC2 -> giant fibre DNp01) showed up.

usage: python pipeline/probe_lc.py [--dataset flywire|malecns]
"""
import argparse
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
con = config.load(args.dataset)
m = con.meta
dn = np.flatnonzero((m.super_class == "descending").values)
types = ["LC10a", "LC10b", "LC10c-1", "LC10d", "LC9", "LC11", "LC12", "LC15", "LC17", "LC18",
         "LC21", "LC22", "LPLC1", "LPLC2"]
rows = []
for ct in types:
    for side in ["left", "right"]:
        stim = con.by_type(ct, side)
        if len(stim) == 0:
            continue
        tot = np.zeros(con.n)
        for s in range(3):
            c, _ = flysim.run(con, stim, rate_hz=100.0, t_ms=500, seed=s)
            tot += c
        rate = tot / 3 / 0.5
        act = rate[dn]
        top = np.argsort(-act)[:6]
        rows.append(dict(type=ct, side=side, n_stim=len(stim), n_active=int((rate > 0).sum()),
                         top_DNs=", ".join(f"{m.cell_type.iloc[dn[i]]}_{str(m.side.iloc[dn[i]])[0].upper()}:{act[i]:.0f}"
                                           for i in top if act[i] > 1)))
    print(ct, flush=True)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 150)
print(pd.DataFrame(rows).to_string())
