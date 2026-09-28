"""
How much the steering readout's modelled choices matter:
turn = (R DNa02 - L DNa02) + beta01 * (R DNa01 - L DNa01), smoothed with tau_dn,
paddle speed = K * turn (capped at v_max).

Varies one choice at a time around the game's values (K 0.045, beta01 0.5, tau_dn 25 ms)
and measures the return rate against a wall that sends back random angles,
for the naive (eye gain 45/s) and courting (120/s) fly. 3 seeds x 5 minutes each,
the same seeds as verify_stats.py, so the game's own row reproduces the README.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pong_sim import Brain, PongConfig, FlyPong  # noqa: E402

BASE = dict(K=0.045, beta01=0.5, tau_dn=25.0)
VARIANTS = [("game's values", {})] + \
    [(f"DNa01 weight {b:g}", {"beta01": b}) for b in (0.0, 1.0)] + \
    [(f"gain K {k:g}", {"K": k}) for k in (0.015, 0.03, 0.07, 0.1)] + \
    [(f"smoothing {t:g} ms", {"tau_dn": t}) for t in (10.0, 50.0)]
SEEDS, SECONDS = (11, 12, 13), 300


def job(args):
    label, over, gain, seed = args
    b = Brain(seed=0); b.reset_state(seed)
    cfg = PongConfig(gain_hz=gain, **{**BASE, **over})
    g = FlyPong(b, cfg, seed=seed + 50); h = m = 0
    for _ in range(int(SECONDS * 1000 / cfg.frame_ms)):
        ev = g.step()
        h += ev == "hit"; m += ev == "miss"
    return label, gain, h, m


if __name__ == "__main__":
    jobs = [(lab, over, gain, s) for lab, over in VARIANTS for gain in (45.0, 120.0) for s in SEEDS]
    res = {}
    with Pool(2) as p:
        for lab, gain, h, m in p.imap(job, jobs):
            res.setdefault((lab, gain), []).append((h, m))
    print(f"return rate, {len(SEEDS)} runs x {SECONDS // 60} min each: pooled (per-run min-max)")
    print(f"{'readout':<22}{'naive (45/s)':>18}{'courting (120/s)':>20}")
    for lab, _ in VARIANTS:
        cells = []
        for gain in (45.0, 120.0):
            v = res[(lab, gain)]; r = [h / (h + m) for h, m in v]
            pooled = sum(h for h, _ in v) / sum(h + m for h, m in v)
            cells.append(f"{100 * pooled:.0f}% ({100 * min(r):.0f}-{100 * max(r):.0f})")
        print(f"{lab:<22}{cells[0]:>18}{cells[1]:>20}")
