"""
Compare reward schemes on a naive fly (eye gain 45/s): none, rally reward only (sugar per
return, heat per miss), and the coach (continuous sugar/heat by steering direction).
Prints the return rate per minute over 6 minutes of training.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pong_sim import Brain, PongConfig, FlyPong, DopamineLearner  # noqa: E402


def run(args):
    mode, eta, eta_c = args
    b = Brain(seed=0); b.reset_state(1)
    cfg = PongConfig(gain_hz=45.0, K=0.045, tau_dn=25.0)
    L = None if mode == "none" else DopamineLearner(b, eta=eta, mode=mode, eta_coach=eta_c)
    g = FlyPong(b, cfg, seed=101); out = []; h = m = 0
    for f in range(int(360 * 1000 / cfg.frame_ms)):
        ev = g.step()
        if L: L.after_frame(g, ev)
        h += ev == "hit"; m += ev == "miss"
        if (f + 1) % 6000 == 0:
            out.append(round(h / max(1, h + m), 2)); h = m = 0
    return mode, eta, eta_c, out, float(b.mult.mean())


if __name__ == "__main__":
    jobs = [("none", 0, 0), ("sparse", 0.01, 0), ("coach", 0.0, 0.01)]
    with Pool(2) as p:
        for mode, eta, eta_c, out, mm in p.imap(run, jobs):
            print(f"{mode:6s}  return rate per minute: {out}   mean synapse multiplier {mm:.2f}")
