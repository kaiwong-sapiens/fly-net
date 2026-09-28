"""
Return rates quoted in the README: courting fly, naive fly, motionless paddle, and a naive
fly after 2 minutes of coaching (evaluated with learning off). 3 seeds x 5 minutes each.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pong_sim import Brain, PongConfig, FlyPong, DopamineLearner  # noqa: E402


def evaluate(b, gain, seconds, seed):
    cfg = PongConfig(gain_hz=gain, K=0.045, tau_dn=25.0)
    g = FlyPong(b, cfg, seed=seed); h = m = 0
    for _ in range(int(seconds * 1000 / cfg.frame_ms)):
        ev = g.step()
        h += ev == "hit"; m += ev == "miss"
    return h, m


def job(args):
    kind, seed = args
    b = Brain(seed=0); b.reset_state(seed)
    if kind == "coached":
        cfg = PongConfig(gain_hz=45.0, K=0.045, tau_dn=25.0)
        L = DopamineLearner(b, eta=0.0, mode="coach", eta_coach=0.01)
        g = FlyPong(b, cfg, seed=seed + 100)
        for _ in range(int(120 * 1000 / cfg.frame_ms)):
            ev = g.step(); L.after_frame(g, ev)
        return kind, evaluate(b, 45.0, 300, seed + 900)
    gain = {"courting": 120.0, "naive": 45.0, "motionless": 0.0}[kind]
    return kind, evaluate(b, gain, 300, seed + 50)


if __name__ == "__main__":
    jobs = [(k, s) for k in ["courting", "naive", "motionless", "coached"] for s in [11, 12, 13]]
    res = {}
    with Pool(2) as p:
        for kind, (h, m) in p.imap(job, jobs):
            res.setdefault(kind, []).append((h, m))
    for k, v in res.items():
        h = sum(x[0] for x in v); t = sum(x[0] + x[1] for x in v)
        print(f"{k:10s} {h}/{t} = {h / t:.2f}   per run {[round(x[0] / (x[0] + x[1]), 2) for x in v]}")
