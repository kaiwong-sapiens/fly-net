"""Steering response vs ball direction and size (sub-circuit engine; 4 seeds x 0.5 s per point)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pong_sim import Brain  # noqa: E402

for size, gain in [(5.0, 120.0), (20.0, 120.0), (40.0, 120.0), (20.0, 45.0)]:
    b = Brain(seed=0); S = b.steer; A = b.d["aotu"]
    print(f"--- ball {size:.0f} deg wide, eye gain {gain:.0f}/s  [DNa02 L R | DNa01 L R | AOTU019 L R | AOTU025 L R]")
    for az in [-90, -70, -50, -30, -15, 0, 15, 30, 50, 70, 90]:
        tot = np.zeros(8)
        for seed in range(4):
            b.reset_state(seed)
            for _ in range(50):
                c = b.run(10.0, b.lc_rates(az, 0.0, size, gain))
                tot += [c[S["DNa02_L"]], c[S["DNa02_R"]], c[S["DNa01_L"]], c[S["DNa01_R"]],
                        c[A["AOTU019_L"]], c[A["AOTU019_R"]], c[A["AOTU025_L"]], c[A["AOTU025_R"]]]
        r = tot / 2.0
        turn = (r[1] - r[0]) + 0.5 * (r[3] - r[2])
        print(f"  az={az:4d}  {r[0]:5.1f} {r[1]:5.1f} | {r[2]:4.1f} {r[3]:4.1f} | {r[4]:4.0f} {r[5]:4.0f} | {r[6]:4.0f} {r[7]:4.0f}   turn={turn:+6.1f}", flush=True)
