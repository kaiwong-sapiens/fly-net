"""
Closed-loop Fly Pong in Python (reference implementation for the browser game).

ball -> LC10a receptive fields -> spiking sub-circuit (Shiu et al. LIF) -> DNa02/DNa01 -> paddle
Optional dopamine-gated plasticity on LC10a output synapses (three-factor rule).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

CIRCUIT = config.WORK / "flypong_circuit.json"


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def _block(indptr, indices, weff, u, g, ref_until, buf, t0, n_steps, stim_idx, rates,
           is_stim, counts, am, c, as_, thr, kick, D, R, dt, wsyn):
    n = u.shape[0]
    spk = np.empty(n, np.int32)
    for t in range(t0, t0 + n_steps):
        for i in range(n):
            if t < ref_until[i]:
                continue
            gi = g[i]
            if gi != 0.0 or u[i] != 0.0:
                u[i] = u[i] * am + gi * c
                g[i] = gi * as_
        ns = 0
        for i in range(n):
            if u[i] > thr and t >= ref_until[i]:
                spk[ns] = i
                ns += 1
                ref_until[i] = t + (1 if is_stim[i] else R)
        slot = t % D
        row = buf[slot]
        for i in range(n):
            if row[i] != 0.0:
                if t >= ref_until[i]:
                    g[i] += row[i]
                row[i] = 0.0
        for k in range(stim_idx.shape[0]):
            r = rates[k]
            if r > 0.0 and np.random.random() < r * dt * 1e-3:
                u[stim_idx[k]] += kick
        for s in range(ns):
            i = spk[s]
            u[i] = 0.0
            g[i] = 0.0
            counts[i] += 1
            for e in range(indptr[i], indptr[i + 1]):
                buf[slot, indices[e]] += weff[e] * wsyn
    return t0 + n_steps


class Brain:
    def __init__(self, path=CIRCUIT, seed=0):
        d = json.loads(Path(path).read_text())
        self.d = d
        p = d["params"]
        self.n = n = d["n"]
        pre = np.array(d["edges"]["pre"], np.int64)
        post = np.array(d["edges"]["post"], np.int32)
        w = np.array(d["edges"]["w"], np.float32)
        self.indptr = np.zeros(n + 1, np.int64)
        np.add.at(self.indptr, pre + 1, 1)
        self.indptr = np.cumsum(self.indptr)
        self.pre, self.indices, self.w0 = pre, post, w
        self.weff = w.copy()
        self.dt = p["dt"]
        self.am = math.exp(-self.dt / p["t_mbr"]); self.as_ = math.exp(-self.dt / p["tau"])
        self.c = p["tau"] / (p["tau"] - p["t_mbr"]) * (self.as_ - self.am)
        self.thr = p["v_th"] - p["v_0"]; self.kick = p["f_poi"] * p["w_syn"]
        self.D = int(round(p["t_dly"] / self.dt)); self.R = int(round(p["t_rfc"] / self.dt))
        self.wsyn = p["w_syn"]
        lc = d["lc10a"]
        self.lc_idx = np.array(lc["i"], np.int64)
        self.lc_az = np.radians(np.array(lc["azimuth"])); self.lc_el = np.radians(np.array(lc["elevation"]))
        self.is_stim = np.zeros(n, np.bool_); self.is_stim[self.lc_idx] = True
        self.steer = d["steer"]
        # plastic synapses: LC10a -> anything
        self.plastic = np.flatnonzero(self.is_stim[pre])
        self.mult = np.ones(len(self.plastic), np.float32)
        self.elig = np.zeros(len(self.plastic), np.float32)
        self.reset_state(seed)

    def reset_state(self, seed=0):
        n = self.n
        self.u = np.zeros(n, np.float32); self.g = np.zeros(n, np.float32)
        self.ref_until = np.zeros(n, np.int64); self.buf = np.zeros((self.D, n), np.float32)
        self.t = 0
        _seed(seed)

    def set_mult(self, m):
        self.mult[:] = m
        self.weff[self.plastic] = self.w0[self.plastic] * self.mult

    def lc_rates(self, az_deg, el_deg, size_deg, gain, sigma_rf=16.0):
        baz, bel = math.radians(az_deg), math.radians(el_deg)
        cosd = np.sin(self.lc_el) * math.sin(bel) + np.cos(self.lc_el) * math.cos(bel) * np.cos(self.lc_az - baz)
        dd = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
        s2 = sigma_rf ** 2 + (size_deg / 2) ** 2
        return gain * np.exp(-dd ** 2 / (2 * s2))

    def run(self, ms, rates):
        counts = np.zeros(self.n, np.int32)
        steps = int(round(ms / self.dt))
        self.t = _block(self.indptr, self.indices, self.weff, self.u, self.g, self.ref_until, self.buf,
                        self.t, steps, self.lc_idx, rates.astype(np.float64), self.is_stim, counts,
                        self.am, self.c, self.as_, self.thr, self.kick, self.D, self.R, self.dt, self.wsyn)
        return counts


@dataclass
class PongConfig:
    W: float = 1.0
    H: float = 1.4
    paddle_half: float = 0.09
    ball_r: float = 0.03
    speed0: float = 0.75          # ball speed (field heights / s)
    speed_up: float = 1.03
    speed_max: float = 1.4
    gain_hz: float = 120.0        # LC10a peak rate (arousal)
    K: float = 0.016              # paddle speed (W/s) per Hz of steering difference
    v_max: float = 2.0
    tau_dn: float = 40.0          # ms, smoothing of steering-neuron rates
    beta01: float = 0.5           # weight of DNa01 relative to DNa02
    frame_ms: float = 10.0


class FlyPong:
    """Fly at the bottom (y = H), facing up. Opponent = perfect wall at the top returning random angles."""

    def __init__(self, brain: Brain, cfg: PongConfig, seed=0):
        self.b, self.cfg = brain, cfg
        self.rng = np.random.default_rng(seed)
        self.px = cfg.W / 2
        self.rate = np.zeros(4)
        self.serve()

    def serve(self):
        c = self.cfg
        self.speed = c.speed0
        self.bx, self.by = self.rng.uniform(0.2, 0.8) * c.W, 0.1
        ang = self.rng.uniform(-0.6, 0.6)
        self.vx, self.vy = math.sin(ang), math.cos(ang)

    def view(self):
        c = self.cfg
        dx, dy = self.bx - self.px, (c.H - 0.02) - self.by
        az = math.degrees(math.atan2(dx, max(dy, 1e-3)))
        dist = math.hypot(dx, dy)
        size = math.degrees(2 * math.atan(c.ball_r / max(dist, 1e-3)))
        return az, size

    def step(self):
        """Advance one frame. Returns 'hit', 'miss' or None."""
        c, b = self.cfg, self.b
        az, size = self.view()
        counts = b.run(c.frame_ms, b.lc_rates(az, 0.0, size, c.gain_hz))
        s = b.steer
        raw = np.array([counts[s["DNa02_L"]], counts[s["DNa02_R"]], counts[s["DNa01_L"]], counts[s["DNa01_R"]]]) * (1000.0 / c.frame_ms)
        a = math.exp(-c.frame_ms / c.tau_dn)
        self.rate = a * self.rate + (1 - a) * raw
        turn = (self.rate[1] - self.rate[0]) + c.beta01 * (self.rate[3] - self.rate[2])
        v = float(np.clip(c.K * turn, -c.v_max, c.v_max))
        dt = c.frame_ms / 1000.0
        self.px = float(np.clip(self.px + v * dt, c.paddle_half, c.W - c.paddle_half))
        self.last_counts, self.last_v = counts, v
        # ball
        sp = self.speed * c.H
        self.bx += self.vx * sp * dt; self.by += self.vy * sp * dt
        if self.bx < c.ball_r: self.bx, self.vx = c.ball_r, abs(self.vx)
        if self.bx > c.W - c.ball_r: self.bx, self.vx = c.W - c.ball_r, -abs(self.vx)
        if self.by < 0.05 and self.vy < 0:            # opponent wall returns
            ang = self.rng.uniform(-0.7, 0.7)
            self.vx, self.vy = math.sin(ang), math.cos(ang)
        if self.vy > 0 and self.by >= c.H - 0.05:
            if abs(self.bx - self.px) <= c.paddle_half + c.ball_r:
                off = (self.bx - self.px) / c.paddle_half
                ang = 0.6 * off
                self.vx, self.vy = math.sin(ang), -math.cos(ang)
                self.speed = min(self.speed * c.speed_up, c.speed_max)
                return "hit"
            self.serve()
            return "miss"
        return None


def play(brain, cfg, seconds=60.0, seed=0, learner=None):
    game = FlyPong(brain, cfg, seed=seed)
    hits = misses = 0
    log = []
    for f in range(int(seconds * 1000 / cfg.frame_ms)):
        ev = game.step()
        if learner is not None:
            learner.after_frame(game, ev)
        if ev == "hit": hits += 1
        elif ev == "miss": misses += 1
        if ev: log.append(ev)
    return hits, misses, log


class DopamineLearner:
    """
    Three-factor rule on LC10a output synapses (the mushroom body's form of plasticity,
    applied here as a modelling assumption):
        eligibility  e <- e * exp(-dt/tau_e) + (pre spikes x post spikes in this frame)
        dopamine     DA = reward - expected reward   (reward prediction error)
        update       w_mult += eta * DA * e,   clipped to [w_min, w_max]
    mode 'sparse': reward +1 on a hit ("sugar"), -1 on a miss ("heat").
    mode 'coach' : additionally, every frame, sugar/heat in proportion to whether the
                   paddle is moving toward or away from the ball.
    """

    def __init__(self, brain, eta=0.02, tau_e=800.0, mode="sparse", w_min=0.0, w_max=3.0,
                 coach_gain=0.02, baseline_alpha=0.1, eta_coach=None):
        self.b, self.eta, self.tau_e, self.mode = brain, eta, tau_e, mode
        self.w_min, self.w_max, self.coach_gain = w_min, w_max, coach_gain
        self.pp = brain.pre[brain.plastic]
        self.qq = brain.indices[brain.plastic]
        self.rbar = 0.0
        self.alpha = baseline_alpha
        self.eta_coach = eta_coach if eta_coach is not None else eta * coach_gain
        self.da_log = []

    def apply(self, da, eta=None):
        b = self.b
        b.mult += (self.eta if eta is None else eta) * da * b.elig
        np.clip(b.mult, self.w_min, self.w_max, out=b.mult)
        b.weff[b.plastic] = b.w0[b.plastic] * b.mult

    def after_frame(self, game, ev):
        b, c = self.b, game.last_counts
        b.elig *= math.exp(-game.cfg.frame_ms / self.tau_e)
        b.elig += c[self.pp].astype(np.float32) * c[self.qq].astype(np.float32)
        if ev is not None:
            r = 1.0 if ev == "hit" else -1.0
            da = r - self.rbar
            self.rbar += self.alpha * (r - self.rbar)
            self.apply(da)
            self.da_log.append(da)
        if self.mode == "coach":
            toward = math.copysign(1.0, game.bx - game.px) * game.last_v / game.cfg.v_max
            self.apply(toward, eta=self.eta_coach)
