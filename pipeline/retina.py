"""
Give every LC10a neuron a receptive-field direction, derived from the connectome.

1. Dendritic position: each LC10a's input-weighted centroid of its presynaptic
   optic-lobe (columnar) neurons' anchor points. Columnar neurons sit in one eye
   column each, so this places the LC10a dendrite on the retinotopic sheet.
2. Orientation: Collie et al. 2026 (Neuron, Wilson lab) report that AOTU019 serves
   the central visual field and AOTU025 the periphery. Each LC10a's preference for
   AOTU019 vs AOTU025 correlates r~0.9 with one anatomical axis of that sheet, so
   that axis is azimuth, oriented so the AOTU019 end faces forward. (This agrees with
   the anterior-posterior inversion at the outer optic chiasm.) The orthogonal axis
   in the sheet is elevation (dorsal = up).
3. Degrees: ranks along each axis are mapped onto an assumed field of view.
   The exact extents are approximations, not measurements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# assumed field of view of one eye (degrees; + = toward that eye's side)
AZ_FRONT, AZ_BACK = -15.0, 150.0      # ~30 deg binocular overlap in front
EL_LOW, EL_HIGH = -60.0, 70.0


def lc10a_receptive_fields(con, cell_type="LC10a", columnar=None, centroids=None, dorsal=(0.0, -1.0, 0.0)):
    """
    columnar  : bool mask of neurons whose positions mark eye columns (default: FlyWire super_class 'optic')
    centroids : optional {model index: (x, y, z) in um} dendrite centroids, e.g. from synapse locations
    dorsal    : direction of 'up' in the dataset's coordinates (FlyWire: -y)
    """
    m = con.meta
    vox = np.asarray(getattr(con, "voxel_nm", (4, 4, 40)), float) / 1000.0
    P = m[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * vox  # um
    pre_of = np.repeat(np.arange(con.n), np.diff(con.indptr))
    optic = (m.super_class.values == "optic") if columnar is None else np.asarray(columnar, bool)
    out = []
    for side, sign in (("right", 1.0), ("left", -1.0)):
        lc = con.by_type(cell_type, side)
        a19 = con.by_type("AOTU019", side)
        a25 = con.by_type("AOTU025", side)
        mask = np.isin(con.indices, lc) & optic[pre_of]
        pre, post, w = pre_of[mask], con.indices[mask], np.abs(con.weights[mask]).astype(float)
        cen = np.zeros((len(lc), 3))
        pref = np.zeros(len(lc))
        for k, i in enumerate(lc):
            if centroids is not None:
                cen[k] = centroids[int(i)]
            else:
                sel = post == i
                cen[k] = (P[pre[sel]] * w[sel, None]).sum(0) / w[sel].sum()
            s0, s1 = con.indptr[i], con.indptr[i + 1]
            tg, ww = con.indices[s0:s1], con.weights[s0:s1]
            t19, t25 = ww[np.isin(tg, a19)].sum(), ww[np.isin(tg, a25)].sum()
            pref[k] = (t19 - t25) / (t19 + t25) if (t19 + t25) > 0 else np.nan
        c0 = cen - cen.mean(0)
        ok = ~np.isnan(pref)
        # azimuth axis = 3-D direction that best predicts AOTU019-preference (central view)
        beta, *_ = np.linalg.lstsq(c0[ok], pref[ok] - pref[ok].mean(), rcond=None)
        az_dir = beta / np.linalg.norm(beta)
        # elevation axis = dorsal direction (-y), orthogonalised against azimuth
        up = np.asarray(dorsal, float)
        el_dir = up - (up @ az_dir) * az_dir
        el_dir /= np.linalg.norm(el_dir)
        az_score, el_score = c0 @ az_dir, c0 @ el_dir
        r_pref = np.corrcoef(az_score[ok], pref[ok])[0, 1]
        # rank -> degrees (central end of the axis = frontal)
        az_rank = pd.Series(az_score).rank(pct=True).to_numpy()      # 1 = most central
        el_rank = pd.Series(el_score).rank(pct=True).to_numpy()      # 1 = most dorsal
        az = AZ_BACK + (AZ_FRONT - AZ_BACK) * az_rank                # eye-centred, + = lateral
        el = EL_LOW + (EL_HIGH - EL_LOW) * el_rank
        for k, i in enumerate(lc):
            out.append(dict(idx=int(i), root_id=int(con.root_ids[i]), side=side,
                            azimuth=float(sign * az[k]), elevation=float(el[k]),
                            aotu019_pref=float(pref[k]) if ok[k] else None,
                            cx=float(cen[k, 0]), cy=float(cen[k, 1]), cz=float(cen[k, 2])))
        print(f"{cell_type} {side}: n={len(lc)}  corr(azimuth axis, AOTU019-pref) = {r_pref:.2f}")
    return pd.DataFrame(out)


def lc10a_rates(rf, ball_az, ball_el, ball_size_deg, gain_hz=120.0, sigma_rf=16.0):
    """Poisson rate for each LC10a given a ball direction (deg, fly-centred, + = right)."""
    az = np.radians(rf["azimuth"].to_numpy()); el = np.radians(rf["elevation"].to_numpy())
    baz, bel = np.radians(ball_az), np.radians(ball_el)
    cosd = np.sin(el) * np.sin(bel) + np.cos(el) * np.cos(bel) * np.cos(az - baz)
    d = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
    s2 = sigma_rf ** 2 + (ball_size_deg / 2) ** 2
    return gain_hz * np.exp(-d ** 2 / (2 * s2))
