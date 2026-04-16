# In[1]:
import os
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


@dataclass(frozen=True)
class ModelConfig:
    h: float
    beta_deg: float
    phi_deg: float
    c: float
    gamma_unsat: float
    gamma_sat: float
    gamma_w: float = 10.0
    q: float = 0.0
    q_start: float | None = None
    q_end: float | None = None
    y_w: float | None = None
    water_table_points: tuple[tuple[float, float], ...] | None = None
    n_slices: int = 30
    max_iter: int = 200
    tol: float = 1e-6


@dataclass(frozen=True)
class SearchConfig:
    n_xc: int = 31
    n_yc: int = 31
    n_R: int = 30
    xc_range: tuple[float, float] = (-5.0, 20.0)
    yc_range: tuple[float, float] = (6.0, 30.0)
    R_range: tuple[float, float] = (8.0, 60.0)
    max_workers: int | None = None


@dataclass(frozen=True)
class SlipCircleResult:
    F: float
    xc: float
    yc: float
    R: float
    x_left: float
    x_right: float
    points: tuple[tuple[float, float], ...]


# In[3]:


# ============================================================
# Ground geometry
# ============================================================
def ground_y(x, h, beta_rad):
    x = np.asarray(x, dtype=float)
    x_crest = h / np.tan(beta_rad)

    y = np.zeros_like(x)
    mask_slope = (x >= 0.0) & (x <= x_crest)
    y[mask_slope] = x[mask_slope] * np.tan(beta_rad)
    y[x > x_crest] = h
    return y


# ============================================================
# Water table geometry
# ============================================================
def validate_model_config(cfg: ModelConfig):
    if cfg.y_w is not None and cfg.water_table_points is not None:
        raise ValueError("Use either y_w or water_table_points, not both.")
    if cfg.water_table_points is not None:
        if len(cfg.water_table_points) < 2:
            raise ValueError("water_table_points must contain at least two (x, y) points.")
        xs = [p[0] for p in cfg.water_table_points]
        if any(xs[i] >= xs[i + 1] for i in range(len(xs) - 1)):
            raise ValueError("water_table_points x coordinates must be strictly increasing.")


def water_table_y(x, cfg: ModelConfig):
    """
    Piecewise-linear water table defined by cfg.water_table_points, or a
    horizontal level y_w if provided. Outside the support-point range,
    endpoint values are held constant.
    """
    x = np.asarray(x, dtype=float)

    if cfg.water_table_points is not None:
        pts = np.asarray(cfg.water_table_points, dtype=float)
        return np.interp(x, pts[:, 0], pts[:, 1])

    if cfg.y_w is not None:
        return np.full_like(x, cfg.y_w, dtype=float)

    return None


# ============================================================
# Circular slip surface
# ============================================================
def slip_y_lower(x, xc, yc, R):
    x = np.asarray(x, dtype=float)
    rad = R**2 - (x - xc) ** 2
    if np.any(rad < 0.0):
        return None
    return yc - np.sqrt(rad)


def circle_line_intersections_horizontal(y0, xc, yc, R):
    val = R**2 - (y0 - yc) ** 2
    if val < 0.0:
        return []
    dx = np.sqrt(max(val, 0.0))
    return [xc - dx, xc + dx]


def circle_line_intersections_slope(m, xc, yc, R):
    a = 1.0 + m**2
    b = -2.0 * xc - 2.0 * m * yc
    c = xc**2 + yc**2 - R**2

    disc = b**2 - 4.0 * a * c
    if disc < 0.0:
        return []

    sqrt_disc = np.sqrt(max(disc, 0.0))
    x1 = (-b - sqrt_disc) / (2.0 * a)
    x2 = (-b + sqrt_disc) / (2.0 * a)
    return [x1, x2]


def unique_points(points, tol=1e-6):
    unique_pts = []
    for p in points:
        if not any(abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol for q in unique_pts):
            unique_pts.append(p)
    unique_pts.sort(key=lambda p: p[0])
    return unique_pts


def find_circle_ground_intersections(h, beta_rad, xc, yc, R, tol=1e-8):
    x_crest = h / np.tan(beta_rad)
    m = np.tan(beta_rad)

    pts = []

    pts.extend((x, 0.0) for x in circle_line_intersections_horizontal(0.0, xc, yc, R) if x <= tol)
    pts.extend((x, m * x) for x in circle_line_intersections_slope(m, xc, yc, R) if -tol <= x <= x_crest + tol)
    pts.extend((x, h) for x in circle_line_intersections_horizontal(h, xc, yc, R) if x >= x_crest - tol)

    return unique_points(pts)


# In[4]:


# ============================================================
# Slice helpers
# ============================================================
def loaded_width_per_slice(x_left_edges, x_right_edges, q_start, q_end):
    overlap = np.minimum(x_right_edges, q_end) - np.maximum(x_left_edges, q_start)
    return np.maximum(overlap, 0.0)


def compute_slice_areas_with_gwt(y_ground_edges, y_slip_edges, dx, y_w_edges):
    """
    y_w_edges: water-table elevation evaluated at slice edges.
    """
    y_w_edges = np.asarray(y_w_edges, dtype=float)

    h_unsat_left = np.maximum(y_ground_edges[:-1] - np.maximum(y_w_edges[:-1], y_slip_edges[:-1]), 0.0)
    h_unsat_right = np.maximum(y_ground_edges[1:] - np.maximum(y_w_edges[1:], y_slip_edges[1:]), 0.0)

    h_sat_left = np.maximum(np.minimum(y_w_edges[:-1], y_ground_edges[:-1]) - y_slip_edges[:-1], 0.0)
    h_sat_right = np.maximum(np.minimum(y_w_edges[1:], y_ground_edges[1:]) - y_slip_edges[1:], 0.0)

    h_water_top_left = np.maximum(y_w_edges[:-1] - y_ground_edges[:-1], 0.0)
    h_water_top_right = np.maximum(y_w_edges[1:] - y_ground_edges[1:], 0.0)

    area_unsat = 0.5 * (h_unsat_left + h_unsat_right) * dx
    area_sat = 0.5 * (h_sat_left + h_sat_right) * dx
    area_water_top = 0.5 * (h_water_top_left + h_water_top_right) * dx
    return area_unsat, area_sat, area_water_top


def slice_top_water_force_components(x_left_edges, x_right_edges, y_ground_edges, y_w_edges, gamma_w):
    xL = x_left_edges
    xR = x_right_edges
    yL = y_ground_edges[:-1]
    yR = y_ground_edges[1:]

    dx_top = xR - xL
    dy_top = yR - yL
    s_top = np.sqrt(dx_top**2 + dy_top**2)

    y_w_edges = np.asarray(y_w_edges, dtype=float)
    hL = np.maximum(y_w_edges[:-1] - y_ground_edges[:-1], 0.0)
    hR = np.maximum(y_w_edges[1:] - y_ground_edges[1:], 0.0)

    Pn = gamma_w * 0.5 * (hL + hR) * s_top

    with np.errstate(invalid="ignore", divide="ignore"):
        nx_in = np.where(s_top > 0.0, dy_top / s_top, 0.0)
        ny_in = np.where(s_top > 0.0, -dx_top / s_top, -1.0)

    Fx = Pn * nx_in
    Fy = Pn * ny_in
    Fv = -Fy  # downward positive

    # Center of pressure along the top segment for linearly varying pressure.
    denom = hL + hR
    #frac = np.where(denom > 1e-12, (hL + 2.0 * hR) / (3.0 * denom), 0.5)
    frac = 0.5
    x_cp = xL + frac * dx_top
    y_cp = yL + frac * dy_top
    return Fv, Fx, Fy, x_cp, y_cp, hL, hR


def slice_base_pore_force_components(x_mid, y_slip_mid, xc, yc, U):
    rloc = np.sqrt((x_mid - xc) ** 2 + (y_slip_mid - yc) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        nx = np.where(rloc > 0.0, (xc - x_mid) / rloc, 0.0)
        ny = np.where(rloc > 0.0, (yc - y_slip_mid) / rloc, 1.0)
    Ux = U * nx
    Uy = U * ny
    return Ux, Uy, nx, ny


def build_slice_geometry(h, beta_deg, xc, yc, R, x_left, x_right, n_slices):
    beta_rad = np.radians(beta_deg)

    x_edges = np.linspace(x_left, x_right, n_slices + 1)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx = np.diff(x_edges)

    y_ground_edges = ground_y(x_edges, h, beta_rad)
    y_ground_mid = ground_y(x_mid, h, beta_rad)

    y_slip_edges = slip_y_lower(x_edges, xc, yc, R)
    y_slip_mid = slip_y_lower(x_mid, xc, yc, R)
    if y_slip_edges is None or y_slip_mid is None:
        return None

    if np.any((y_slip_mid - y_ground_mid) > 1e-6):
        return None

    rad_mid = R**2 - (x_mid - xc) ** 2
    if np.any(rad_mid <= 0.0):
        return None

    dydx = -(x_mid - xc) / np.sqrt(rad_mid)
    alpha = np.arctan(dydx)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    if np.any(np.abs(cos_alpha) < 1e-12):
        return None

    return {
        "x_edges": x_edges,
        "x_mid": x_mid,
        "dx": dx,
        "y_ground_edges": y_ground_edges,
        "y_ground_mid": y_ground_mid,
        "y_slip_edges": y_slip_edges,
        "y_slip_mid": y_slip_mid,
        "alpha": alpha,
        "cos_alpha": cos_alpha,
        "sin_alpha": sin_alpha,
    }


def build_slice_loads(geom, cfg: ModelConfig, xc, yc):
    dx = geom["dx"]
    y_ground_edges = geom["y_ground_edges"]
    y_slip_edges = geom["y_slip_edges"]
    y_slip_mid = geom["y_slip_mid"]
    x_edges = geom["x_edges"]
    x_mid = geom["x_mid"]

    y_w_edges = water_table_y(x_edges, cfg)
    y_w_mid = water_table_y(x_mid, cfg)

    if y_w_edges is None:
        heights_left = y_ground_edges[:-1] - y_slip_edges[:-1]
        heights_right = y_ground_edges[1:] - y_slip_edges[1:]
        areas = 0.5 * (heights_left + heights_right) * dx
        if np.any(areas <= 0.0):
            return None

        W_soil = cfg.gamma_unsat * areas
        W_water_top_v = np.zeros_like(W_soil)
        Fx_top = np.zeros_like(W_soil)
        Fy_top = np.zeros_like(W_soil)
        x_cp_top = geom["x_mid"]
        y_cp_top = geom["y_ground_mid"]
        u = np.zeros_like(W_soil)
        area_unsat = areas
        area_sat = np.zeros_like(areas)
        h_water_top_left = np.zeros_like(areas)
        h_water_top_right = np.zeros_like(areas)
    else:
        area_unsat, area_sat, _ = compute_slice_areas_with_gwt(y_ground_edges, y_slip_edges, dx, y_w_edges)
        total_soil_area = area_unsat + area_sat
        if np.any(total_soil_area <= 0.0):
            return None

        W_soil = cfg.gamma_unsat * area_unsat + cfg.gamma_sat * area_sat
        W_water_top_v, Fx_top, Fy_top, x_cp_top, y_cp_top, h_water_top_left, h_water_top_right = slice_top_water_force_components(
            x_edges[:-1], x_edges[1:], y_ground_edges, y_w_edges, cfg.gamma_w
        )
        pressure_head = np.maximum(y_w_mid - y_slip_mid, 0.0)
        u = cfg.gamma_w * pressure_head

    if cfg.q > 0.0 and cfg.q_start is not None and cfg.q_end is not None and cfg.q_end > cfg.q_start:
        b_loaded = loaded_width_per_slice(x_edges[:-1], x_edges[1:], cfg.q_start, cfg.q_end)
        W_q = cfg.q * b_loaded
    else:
        W_q = np.zeros_like(W_soil)

    W = W_soil + W_q + W_water_top_v
    M_vertical = -(geom["x_mid"] - xc) * W
    M_horizontal = -(y_cp_top - yc) * Fx_top
    M_total = M_vertical + M_horizontal

    return {
        "y_w_edges": y_w_edges,
        "y_w_mid": y_w_mid,
        "h_water_top_left": h_water_top_left,
        "h_water_top_right": h_water_top_right,
        "area_unsat": area_unsat,
        "area_sat": area_sat,
        "W_soil": W_soil,
        "W_q": W_q,
        "W_water_top_v": W_water_top_v,
        "W": W,
        "Fx_top": Fx_top,
        "Fy_top": Fy_top,
        "x_cp_top": x_cp_top,
        "y_cp_top": y_cp_top,
        "u": u,
        "M_vertical": M_vertical,
        "M_horizontal": M_horizontal,
        "M_total": M_total,
    }


# In[5]:


# ============================================================
# Hybrid Bishop solver with horizontal surface-water contribution
# ============================================================
def bishop_fs(cfg: ModelConfig, xc, yc, R, x_left, x_right, return_details=False):
    geom = build_slice_geometry(cfg.h, cfg.beta_deg, xc, yc, R, x_left, x_right, cfg.n_slices)
    if geom is None:
        return None

    loads = build_slice_loads(geom, cfg, xc, yc)
    if loads is None:
        return None

    phi_rad = np.radians(cfg.phi_deg)
    dx = geom["dx"]
    cos_alpha = geom["cos_alpha"]
    sin_alpha = geom["sin_alpha"]

    M_drive = np.sum(loads["M_total"])
    if abs(M_drive) < 1e-12:
        return None

    s_drive = np.sign(M_drive)
    driving = abs(M_drive) / R

    F = 1.5
    history = []
    for _ in range(cfg.max_iter):
        denom = cos_alpha + (s_drive * sin_alpha * np.tan(phi_rad)) / F
        if np.any(np.abs(denom) < 1e-12):
            return None

        bishop_effective_load = loads["W"] - loads["u"] * dx
        resisting_terms = (cfg.c * dx + bishop_effective_load * np.tan(phi_rad)) / denom
        resisting = np.sum(resisting_terms)
        F_new = resisting / driving
        history.append(F_new)

        if not np.isfinite(F_new) or F_new <= 0.0:
            return None

        if abs(F_new - F) < cfg.tol:
            if return_details:
                return F_new, {
                    **geom,
                    **loads,
                    "bishop_effective_load": bishop_effective_load,
                    "resisting_terms": resisting_terms,
                    "driving": driving,
                    "M_drive": M_drive,
                    "s_drive": s_drive,
                    "iterations": len(history),
                    "history": np.array(history, dtype=float),
                    "denom": denom,
                }
            return F_new

        F = F_new

    return None


# In[6]:


def build_slice_dataframe(result: SlipCircleResult, cfg: ModelConfig):
    solved = bishop_fs(cfg, result.xc, result.yc, result.R, result.x_left, result.x_right, return_details=True)
    if solved is None:
        return None

    F, details = solved
    dx = details["dx"]
    U = details["u"] * dx
    Ux, Uy, _, _ = slice_base_pore_force_components(details["x_mid"], details["y_slip_mid"], result.xc, result.yc, U)

    df = pd.DataFrame({
        "slice": np.arange(1, len(dx) + 1),
        "x_left": details["x_edges"][:-1],
        "x_right": details["x_edges"][1:],
        "x_mid": details["x_mid"],
        "dx": dx,
        "y_ground_left": details["y_ground_edges"][:-1],
        "y_ground_right": details["y_ground_edges"][1:],
        "y_ground_mid": details["y_ground_mid"],
        "y_slip_left": details["y_slip_edges"][:-1],
        "y_slip_right": details["y_slip_edges"][1:],
        "y_slip_mid": details["y_slip_mid"],
        "y_w_left": np.nan if details["y_w_edges"] is None else details["y_w_edges"][:-1],
        "y_w_right": np.nan if details["y_w_edges"] is None else details["y_w_edges"][1:],
        "y_w_mid": np.nan if details["y_w_mid"] is None else details["y_w_mid"],
        "alpha_rad": details["alpha"],
        "alpha_deg": np.degrees(details["alpha"]),
        "sin_alpha": details["sin_alpha"],
        "cos_alpha": details["cos_alpha"],
        "area_unsat": details["area_unsat"],
        "area_sat": details["area_sat"],
        "W_soil": details["W_soil"],
        "W_q": details["W_q"],
        "W_water_top_v": details["W_water_top_v"],
        "W_total_vertical": details["W"],
        "u": details["u"],
        "u_times_b": U,
        "top_water_h_left": details["h_water_top_left"],
        "top_water_h_right": details["h_water_top_right"],
        "Fx_top": details["Fx_top"],
        "Fy_top": details["Fy_top"],
        "x_cp_top": details["x_cp_top"],
        "y_cp_top": details["y_cp_top"],
        "Ux_base": Ux,
        "Uy_base": Uy,
        "M_vertical": details["M_vertical"],
        "M_horizontal": details["M_horizontal"],
        "M_total": details["M_total"],
        "bishop_effective_load": details["bishop_effective_load"],
        "resisting_term": details["resisting_terms"],
        "driving_force_equivalent": details["M_total"] / result.R,
    })

    df.attrs["F"] = F
    df.attrs["mu"] = 1.0 / F
    df.attrs["M_drive"] = details["M_drive"]
    df.attrs["driving"] = details["driving"]
    df.attrs["iterations"] = details["iterations"]
    return df


# In[7]:


def _evaluate_center(args):
    ix, iy, xc, yc, R_vals, cfg = args
    beta_rad = np.radians(cfg.beta_deg)

    best_F_at_center = None
    best_local = None
    checked = 0
    valid = 0

    for R in R_vals:
        checked += 1

        if yc - R > cfg.h:
            continue

        pts = find_circle_ground_intersections(cfg.h, beta_rad, xc, yc, R)
        if len(pts) < 2:
            continue

        x_left = pts[0][0]
        x_right = pts[-1][0]
        if x_right - x_left < 0.5:
            continue

        F = bishop_fs(cfg, xc, yc, R, x_left, x_right)
        if F is None:
            continue

        valid += 1

        if best_F_at_center is None or F < best_F_at_center:
            best_F_at_center = F

        if best_local is None or F < best_local.F:
            best_local = SlipCircleResult(
                F=F,
                xc=xc,
                yc=yc,
                R=R,
                x_left=x_left,
                x_right=x_right,
                points=tuple(pts),
            )

    return ix, iy, best_F_at_center, best_local, checked, valid


# ============================================================
# Parallel grid search
# ============================================================
def search_critical_circle_grid_parallel(cfg: ModelConfig, search: SearchConfig):
    validate_model_config(cfg)

    xc_vals = np.linspace(search.xc_range[0], search.xc_range[1], search.n_xc)
    yc_vals = np.linspace(search.yc_range[0], search.yc_range[1], search.n_yc)
    R_vals = np.linspace(search.R_range[0], search.R_range[1], search.n_R)

    F_center_grid = np.full((search.n_yc, search.n_xc), np.nan)
    best = None
    checked = 0
    valid = 0

    tasks = [
        (ix, iy, xc, yc, R_vals, cfg)
        for ix, xc in enumerate(xc_vals)
        for iy, yc in enumerate(yc_vals)
    ]

    with ProcessPoolExecutor(max_workers=search.max_workers) as executor:
        for ix, iy, best_F_at_center, best_local, checked_i, valid_i in executor.map(_evaluate_center, tasks):
            checked += checked_i
            valid += valid_i

            if best_F_at_center is not None:
                F_center_grid[iy, ix] = best_F_at_center

            if best_local is not None and (best is None or best_local.F < best.F):
                best = best_local

    return best, checked, valid, xc_vals, yc_vals, F_center_grid


# In[8]:


# ============================================================
# Plot helpers
# ============================================================
def _plot_search_grid_and_contours(ax, fig, xc_vals, yc_vals, F_center_grid):
    Xc, Yc = np.meshgrid(xc_vals, yc_vals)
    mu_center_grid = 1.0 / F_center_grid
    mu_masked = np.ma.masked_invalid(mu_center_grid)
    valid_vals = mu_center_grid[np.isfinite(mu_center_grid)]

    if valid_vals.size > 0:
        mumin = np.nanmin(valid_vals)
        mumax = np.nanmax(valid_vals)
        if mumax > mumin:
            levels = np.linspace(mumin, mumax, 12)
            cf = ax.contourf(Xc, Yc, mu_masked, levels=levels, alpha=0.35, cmap="coolwarm")
            cs = ax.contour(Xc, Yc, mu_masked, levels=levels, linewidths=2.0,
                            cmap="coolwarm", linestyles="solid")
            ax.clabel(cs, inline=True, fontsize=14, fmt="%.2f", colors="black")

    XX, YY = np.meshgrid(xc_vals, yc_vals)
    ax.plot(XX.ravel(), YY.ravel(), marker=".", linestyle="None", markersize=3, color="black", alpha=0.35, label="Search grid")


def _plot_ground_water_and_surcharge(ax, cfg: ModelConfig, x_plot, y_ground, x_min, x_max, y_min, y_max):
    y_water_plot = water_table_y(x_plot, cfg)
    if y_water_plot is not None:
        mask_water = y_water_plot > y_ground
        ax.fill_between(x_plot, y_ground, y_water_plot, where=mask_water, alpha=0.25, interpolate=True, label="Water above ground")
        ax.plot(x_plot, y_water_plot, linestyle="--", linewidth=1.8, label="Water level")

    ax.plot(x_plot, y_ground, color="black", linewidth=2.2, label="Ground")
    ax.fill_between(x_plot, y_ground, y_min - 5 * cfg.h, color="white", alpha=0.85)
    ax.plot(x_plot, y_ground, color="black", linewidth=2.2)

    if cfg.q > 0.0 and cfg.q_start is not None and cfg.q_end is not None and cfg.q_end > cfg.q_start:
        xq = np.linspace(cfg.q_start, cfg.q_end, 9)
        yq = ground_y(xq, cfg.h, np.radians(cfg.beta_deg))
        ax.plot([cfg.q_start, cfg.q_end], [yq[0], yq[-1]], linewidth=4, alpha=0.9, label=f"Distributed load q = {cfg.q:.1f} kN/m²")

        arrow_len = 0.08 * (y_max - y_min)
        for xi, yi in zip(xq, yq):
            ax.arrow(xi, yi + arrow_len, 0.0, -0.8 * arrow_len, width=0.0, head_width=0.25, head_length=0.25, length_includes_head=True)


def _plot_critical_circle_and_slices(ax, cfg: ModelConfig, result: SlipCircleResult):
    x_slip = np.linspace(result.x_left, result.x_right, 600)
    y_slip = slip_y_lower(x_slip, result.xc, result.yc, result.R)
    ax.plot(x_slip, y_slip, color="red", linewidth=2.2, label="Critical slip circle")

    geom = build_slice_geometry(cfg.h, cfg.beta_deg, result.xc, result.yc, result.R, result.x_left, result.x_right, cfg.n_slices)
    for xe, yg, ys in zip(geom["x_edges"], geom["y_ground_edges"], geom["y_slip_edges"]):
        ax.plot([xe, xe], [ys, yg], color="gray", linewidth=0.7, alpha=0.55)

    ax.plot(result.xc, result.yc, marker="o", markersize=8, color="red", label="Critical center")
    for i, (x, y) in enumerate(result.points):
        ax.plot(x, y, marker="s", color="red", markersize=6, label="Intersections" if i == 0 else None)

    return geom


def _draw_arrow_with_scaled_head(ax, x0, y0, dx, dy, color, label=None, linewidth=1.5, reverse_anchor=False):
    L = np.hypot(dx, dy)
    if L <= 1e-12:
        return False

    head_length = 0.35 * L
    head_width = 0.20 * L

    if reverse_anchor:
        x0, y0, dx, dy = x0 + dx, y0 + dy, -dx, -dy

    ax.arrow(
        x0,
        y0,
        dx,
        dy,
        length_includes_head=True,
        head_length=head_length,
        head_width=head_width,
        linewidth=linewidth,
        color=color,
        label=label,
    )
    return True


def _plot_water_force_arrows(ax, cfg: ModelConfig, result: SlipCircleResult, geom, water_force_stride=2):
    loads = build_slice_loads(geom, cfg, result.xc, result.yc)
    if loads is None or loads["y_w_edges"] is None:
        return

    Pn_top = np.sqrt(loads["Fx_top"] ** 2 + loads["Fy_top"] ** 2)
    U = loads["u"] * geom["dx"]
    Ux, Uy, _, _ = slice_base_pore_force_components(geom["x_mid"], geom["y_slip_mid"], result.xc, result.yc, U)
    Umag = np.sqrt(Ux**2 + Uy**2)

    max_top = np.max(Pn_top[Pn_top > 1e-12]) if np.any(Pn_top > 1e-12) else 1.0
    max_base = np.max(Umag[Umag > 1e-12]) if np.any(Umag > 1e-12) else 1.0
    top_scale = 0.12 * cfg.h / max_top
    base_scale = 0.12 * cfg.h / max_base

    first_top = True
    first_base = True
    idxs = np.arange(0, len(geom["x_mid"]), max(1, water_force_stride))

    for i in idxs:
        if Pn_top[i] > 1e-12:
            dxr = loads["Fx_top"][i] * top_scale
            dyr = loads["Fy_top"][i] * top_scale
            drawn = _draw_arrow_with_scaled_head(
                ax,
                loads["x_cp_top"][i] - dxr,
                loads["y_cp_top"][i] - dyr,
                dxr,
                dyr,
                color="tab:blue",
                label="Surface water resultant" if first_top else None,
            )
            if drawn:
                first_top = False

        dxu = -Ux[i] * base_scale
        dyu = -Uy[i] * base_scale
        drawn = _draw_arrow_with_scaled_head(
            ax,
            geom["x_mid"][i],
            geom["y_slip_mid"][i],
            dxu,
            dyu,
            color="tab:green",
            label="Base pore-water resultant U" if first_base else None,
            reverse_anchor=True,
        )
        if drawn:
            first_base = False


# ============================================================
# Plot
# ============================================================
def plot_slope_with_searchgrid_and_isoasphals(cfg: ModelConfig, result: SlipCircleResult, xc_vals, yc_vals, F_center_grid, show_water_forces=True, water_force_stride=2):
    validate_model_config(cfg)

    beta_rad = np.radians(cfg.beta_deg)
    x_crest = cfg.h / np.tan(beta_rad)

    x_min = min(-0.5 * cfg.h, result.x_left - 0.3 * cfg.h, np.min(xc_vals) - 0.05 * cfg.h)
    x_max = max(x_crest + 1.5 * cfg.h, result.x_right + 0.3 * cfg.h, np.max(xc_vals) + 0.05 * cfg.h)
    y_min = min(np.nanmin(yc_vals) - 0.05 * cfg.h, -0.2 * cfg.h)
    y_max = max(np.nanmax(yc_vals) + 0.05 * cfg.h, result.yc + 0.2 * cfg.h, cfg.h + 0.5 * cfg.h)

    x_plot = np.linspace(x_min, x_max, 1200)
    y_ground = ground_y(x_plot, cfg.h, beta_rad)

    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_search_grid_and_contours(ax, fig, xc_vals, yc_vals, F_center_grid)
    _plot_ground_water_and_surcharge(ax, cfg, x_plot, y_ground, x_min, x_max, y_min, y_max)
    geom = _plot_critical_circle_and_slices(ax, cfg, result)

    if show_water_forces:
        _plot_water_force_arrows(ax, cfg, result, geom, water_force_stride=water_force_stride)

    ax.set_title(f"Slope with water, search grid, isoasphals and critical slip circle (µ = {(1 / result.F):.3f})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig("Bishop_slices_water_up.pdf")


# In[9]:


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    model = ModelConfig(
        h=4.5,
        beta_deg=26.57,
        phi_deg=20.0,
        c=5.0,
        gamma_unsat=1.6 * 9.81,
        gamma_sat=2.1 * 9.81,
        gamma_w=9.81,
        q=30.0,
        q_start=9.5,
        q_end=12.5,
        water_table_points=(( -1.0, 2.0), (4.0, 2.0), (15.0, 3.0)),
        n_slices=80,
    )

    search = SearchConfig(
        n_xc=51,
        n_yc=51,
        n_R=30,
        xc_range=(-1.0, 10.0),
        yc_range=(5.0, 15.0),
        R_range=(5.0, 15.0),
        max_workers=os.cpu_count(),
    )

    best, checked, valid, xc_vals, yc_vals, F_center_grid = search_critical_circle_grid_parallel(model, search)

    print(f"Checked combinations: {checked}")
    print(f"Valid slip circles:   {valid}")

    if best is None:
        print("No valid slip surface found.")
    else:
        print("\nCritical slip surface:")
        print(f"  F       = {best.F:.4f}")
        print(f"  µ       = {(1 / best.F):.4f}")
        print(f"  xc      = {best.xc:.4f} m")
        print(f"  yc      = {best.yc:.4f} m")
        print(f"  R       = {best.R:.4f} m")
        print(f"  x_left  = {best.x_left:.4f} m")
        print(f"  x_right = {best.x_right:.4f} m")

        critical_circle_df = build_slice_dataframe(best, model)
        summary_df = pd.DataFrame([{
            **asdict(best),
            "mu": 1.0 / best.F,
            "n_slices": model.n_slices,
            "y_w": model.y_w,
            "water_table_points": model.water_table_points,
            "q": model.q,
        }]).drop(columns=["points"])

        print("\nKritischster Kreis als DataFrame:")
        print(summary_df.to_string(index=False))

        print("\nLamellen-DataFrame (erste 10 Zeilen):")
        print(critical_circle_df.head(10).to_string(index=False))

        plot_slope_with_searchgrid_and_isoasphals(
            cfg=model,
            result=best,
            xc_vals=xc_vals,
            yc_vals=yc_vals,
            F_center_grid=F_center_grid,
            show_water_forces=True,
            water_force_stride=2,
        )
# %%