"""Pure-Python 3DEP Airborne LiDAR Entwine Point Tile (EPT) processor.

Streams USGS 3DEP leaf-off LiDAR point clouds directly from AWS S3
(s3-us-west-2.amazonaws.com/usgs-lidar-public) without authentication or rate
limiting, normalizes heights above ground using LAS Class 2 ground returns,
and computes Beer-Lambert penetration-adjusted vertical structure metrics
(understory 0.5-2 m, midstory 2-10 m, upper canopy >10 m, canopy height, FHD).
"""

from __future__ import annotations

import datetime
import io
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import laspy
import numpy as np
import pyproj
import requests
from scipy.spatial import cKDTree

S3_EPT_BASE = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"

# LAS standard classification codes to EXCLUDE (noise=7, high_noise=18, water=9, building=6)
BAD_CLASSES = {6, 7, 9, 18}


class EPTClient:
    """Thread-safe cached client for querying USGS 3DEP EPT point clouds on S3."""

    def __init__(self, dataset: str, max_workers: int = 12, timeout: int = 45):
        self.dataset = dataset
        self.base = f"{S3_EPT_BASE}/{dataset}"
        self.max_workers = max_workers
        self.timeout = timeout
        self._lock = threading.Lock()
        self._hier_cache: dict[str, dict] = {}
        self._node_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers * 2, pool_maxsize=max_workers * 2)
        self._session.mount("https://", adapter)
        self.meta = self._session.get(f"{self.base}/ept.json", timeout=timeout).json()
        self.bounds = self.meta["bounds"]  # [xmin, ymin, zmin, xmax, ymax, zmax] in EPSG:3857
        self.fwd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def _get_hier(self, key: str) -> dict:
        with self._lock:
            if key in self._hier_cache:
                return self._hier_cache[key]
        r = self._session.get(f"{self.base}/ept-hierarchy/{key}.json", timeout=self.timeout)
        r.raise_for_status()
        h = r.json()
        with self._lock:
            self._hier_cache[key] = h
        return h

    def _collect_nodes(self, qx0: float, qy0: float, qx1: float, qy1: float) -> list[str]:
        root_h = self._get_hier("0-0-0-0")
        nodes: list[str] = []
        b0 = self.bounds

        def walk(d, x, y, z, bx0, by0, bz0, bx1, by1, bz1, h):
            key = f"{d}-{x}-{y}-{z}"
            if key not in h:
                return
            if bx1 < qx0 or bx0 > qx1 or by1 < qy0 or by0 > qy1:
                return
            val = h[key]
            if val == -1:
                h = self._get_hier(key)
                val = h.get(key, 0)
            if val > 0:
                nodes.append(key)
            mx = 0.5 * (bx0 + bx1)
            my = 0.5 * (by0 + by1)
            mz = 0.5 * (bz0 + bz1)
            for dx in (0, 1):
                nx0, nx1 = (bx0, mx) if dx == 0 else (mx, bx1)
                if nx1 < qx0 or nx0 > qx1:
                    continue
                for dy in (0, 1):
                    ny0, ny1 = (by0, my) if dy == 0 else (my, by1)
                    if ny1 < qy0 or ny0 > qy1:
                        continue
                    for dz in (0, 1):
                        nz0, nz1 = (bz0, mz) if dz == 0 else (mz, bz1)
                        walk(d + 1, x * 2 + dx, y * 2 + dy, z * 2 + dz, nx0, ny0, nz0, nx1, ny1, nz1, h)

        walk(0, 0, 0, 0, b0[0], b0[1], b0[2], b0[3], b0[4], b0[5], root_h)
        return nodes

    def _fetch_node(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        with self._lock:
            if key in self._node_cache:
                return self._node_cache[key]
        last_err = None
        for attempt in range(3):
            try:
                r = self._session.get(f"{self.base}/ept-data/{key}.laz", timeout=self.timeout)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        else:
            raise last_err
        las = laspy.read(io.BytesIO(r.content))
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        z = np.asarray(las.z, dtype=np.float64)
        c = np.asarray(las.classification, dtype=np.uint8)
        rn = (
            np.asarray(las.return_number, dtype=np.uint8)
            if hasattr(las, "return_number") and len(las.points) > 0
            else np.ones(len(x), dtype=np.uint8)
        )
        gt = float(np.median(las.gps_time)) if hasattr(las, "gps_time") and len(las.points) > 0 else float("nan")
        res = (x, y, z, c, rn, gt)
        # Cache shallow nodes (depth <= 7) which are shared across many nearby plots
        depth = int(key.split("-")[0])
        if depth <= 7:
            with self._lock:
                if len(self._node_cache) < 250:
                    self._node_cache[key] = res
        return res

    def query_plot_points(
        self, lon: float, lat: float, radius_m: float = 30.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Fetch all points inside a square box of half-width `radius_m` (true ground meters)
        centered at (lon, lat), returning (x, y, z, classification, return_number, median_gps_time)."""
        cx, cy = self.fwd.transform(lon, lat)
        # Correct Web Mercator scale factor (1 / cos(lat)) so query box is true meters on ground
        scale = 1.0 / math.cos(math.radians(lat))
        pad = (radius_m + 6.0) * scale  # 6 m buffer for edge ground interpolation
        qx0, qx1 = cx - pad, cx + pad
        qy0, qy1 = cy - pad, cy + pad
        nodes = self._collect_nodes(qx0, qy0, qx1, qy1)
        if not nodes:
            empty = np.array([], dtype=np.float64)
            empty_u8 = np.array([], dtype=np.uint8)
            return empty, empty, empty, empty_u8, empty_u8, float("nan")

        xs, ys, zs, cs, rns, gts = [], [], [], [], [], []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [ex.submit(self._fetch_node, k) for k in nodes]
            for fut in as_completed(futs):
                x, y, z, c, rn, gt = fut.result()
                m = (x >= qx0) & (x <= qx1) & (y >= qy0) & (y <= qy1)
                if m.any():
                    xs.append(x[m])
                    ys.append(y[m])
                    zs.append(z[m])
                    cs.append(c[m])
                    rns.append(rn[m])
                    if not math.isnan(gt):
                        gts.append(gt)
        if not xs:
            empty = np.array([], dtype=np.float64)
            empty_u8 = np.array([], dtype=np.uint8)
            return empty, empty, empty, empty_u8, empty_u8, float("nan")
        return (
            np.concatenate(xs),
            np.concatenate(ys),
            np.concatenate(zs),
            np.concatenate(cs),
            np.concatenate(rns),
            float(np.median(gts)) if gts else float("nan"),
        )


def gps_time_to_iso(gps_time: float) -> str | None:
    if gps_time is None or math.isnan(gps_time) or gps_time <= 0:
        return None
    # Standard Adjusted GPS Time = GPS seconds - 1e9; raw GPS week seconds are < 604800
    if gps_time < 604800:
        return None
    offset = 1e9 if gps_time < 6e8 else 0.0
    dt = datetime.datetime(1980, 1, 6) + datetime.timedelta(seconds=float(gps_time) + offset)
    return dt.strftime("%Y-%m-%d")


def compute_plot_metrics(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    c: np.ndarray,
    lon: float,
    lat: float,
    radius_m: float = 30.0,
    gps_time: float = float("nan"),
    subsample_frac: float | None = None,
    rng: np.random.Generator | None = None,
    rn: np.ndarray | None = None,
) -> dict[str, object] | None:
    """Compute height-above-ground vertical structural metrics inside a circular
    plot of true ground radius `radius_m` centered at (lon, lat).

    Uses LAS Class 2 (ground) k-NN inverse-distance-weighted surface interpolation
    on the buffered cloud, then evaluates returns strictly inside the circle.
    When `rn` (LAS return_number) is provided, also computes unbiased first-return
    upper canopy cover (>=10 m) and gap fraction, which are invariant to subcanopy
    return removal.
    """
    if len(x) < 100:
        return None

    # Filter out noise / buildings / water
    good = np.ones(len(c), dtype=bool)
    for bad_c in BAD_CLASSES:
        good &= c != bad_c
    x, y, z, c = x[good], y[good], z[good], c[good]
    if rn is not None:
        rn = rn[good]
    if len(x) < 100:
        return None

    # Optional random return thinning for sampling-density / ground-IDW invariance tests
    if subsample_frac is not None and 0.0 < subsample_frac < 1.0:
        g = rng if rng is not None else np.random.default_rng(42)
        keep = g.random(len(x)) < subsample_frac
        if keep.sum() < 100:
            return None
        x, y, z, c = x[keep], y[keep], z[keep], c[keep]
        if rn is not None:
            rn = rn[keep]

    # Convert EPSG:3857 offsets to true local ground meters around plot center
    fwd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    cx, cy = fwd.transform(lon, lat)
    cos_lat = math.cos(math.radians(lat))
    lx = (x - cx) * cos_lat
    ly = (y - cy) * cos_lat

    gnd_mask = c == 2
    if gnd_mask.sum() < 25:
        return None

    # Fit ground surface from buffered ground returns (k=6 IDW)
    gnd_xy = np.column_stack([lx[gnd_mask], ly[gnd_mask]])
    gnd_z = z[gnd_mask]
    tree = cKDTree(gnd_xy)

    # Evaluate only returns inside circular plot of radius_m
    in_plot = (lx * lx + ly * ly) <= (radius_m * radius_m)
    if in_plot.sum() < 100:
        return None

    px, py, pz, pc = lx[in_plot], ly[in_plot], z[in_plot], c[in_plot]
    prn = rn[in_plot] if rn is not None else None
    k = min(6, len(gnd_z))
    dists, idxs = tree.query(np.column_stack([px, py]), k=k)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    w = 1.0 / np.maximum(dists, 0.05)
    gz = np.sum(w * gnd_z[idxs], axis=1) / np.sum(w, axis=1)

    hag = pz - gz
    # Clamp implausible outliers (ground points sit near 0; exclude >55 m birds/noise)
    valid_h = (hag >= -0.5) & (hag <= 55.0)
    hag = hag[valid_h]
    pc = pc[valid_h]
    if prn is not None:
        prn = prn[valid_h]
    # Ground returns forced to 0.0 m HAG
    hag = np.where(pc == 2, 0.0, np.maximum(hag, 0.0))

    n_tot = len(hag)
    if n_tot < 100:
        return None

    area_m2 = math.pi * radius_m * radius_m
    n_gnd = int(np.sum(hag < 0.5))
    n_under = int(np.sum((hag >= 0.5) & (hag < 2.0)))
    n_low_mid = int(np.sum((hag >= 2.0) & (hag < 5.0)))
    n_high_mid = int(np.sum((hag >= 5.0) & (hag < 10.0)))
    n_mid = n_low_mid + n_high_mid  # 2.0 - 10.0 m mesophication stratum
    n_canopy = int(np.sum(hag >= 10.0))

    # Beer-Lambert penetration-adjusted Normalized Relative Point Density (NRD)
    # Controls for upper-canopy occlusion and pulse-density variation:
    # NRD(z1, z2) = N(z1..z2) / N(<= z2)
    denom_under = n_gnd + n_under
    denom_low_mid = denom_under + n_low_mid
    denom_mid = denom_under + n_mid
    nrd_understory = n_under / denom_under if denom_under > 0 else float("nan")
    nrd_low_midstory = n_low_mid / denom_low_mid if denom_low_mid > 0 else float("nan")
    nrd_midstory = n_mid / denom_mid if denom_mid > 0 else float("nan")
    nrd_tall_midstory = n_high_mid / denom_mid if denom_mid > 0 else float("nan")

    # Fine 1 m vertical NRD profile (0.5-10 m) for continuous height-profile diagnostics
    fine_bins = [
        ("0.5_1", 0.5, 1.0), ("1_2", 1.0, 2.0), ("2_3", 2.0, 3.0), ("3_4", 3.0, 4.0),
        ("4_5", 4.0, 5.0), ("5_6", 5.0, 6.0), ("6_7", 6.0, 7.0), ("7_8", 7.0, 8.0),
        ("8_9", 8.0, 9.0), ("9_10", 9.0, 10.0),
    ]
    nrd_1m: dict[str, float] = {}
    for lbl, z1, z2 in fine_bins:
        nb = int(np.sum((hag >= z1) & (hag < z2)))
        nd = int(np.sum(hag < z2))
        nrd_1m[lbl] = round(float(nb / nd), 5) if nd > 0 else float("nan")

    # Raw all-return proportions (note: canopy_cover_10m is mechanically coupled to subcanopy returns;
    # use first_return_canopy_10m / canopy_gap_10m for unbiased upper-canopy inference)
    frac_understory = n_under / n_tot
    frac_midstory = n_mid / n_tot
    canopy_cover_10m = n_canopy / n_tot

    first_return_canopy_10m: float | None = None
    canopy_gap_10m: float | None = None
    frac_first_returns: float | None = None
    if prn is not None:
        first_mask = prn == 1
        n_first = int(np.sum(first_mask))
        frac_first_returns = round(float(n_first / n_tot), 5)
        if n_first >= 25:
            frc = float(np.sum(first_mask & (hag >= 10.0)) / n_first)
            first_return_canopy_10m = round(frc, 5)
            canopy_gap_10m = round(1.0 - frc, 5)

    # Canopy height metrics on vegetation returns (>= 0.5 m)
    veg = hag[hag >= 0.5]
    chm_p95 = float(np.percentile(veg, 95)) if len(veg) >= 10 else 0.0
    chm_p50 = float(np.percentile(veg, 50)) if len(veg) >= 10 else 0.0
    chm_mean = float(np.mean(veg)) if len(veg) >= 10 else 0.0

    # Foliage Height Diversity (Shannon entropy across 2 m bins from 0 to 40 m)
    hist, _ = np.histogram(hag, bins=np.arange(0, 42, 2))
    p = hist[hist > 0] / float(hist.sum())
    fhd = float(-np.sum(p * np.log(p)))

    out: dict[str, object] = {
        "n_pts": int(n_tot),
        "pulse_density": round(n_tot / area_m2, 3),
        "nrd_understory": round(float(nrd_understory), 5),
        "nrd_low_midstory": round(float(nrd_low_midstory), 5),
        "nrd_tall_midstory": round(float(nrd_tall_midstory), 5),
        "nrd_midstory": round(float(nrd_midstory), 5),
        "nrd_1m": nrd_1m,
        "frac_understory": round(float(frac_understory), 5),
        "frac_midstory": round(float(frac_midstory), 5),
        "canopy_cover_10m": round(float(canopy_cover_10m), 5),
        "chm_p95": round(chm_p95, 2),
        "chm_p50": round(chm_p50, 2),
        "chm_mean": round(chm_mean, 2),
        "fhd": round(fhd, 4),
        "acq_date": gps_time_to_iso(gps_time),
    }
    if first_return_canopy_10m is not None:
        out["first_return_canopy_10m"] = first_return_canopy_10m
        out["canopy_gap_10m"] = canopy_gap_10m
    if frac_first_returns is not None:
        out["frac_first_returns"] = frac_first_returns
    return out
