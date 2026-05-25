#!/usr/bin/env python3
from __future__ import annotations

import gc
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import dask
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

# ============================================================
# USER SETTINGS (edit here only)
# ============================================================
HDF5_DIR = Path("/home/NAS/homes/pinghung-10018/NCDR_RADAR/QPESUMS_3d_CWB/All_day_h5/2024_all_day_h5")
HDF5_PATTERN = "MREF3D21L.*.hdf5"
OUTPUT_ZARR = Path("/home/NAS/homes/pinghung-10018/Course_data/Isungche/Final_project/radar_taiwan_2024_with_derived.zarr")

# Time filter. Use None for full range.
START_TIME = None  # e.g. "2024-01-01"
END_TIME = None    # e.g. "2024-12-31 23:59"

# Chunk strategy (requested): DBZH chunk = (time, alt, lat, lon) = (1, 21, 561, 441)
# Values larger than real dimension size are clipped to the dimension size.
DBZH_CHUNK_TIME = 1
DBZH_CHUNK_ALT = 21
DBZH_CHUNK_LAT = 561
DBZH_CHUNK_LON = 441

# Batch mode for year-scale run: "daily", "monthly", "all"
BATCH_MODE = "monthly"

# Other options
OVERWRITE_OUTPUT = True
ZARR_FORMAT = 3
DASK_SCHEDULER = "threads"  # "threads", "processes", "synchronous"

# Dask workers (LocalCluster)
USE_DISTRIBUTED = True
N_WORKERS = 8
THREADS_PER_WORKER = 1
MEMORY_LIMIT_PER_WORKER = "4GB"  # e.g. "8GB", "16GB", "0" (no limit)
DASK_DASHBOARD_ADDRESS = ":8787"

# ============================================================
# Constants
# ============================================================
DEFAULT_ALT_KM = np.array(
    [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    dtype=np.float32,
)
VIL_MFAC = 3.44e-3
VIL_DBZ_FLOOR = 0.0


def parse_file_time(path: Path) -> datetime:
    parts = path.stem.split(".")
    return datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M")


def to_float_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, bytes):
        text = value.decode("utf-8")
    else:
        text = str(value)
    text = text.replace(",", " ").replace("[", " ").replace("]", " ")
    return np.asarray([float(x) for x in text.split() if x.strip()], dtype=np.float32)


def read_grid(sample_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(sample_file, "r") as f:
        where = f["where"]
        xsize = int(where.attrs["xsize"])
        ysize = int(where.attrs["ysize"])
        xscale = float(where.attrs["xscale"])
        yscale = float(where.attrs["yscale"])
        ll_lon = float(where.attrs["LL_lon"])
        ll_lat = float(where.attrs["LL_lat"])

        zlevels_raw = where.attrs.get("zlevels", None)
        if zlevels_raw is None:
            alt_km = DEFAULT_ALT_KM
        else:
            try:
                alt_km = to_float_array(zlevels_raw)
            except Exception:
                alt_km = DEFAULT_ALT_KM

    lon = np.round(ll_lon + xscale * np.arange(xsize), 6).astype(np.float64)
    lat = np.round(ll_lat + yscale * np.arange(ysize - 1, -1, -1), 6).astype(np.float64)
    return lat, lon, alt_km.astype(np.float64)


@dask.delayed
def load_one_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Follow hw1.ipynb: dataset1=MAXDBZ, dataset2=DBZH
    with h5py.File(path, "r") as f:
        nodata = float(f["what"].attrs.get("no_data_value", -999))

        maxdbz_raw = f["dataset1/data1/data"][:].astype(np.float32)
        dbzh_raw = f["dataset2/data1/data"][:].astype(np.float32)

        max_gain = float(f["dataset1/what"].attrs.get("gain", 1.0))
        max_offset = float(f["dataset1/what"].attrs.get("offset", 0.0))
        dbzh_gain = float(f["dataset2/what"].attrs.get("gain", 1.0))
        dbzh_offset = float(f["dataset2/what"].attrs.get("offset", 0.0))

    max_mask = maxdbz_raw == nodata
    dbzh_mask = dbzh_raw == nodata

    maxdbz = maxdbz_raw * max_gain + max_offset
    dbzh = dbzh_raw * dbzh_gain + dbzh_offset

    maxdbz[max_mask] = np.nan
    dbzh[dbzh_mask] = np.nan
    return maxdbz, dbzh


def ensure_alt(dbzh: xr.DataArray, alt_km: np.ndarray) -> xr.DataArray:
    if "alt" not in dbzh.dims:
        raise ValueError("DBZH must have 'alt' dimension")
    if "alt" not in dbzh.coords or dbzh.sizes["alt"] != len(alt_km):
        dbzh = dbzh.assign_coords(alt=alt_km)
    return dbzh


def compute_echotop(dbzh: xr.DataArray, threshold_dbz: float, alt_km: np.ndarray, name: str) -> xr.DataArray:
    dbzh = ensure_alt(dbzh, alt_km)
    alt_m = (alt_km * 1000.0).astype(np.float32)
    nalt = len(alt_m)

    if dbzh.sizes["alt"] != nalt:
        raise ValueError("DBZH alt size does not match alt_km length")

    template = dbzh.isel(alt=0, drop=True)
    echotop = xr.zeros_like(template, dtype=np.float32)

    top_mask = dbzh.isel(alt=-1) >= threshold_dbz
    echotop = xr.where(top_mask, alt_m[-1], echotop)

    for i in range(nalt - 2, -1, -1):
        lower = dbzh.isel(alt=i)
        upper = dbzh.isel(alt=i + 1)
        h_lower = np.float32(alt_m[i])
        h_upper = np.float32(alt_m[i + 1])

        cross_mask = (lower >= threshold_dbz) & (upper < threshold_dbz) & (echotop == 0)
        denom = upper - lower
        interp = h_lower + (h_upper - h_lower) * (threshold_dbz - lower) / xr.where(denom == 0, np.nan, denom)
        interp = xr.where(np.isfinite(interp), interp, h_lower)
        echotop = xr.where(cross_mask, interp.astype(np.float32), echotop)

    echotop.name = name
    echotop.attrs["units"] = "m"
    return echotop


def compute_vil(dbzh: xr.DataArray, alt_km: np.ndarray) -> xr.DataArray:
    dbzh = ensure_alt(dbzh, alt_km)

    alt_m = (alt_km * 1000.0).astype(np.float32)
    dh = np.diff(alt_m, prepend=alt_m[0])
    if len(dh) > 1:
        dh[0] = dh[1]
    dh_km = xr.DataArray(dh / 1000.0, dims=["alt"], coords={"alt": dbzh["alt"]})

    dbzh_valid = xr.where(dbzh > VIL_DBZ_FLOOR, dbzh, np.nan)
    z_linear = 10.0 ** (0.1 * dbzh_valid)
    vil_layer = xr.where(np.isfinite(dbzh_valid), VIL_MFAC * (z_linear ** (4.0 / 7.0)) * dh_km, 0.0)

    vil = vil_layer.sum(dim="alt", skipna=True).astype(np.float32)
    vil.name = "VIL"
    vil.attrs["units"] = "kg m-2"
    return vil


def make_groups(file_list: list[Path], mode: str) -> list[list[Path]]:
    if mode not in {"daily", "monthly", "all"}:
        raise ValueError(f"BATCH_MODE must be one of: daily, monthly, all. Got: {mode}")
    buckets: dict[str, list[Path]] = defaultdict(list)
    for path in file_list:
        t = parse_file_time(path)
        if mode == "daily":
            key = t.strftime("%Y-%m-%d")
        elif mode == "monthly":
            key = t.strftime("%Y-%m")
        else:
            key = "all"
        buckets[key].append(path)
    return [sorted(v) for _, v in sorted(buckets.items(), key=lambda kv: kv[0])]


def build_dataset_for_group(group_files: list[Path], lat: np.ndarray, lon: np.ndarray, alt_km: np.ndarray) -> xr.Dataset:
    times = pd.DatetimeIndex([parse_file_time(p) for p in group_files])

    ysize = len(lat)
    xsize = len(lon)
    zsize = len(alt_km)

    lazy_pairs = [load_one_file(p) for p in group_files]
    lazy_max = [da.from_delayed(item[0], shape=(ysize, xsize), dtype=np.float32) for item in lazy_pairs]
    lazy_dbz = [da.from_delayed(item[1], shape=(zsize, ysize, xsize), dtype=np.float32) for item in lazy_pairs]

    max_array = da.stack(lazy_max, axis=0)
    dbz_array = da.stack(lazy_dbz, axis=0)

    ds_raw = xr.Dataset(
        data_vars={
            "MAXDBZ": xr.DataArray(max_array, dims=["time", "lat", "lon"]),
            "DBZH": xr.DataArray(dbz_array, dims=["time", "alt", "lat", "lon"]),
        },
        coords={"time": times, "alt": alt_km, "lat": lat, "lon": lon},
    )

    dbzh_chunks = {
        "time": min(DBZH_CHUNK_TIME, len(times)),
        "alt": min(DBZH_CHUNK_ALT, zsize),
        "lat": min(DBZH_CHUNK_LAT, ysize),
        "lon": min(DBZH_CHUNK_LON, xsize),
    }
    maxdbz_chunks = {
        "time": dbzh_chunks["time"],
        "lat": dbzh_chunks["lat"],
        "lon": dbzh_chunks["lon"],
    }

    dbzh = ds_raw["DBZH"].chunk(dbzh_chunks)
    maxdbz = ds_raw["MAXDBZ"].chunk(maxdbz_chunks)

    top18 = compute_echotop(dbzh, threshold_dbz=18.0, alt_km=alt_km, name="TOP18")
    top45 = compute_echotop(dbzh, threshold_dbz=45.0, alt_km=alt_km, name="TOP45")
    vil = compute_vil(dbzh, alt_km=alt_km)

    ds_out = xr.Dataset(
        data_vars={
            "MAXDBZ": maxdbz,
            "DBZH": dbzh,
            "TOP18": top18,
            "TOP45": top45,
            "VIL": vil,
        },
        coords=ds_raw.coords,
        attrs={
            "Conventions": "CF-1.8",
            "title": "QPESUMS 3D Radar Composite with Derived Fields",
            "source": "CWB QPESUMS MREF3D21L HDF5",
            "time_step": "10 minutes",
        },
    )
    return ds_out


def main() -> None:
    if not HDF5_DIR.exists():
        raise FileNotFoundError(f"HDF5_DIR not found: {HDF5_DIR}")

    all_files = sorted(HDF5_DIR.glob(HDF5_PATTERN))
    if not all_files:
        raise FileNotFoundError(f"No files found with pattern '{HDF5_PATTERN}' under {HDF5_DIR}")

    if START_TIME is not None:
        start_ts = pd.Timestamp(START_TIME)
        all_files = [f for f in all_files if pd.Timestamp(parse_file_time(f)) >= start_ts]
    if END_TIME is not None:
        end_ts = pd.Timestamp(END_TIME)
        all_files = [f for f in all_files if pd.Timestamp(parse_file_time(f)) <= end_ts]
    if not all_files:
        raise RuntimeError("No files left after START_TIME/END_TIME filtering")

    sample_file = all_files[0]
    lat, lon, alt_km = read_grid(sample_file)

    if OUTPUT_ZARR.exists():
        if OVERWRITE_OUTPUT:
            print(f"[INFO] Remove existing output: {OUTPUT_ZARR}")
            shutil.rmtree(OUTPUT_ZARR)
        else:
            raise FileExistsError(f"Output exists: {OUTPUT_ZARR} (set OVERWRITE_OUTPUT=True to replace)")

    groups = make_groups(all_files, BATCH_MODE)

    print(f"[INFO] files      : {len(all_files)}")
    print(f"[INFO] time range : {parse_file_time(all_files[0])} -> {parse_file_time(all_files[-1])}")
    print(f"[INFO] batch mode : {BATCH_MODE} ({len(groups)} groups)")
    print(
        "[INFO] DBZH chunk : "
        f"({DBZH_CHUNK_TIME}, {DBZH_CHUNK_ALT}, {DBZH_CHUNK_LAT}, {DBZH_CHUNK_LON}) "
        "(time, alt, lat, lon; clipped by actual shape)"
    )
    print(f"[INFO] output     : {OUTPUT_ZARR}")

    first_group = True
    t0 = time.time()
    client = None
    cluster = None

    try:
        if USE_DISTRIBUTED:
            try:
                from dask.distributed import Client, LocalCluster
            except Exception as exc:
                raise RuntimeError(
                    "USE_DISTRIBUTED=True but dask.distributed is unavailable. "
                    "Install with: pip install 'dask[distributed]'"
                ) from exc

            cluster = LocalCluster(
                n_workers=N_WORKERS,
                threads_per_worker=THREADS_PER_WORKER,
                memory_limit=MEMORY_LIMIT_PER_WORKER,
                dashboard_address=DASK_DASHBOARD_ADDRESS,
            )
            client = Client(cluster)
            print(f"[INFO] Dask distributed enabled: {client}")
            print(f"[INFO] Dashboard: {client.dashboard_link}")

        with dask.config.set(scheduler=DASK_SCHEDULER):
            for i, group in enumerate(groups, start=1):
                t1 = time.time()
                g_start = parse_file_time(group[0])
                g_end = parse_file_time(group[-1])
                print(f"\n[GROUP {i}/{len(groups)}] files={len(group)}  {g_start} -> {g_end}")

                ds_group = build_dataset_for_group(group, lat=lat, lon=lon, alt_km=alt_km)

                write_mode = "w" if first_group else "a"
                append_dim = None if first_group else "time"

                with ProgressBar():
                    ds_group.to_zarr(
                        OUTPUT_ZARR,
                        mode=write_mode,
                        append_dim=append_dim,
                        zarr_format=ZARR_FORMAT,
                        consolidated=False,
                        align_chunks=True,
                    )

                first_group = False
                del ds_group
                gc.collect()
                print(f"[GROUP {i}] done in {time.time() - t1:.1f}s")
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()

    try:
        import zarr

        zarr.consolidate_metadata(str(OUTPUT_ZARR))
        print("[INFO] metadata consolidated")
    except Exception as exc:
        print(f"[WARN] skip consolidate_metadata: {exc}")

    print(f"\n[DONE] elapsed: {time.time() - t0:.1f}s")
    print(f"[DONE] output : {OUTPUT_ZARR}")


if __name__ == "__main__":
    main()
