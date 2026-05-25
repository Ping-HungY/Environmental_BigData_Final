#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import xarray as xr

# ============================================================
# USER SETTINGS (edit here only)
# ============================================================
SOURCE_ZARR = Path(
    "/home/NAS/homes/pinghung-10018/Course_data/Isungche/Final_project/radar_taiwan_2024_with_derived.zarr"
)
OUTPUT_ROOT = Path(
    "/home/NAS/homes/pinghung-10018/Course_data/Isungche/Final_project/event_exports"
)

# Event times below are in UTC+8 date range (inclusive day boundaries).
EVENTS_UTC8 = [
    ("Gaemi", "2024-07-22", "2024-07-26"),
    ("Krathon", "2024-09-29", "2024-10-04"),
    ("KONG-REY", "2024-10-29", "2024-11-01"),
    ("Usagi", "2024-11-14", "2024-11-16"),
]

# Radar time is UTC+0, event dates are UTC+8.
UTC8_TO_UTC0_HOURS = 8

# Per your requirement: "多給 8 小時資料"
# Final UTC range = (UTC+8 converted to UTC+0) + EXTRA_HOURS_AFTER_CONVERSION on end side.
EXTRA_HOURS_AFTER_CONVERSION = 8

OVERWRITE_EVENT_FOLDER = True
ZARR_FORMAT = 3
TARGET_CHUNK_TIME = 48
TARGET_CHUNK_SPATIAL = 256


def make_safe_name(name: str) -> str:
    return (
        name.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def utc_window_from_utc8_dates(start_date_utc8: str, end_date_utc8: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_local = pd.Timestamp(f"{start_date_utc8} 00:00:00")
    end_local = pd.Timestamp(f"{end_date_utc8} 23:59:59")

    start_utc = start_local - pd.Timedelta(hours=UTC8_TO_UTC0_HOURS)
    end_utc = end_local - pd.Timedelta(hours=UTC8_TO_UTC0_HOURS) + pd.Timedelta(
        hours=EXTRA_HOURS_AFTER_CONVERSION
    )
    return start_utc, end_utc


def export_one_variable(
    ds: xr.Dataset,
    var_name: str,
    out_zarr: Path,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> None:
    if var_name not in ds:
        raise KeyError(f"Variable not found in source zarr: {var_name}")

    da = ds[var_name].sel(time=slice(start_utc, end_utc))
    if da.sizes.get("time", 0) == 0:
        raise ValueError(
            f"[{var_name}] empty time slice for UTC window {start_utc} ~ {end_utc}. "
            "No data matched; please check event dates/timezone against source dataset range."
        )

    chunk_map: dict[str, int] = {}
    if "time" in da.dims:
        chunk_map["time"] = TARGET_CHUNK_TIME

    # Prefer geographic names, then grid names.
    spatial_candidates = [("lat", "lon"), ("y", "x")]
    spatial_dims: tuple[str, str] | None = None
    for d1, d2 in spatial_candidates:
        if d1 in da.dims and d2 in da.dims:
            spatial_dims = (d1, d2)
            break

    # Fallback: use first two non-time dimensions if needed.
    if spatial_dims is None:
        non_time_dims = [d for d in da.dims if d != "time"]
        if len(non_time_dims) >= 2:
            spatial_dims = (non_time_dims[0], non_time_dims[1])

    if spatial_dims is not None:
        chunk_map[spatial_dims[0]] = TARGET_CHUNK_SPATIAL
        chunk_map[spatial_dims[1]] = TARGET_CHUNK_SPATIAL

    if chunk_map:
        da = da.chunk(chunk_map)

    out_ds = da.to_dataset(name=var_name)

    # Force output Zarr chunk shape instead of inheriting source encoding chunks.
    target_chunks_by_dim = {}
    for dim in da.dims:
        if dim == "time":
            target_chunks_by_dim[dim] = TARGET_CHUNK_TIME
        else:
            target_chunks_by_dim[dim] = TARGET_CHUNK_SPATIAL
    target_chunk_tuple = tuple(target_chunks_by_dim[dim] for dim in da.dims)
    out_ds[var_name].encoding = {"chunks": target_chunk_tuple}
    if "time" in out_ds.coords:
        time_len = out_ds.sizes["time"]
        out_ds["time"].encoding = {"chunks": (min(TARGET_CHUNK_TIME, time_len),)}
    for coord_name in ("lat", "lon", "y", "x"):
        if coord_name in out_ds.coords and coord_name in out_ds.sizes:
            coord_len = out_ds.sizes[coord_name]
            out_ds[coord_name].encoding = {"chunks": (min(TARGET_CHUNK_SPATIAL, coord_len),)}

    if out_zarr.exists():
        shutil.rmtree(out_zarr)

    out_ds.to_zarr(
        out_zarr,
        mode="w",
        consolidated=True,
        zarr_format=ZARR_FORMAT,
        align_chunks=True,
    )


def main() -> None:
    if not SOURCE_ZARR.exists():
        raise FileNotFoundError(f"SOURCE_ZARR not found: {SOURCE_ZARR}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(SOURCE_ZARR, consolidated=True)

    for event_name, start_date, end_date in EVENTS_UTC8:
        safe_event_name = make_safe_name(event_name)
        event_dir = OUTPUT_ROOT / safe_event_name

        if event_dir.exists() and OVERWRITE_EVENT_FOLDER:
            shutil.rmtree(event_dir)
        event_dir.mkdir(parents=True, exist_ok=True)

        start_utc, end_utc = utc_window_from_utc8_dates(start_date, end_date)
        print(
            f"[{event_name}] UTC+8 {start_date} ~ {end_date} "
            f"=> UTC+0 {start_utc} ~ {end_utc}"
        )

        export_one_variable(ds, "VIL", event_dir / "VIL.zarr", start_utc, end_utc)
        export_one_variable(ds, "TOP18", event_dir / "EchoTop_18.zarr", start_utc, end_utc)
        export_one_variable(ds, "TOP45", event_dir / "EchoTop_45.zarr", start_utc, end_utc)

        print(f"[{event_name}] done -> {event_dir}")

    print(f"\nAll events exported under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
