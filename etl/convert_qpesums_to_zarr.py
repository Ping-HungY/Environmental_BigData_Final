from __future__ import annotations

import re
import shutil
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import Blosc


# =========================================================
# User settings
# =========================================================
INPUT_DIR = Path(r"E:\Data\3D_QPESUMS\All_Variables_10min_data\2024_all_day_h5")   # 改成你的 HDF5 資料夾
MISSING_CSV = Path(r"E:\Data\3D_QPESUMS\Missing_timestamps\Missing_timestamps_2024.csv")  # 改成你的 csv 路徑
OUT_DIR = Path(r"G:\EnviroBigData_Data")     # 輸出資料夾

YEAR = 2024
OVERWRITE = True
BATCH_SIZE = 48   # 可先試 48 / 72 / 96

# HDF5 dataset keys
DBZH_KEY = "dataset2/data1/data"
MAXDBZ_KEY = "dataset1/data1/data"

FNAME_PATTERN = re.compile(r"MREF3D21L\.(\d{8})\.(\d{4})\.hdf5$")

# 建議先用較快壓縮；若還嫌慢可先拿掉 compressor
COMPRESSOR = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
OUT_DTYPE = np.float32


# =========================================================
# Chunk strategies
# =========================================================
DBZH_STRATEGIES = {
    "t15_tile128":   {"time": 15, "z": 21, "y": 128, "x": 128},
    "t90_tile128":   {"time": 90, "z": 21, "y": 128, "x": 128},
    "t1_fullvolume": {"time": 1,  "z": 21, "y": 561, "x": 441},
    "t6_fullvolume": {"time": 6,  "z": 21, "y": 561, "x": 441},
}

MAXDBZ_STRATEGIES = {
    "pysteps":      {"time": 24, "y": 561, "x": 441},
    "balanced":     {"time": 48, "y": 256, "x": 256},
    "spatialtiles": {"time": 6,  "y": 128, "x": 128},
}


# =========================================================
# Utils
# =========================================================
def parse_timestamp_from_filename(path: Path) -> pd.Timestamp:
    m = FNAME_PATTERN.search(path.name)
    if m is None:
        raise ValueError(f"Bad filename format: {path.name}")
    ymd, hm = m.groups()
    return pd.Timestamp(datetime.strptime(ymd + hm, "%Y%m%d%H%M"))


def list_hdf5_files(input_dir: Path, year: int) -> list[Path]:
    files = sorted(input_dir.glob("MREF3D21L.*.hdf5"))
    out = []
    for f in files:
        m = FNAME_PATTERN.search(f.name)
        if m is None:
            continue
        ymd, _ = m.groups()
        if int(ymd[:4]) == year:
            out.append(f)
    return out


def load_missing_timestamps(csv_path: Path) -> set[pd.Timestamp]:
    df = pd.read_csv(csv_path)
    if "missing_timestamps" not in df.columns:
        raise ValueError("CSV must contain column: missing_timestamps")
    ts = pd.to_datetime(df["missing_timestamps"])
    return set(ts)


def filter_valid_files(files: list[Path], missing_ts: set[pd.Timestamp]) -> list[Path]:
    valid = []
    dropped = 0
    for f in files:
        ts = parse_timestamp_from_filename(f)
        if ts in missing_ts:
            dropped += 1
        else:
            valid.append(f)

    print(f"[INFO] total files      = {len(files)}")
    print(f"[INFO] missing excluded = {dropped}")
    print(f"[INFO] valid files      = {len(valid)}")
    return valid


def read_two_vars(path: Path, dbzh_key: str, maxdbz_key: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as hdf:
        dbzh = np.asarray(hdf[dbzh_key][:], dtype=OUT_DTYPE)     # (z, y, x)
        maxdbz = np.asarray(hdf[maxdbz_key][:], dtype=OUT_DTYPE) # (y, x)
    return dbzh, maxdbz


def check_shapes(files: list[Path]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dbzh0, max0 = read_two_vars(files[0], DBZH_KEY, MAXDBZ_KEY)
    dbzh_shape = dbzh0.shape
    max_shape = max0.shape

    for f in files[1:min(20, len(files))]:
        dbzh, maxdbz = read_two_vars(f, DBZH_KEY, MAXDBZ_KEY)
        if dbzh.shape != dbzh_shape:
            raise ValueError(f"DBZH shape mismatch: {f.name}, {dbzh.shape} != {dbzh_shape}")
        if maxdbz.shape != max_shape:
            raise ValueError(f"MAXDBZ shape mismatch: {f.name}, {maxdbz.shape} != {max_shape}")

    return dbzh_shape, max_shape


def make_ds_dbzh(data_list: list[np.ndarray], time_list: list[pd.Timestamp]) -> xr.Dataset:
    arr = np.stack(data_list, axis=0).astype(OUT_DTYPE, copy=False)  # (time, z, y, x)
    _, nz, ny, nx = arr.shape

    ds = xr.Dataset(
        data_vars={"DBZH": (("time", "z", "y", "x"), arr)},
        coords={
            "time": np.array(time_list, dtype="datetime64[ns]"),
            "z": np.arange(nz, dtype=np.int32),
            "y": np.arange(ny, dtype=np.int32),
            "x": np.arange(nx, dtype=np.int32),
        },
        attrs={"source_product": "QPESUMS-3D", "variable": "DBZH", "year": YEAR},
    )
    return ds


def make_ds_maxdbz(data_list: list[np.ndarray], time_list: list[pd.Timestamp]) -> xr.Dataset:
    arr = np.stack(data_list, axis=0).astype(OUT_DTYPE, copy=False)  # (time, y, x)
    _, ny, nx = arr.shape

    ds = xr.Dataset(
        data_vars={"MAXDBZ": (("time", "y", "x"), arr)},
        coords={
            "time": np.array(time_list, dtype="datetime64[ns]"),
            "y": np.arange(ny, dtype=np.int32),
            "x": np.arange(nx, dtype=np.int32),
        },
        attrs={"source_product": "QPESUMS-3D", "variable": "MAXDBZ", "year": YEAR},
    )
    return ds


# =========================================================
# Writer helpers
# =========================================================
def init_output_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = []
    for name in DBZH_STRATEGIES:
        all_paths.append(OUT_DIR / f"dbzh_{name}.zarr")
    for name in MAXDBZ_STRATEGIES:
        all_paths.append(OUT_DIR / f"maxdbz_{name}.zarr")

    if OVERWRITE:
        for p in all_paths:
            if p.exists():
                print(f"[INFO] remove existing store: {p}")
                shutil.rmtree(p)


def write_dbzh_batch(ds_batch: xr.Dataset, first_write_flags: dict[str, bool]):
    for strategy_name, chunks in DBZH_STRATEGIES.items():
        out_path = OUT_DIR / f"dbzh_{strategy_name}.zarr"

        ds = ds_batch.chunk({
            "time": chunks["time"],
            "z": chunks["z"],
            "y": chunks["y"],
            "x": chunks["x"],
        })

        first_write = first_write_flags[strategy_name]
        mode = "w" if first_write else "a"
        append_dim = None if first_write else "time"
        encoding = (
            {"DBZH": {"compressor": COMPRESSOR, "dtype": "float32"}}
            if first_write else None
        )

        ds.to_zarr(
            out_path,
            mode=mode,
            append_dim=append_dim,
            consolidated=True,
            encoding=encoding,
            align_chunks=True
        )
        first_write_flags[strategy_name] = False


def write_maxdbz_batch(ds_batch: xr.Dataset, first_write_flags: dict[str, bool]):
    for strategy_name, chunks in MAXDBZ_STRATEGIES.items():
        out_path = OUT_DIR / f"maxdbz_{strategy_name}.zarr"

        ds = ds_batch.chunk({
            "time": chunks["time"],
            "y": chunks["y"],
            "x": chunks["x"],
        })

        first_write = first_write_flags[strategy_name]
        mode = "w" if first_write else "a"
        append_dim = None if first_write else "time"
        encoding = (
            {"MAXDBZ": {"compressor": COMPRESSOR, "dtype": "float32"}}
            if first_write else None
        )

        ds.to_zarr(
            out_path,
            mode=mode,
            append_dim=append_dim,
            consolidated=True,
            encoding=encoding,
            align_chunks=True
        )
        first_write_flags[strategy_name] = False


# =========================================================
# Main
# =========================================================
def main():
    init_output_dirs()

    files_all = list_hdf5_files(INPUT_DIR, YEAR)
    missing_ts = load_missing_timestamps(MISSING_CSV)
    files = filter_valid_files(files_all, missing_ts)

    if len(files) == 0:
        raise RuntimeError("No valid files found.")

    dbzh_shape, max_shape = check_shapes(files)
    print(f"[INFO] DBZH shape per file   = {dbzh_shape}")
    print(f"[INFO] MAXDBZ shape per file = {max_shape}")

    dbzh_first_write = {k: True for k in DBZH_STRATEGIES}
    maxdbz_first_write = {k: True for k in MAXDBZ_STRATEGIES}

    batch_dbzh = []
    batch_maxdbz = []
    batch_time = []

    total_written = 0

    for i, f in enumerate(files, start=1):
        try:
            ts = parse_timestamp_from_filename(f)
            dbzh, maxdbz = read_two_vars(f, DBZH_KEY, MAXDBZ_KEY)

            batch_dbzh.append(dbzh)
            batch_maxdbz.append(maxdbz)
            batch_time.append(ts)

            do_flush = (len(batch_time) >= BATCH_SIZE) or (i == len(files))
            if not do_flush:
                continue

            # Build once per variable, write to all strategies
            ds_dbzh = make_ds_dbzh(batch_dbzh, batch_time)
            ds_maxdbz = make_ds_maxdbz(batch_maxdbz, batch_time)

            write_dbzh_batch(ds_dbzh, dbzh_first_write)
            write_maxdbz_batch(ds_maxdbz, maxdbz_first_write)

            total_written += len(batch_time)
            print(f"[INFO] written total time steps = {total_written}")

            batch_dbzh.clear()
            batch_maxdbz.clear()
            batch_time.clear()

        except Exception as e:
            print(f"[ERROR] file={f.name}, err={e}")

    print(f"[DONE] All {len(DBZH_STRATEGIES) + len(MAXDBZ_STRATEGIES)} Zarr stores finished.")


if __name__ == "__main__":
    main()