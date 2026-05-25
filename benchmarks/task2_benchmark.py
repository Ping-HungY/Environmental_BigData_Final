from __future__ import annotations

import re
import time
from pathlib import Path

import dask
import distributed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster, performance_report


# =========================================================
# User settings
# =========================================================
ROOT = Path(r"G:\EnviroBigData_Data")
STATION_CSV = Path(r"E:\Data\RG_CIRRUS\Metadata\CWA_Study_Stations_Info_250326_v2.csv")

STORES = [
    "dbzh_t15_tile128.zarr",
    "dbzh_t90_tile128.zarr",
    "dbzh_t1_fullvolume.zarr",
    "dbzh_t6_fullvolume.zarr",
]

REPORT_DIR = Path("./cfad_reports")
FIG_DIR = Path("./Figures")
CSV_PATH = Path("./task2_benchmark_summary.csv")

STATION_BATCH_SIZE = 5

DBZ_EDGES = np.arange(-5, 76, 1)
DBZ_CENTERS = 0.5 * (DBZ_EDGES[:-1] + DBZ_EDGES[1:])

LON_MIN_FULL, LON_MAX_FULL = 118.0, 123.5
LAT_MIN_FULL, LAT_MAX_FULL = 20.0, 27.0

N_WORKERS = 4
THREADS_PER_WORKER = 1
DASHBOARD_ADDRESS = ":8787"


# =========================================================
# Helpers
# =========================================================
def load_selected_stations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df["KNN_val"] == 1].copy().reset_index(drop=True)


def build_qpesums_latlon_1d(
    nx: int = 441,
    ny: int = 561,
    lon_min: float = 118.0,
    lon_max: float = 123.5,
    lat_min: float = 20.0,
    lat_max: float = 27.0,
) -> tuple[np.ndarray, np.ndarray]:
    lon_1d = np.linspace(lon_min, lon_max, nx)
    lat_1d = np.linspace(lat_min, lat_max, ny)
    return lon_1d, lat_1d


def nearest_index_1d(values_1d: np.ndarray, targets: np.ndarray) -> np.ndarray:
    values_1d = np.asarray(values_1d)
    targets = np.asarray(targets)
    return np.abs(values_1d[None, :] - targets[:, None]).argmin(axis=1)


def prepare_station_indices(df_station: pd.DataFrame, ds: xr.Dataset) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    nx = ds.sizes["x"]
    ny = ds.sizes["y"]

    lon_1d, lat_1d = build_qpesums_latlon_1d(
        nx=nx,
        ny=ny,
        lon_min=LON_MIN_FULL,
        lon_max=LON_MAX_FULL,
        lat_min=LAT_MIN_FULL,
        lat_max=LAT_MAX_FULL,
    )

    x_idx = nearest_index_1d(lon_1d, df_station["Longitude_wgs84"].values)
    y_idx = nearest_index_1d(lat_1d, df_station["Latitude_wgs84"].values)

    df_out = df_station.copy()
    df_out["x_idx"] = x_idx
    df_out["y_idx"] = y_idx
    return df_out, lon_1d, lat_1d


def update_cfad_count(cfad_count: np.ndarray, values_3d: np.ndarray, dbz_edges: np.ndarray) -> None:
    _, nz, _ = values_3d.shape
    for iz in range(nz):
        arr = values_3d[:, iz, :].ravel()
        arr = arr[np.isfinite(arr)]
        hist, _ = np.histogram(arr, bins=dbz_edges)
        cfad_count[iz, :] += hist


def finalize_cfad_freq(cfad_count: np.ndarray) -> np.ndarray:
    row_sum = cfad_count.sum(axis=1, keepdims=True)
    return np.divide(
        cfad_count,
        row_sum,
        out=np.zeros_like(cfad_count, dtype=float),
        where=row_sum > 0,
    )


def _parse_time_to_seconds(text: str) -> float:
    text = text.strip()

    m = re.search(r"([0-9.]+)\s*ms", text)
    if m:
        return float(m.group(1)) / 1000.0

    m = re.search(r"([0-9.]+)\s*min\s*([0-9.]+)\s*s", text)
    if m:
        return float(m.group(1)) * 60.0 + float(m.group(2))

    m = re.search(r"([0-9.]+)\s*s", text)
    if m:
        return float(m.group(1))

    return np.nan


def parse_dask_report_summary(html_path: Path) -> dict:
    text = html_path.read_text(encoding="utf-8", errors="ignore")

    out = {
        "report_duration_s": np.nan,
        "n_tasks": np.nan,
        "compute_time_s": np.nan,
        "transfer_time_s": np.nan,
    }

    m = re.search(r"Duration:\s*([^<]+)", text)
    if m:
        out["report_duration_s"] = _parse_time_to_seconds(m.group(1))

    m = re.search(r"number of tasks:\s*([0-9]+)", text)
    if m:
        out["n_tasks"] = int(m.group(1))

    m = re.search(r"compute time:\s*([^<]+)", text)
    if m:
        out["compute_time_s"] = _parse_time_to_seconds(m.group(1))

    m = re.search(r"transfer time:\s*([^<]+)", text)
    if m:
        out["transfer_time_s"] = _parse_time_to_seconds(m.group(1))

    return out


def run_cfad_benchmark(
    store_path: Path,
    df_station: pd.DataFrame,
    dbz_edges: np.ndarray,
    report_dir: Path,
    station_batch_size: int = 5,
) -> dict:
    t0 = time.perf_counter()
    store_name = store_path.stem
    report_path = report_dir / f"report_{store_name}.html"

    with performance_report(filename=str(report_path)):
        ds = xr.open_zarr(store_path, consolidated=False)
        da = ds["DBZH"]

        df_idx, _, _ = prepare_station_indices(df_station, ds)

        nz = int(da.sizes["z"])
        n_bins = len(dbz_edges) - 1
        cfad_count = np.zeros((nz, n_bins), dtype=np.int64)

        for i0 in range(0, len(df_idx), station_batch_size):
            sub = df_idx.iloc[i0:i0 + station_batch_size].copy()

            x_indexer = xr.DataArray(sub["x_idx"].to_numpy(), dims="station")
            y_indexer = xr.DataArray(sub["y_idx"].to_numpy(), dims="station")

            prof_batch = da.isel(y=y_indexer, x=x_indexer).load()
            values_batch = prof_batch.values
            update_cfad_count(cfad_count, values_batch, dbz_edges)

        cfad_freq = finalize_cfad_freq(cfad_count)

    elapsed = time.perf_counter() - t0

    processed_bytes = int(da.sizes["time"]) * int(da.sizes["z"]) * int(len(df_idx)) * 4
    throughput_MBps = processed_bytes / 1e6 / elapsed

    time.sleep(0.5)
    report_summary = parse_dask_report_summary(report_path)

    return {
        "seconds": elapsed,
        "throughput_MBps": throughput_MBps,
        "n_time": int(da.sizes["time"]),
        "n_z": int(da.sizes["z"]),
        "n_station": int(len(df_idx)),
        "cfad_count": cfad_count,
        "cfad_freq": cfad_freq,
        "dbz_centers": DBZ_CENTERS,
        "z_levels": da["z"].values,
        "station_table": df_idx,
        "report_path": str(report_path),
        "n_tasks": report_summary["n_tasks"],
        "compute_time_s": report_summary["compute_time_s"],
        "transfer_time_s": report_summary["transfer_time_s"] * 1000 if pd.notna(report_summary["transfer_time_s"]) else np.nan,
        "report_duration_s": report_summary["report_duration_s"],
    }


def build_plot_labels(df: pd.DataFrame) -> list[str]:
    mapping = {
        "dbzh_t15_tile128.zarr": "t15_tile128\n(strategy A)",
        "dbzh_t90_tile128.zarr": "t90_tile128\n(strategy B)",
        "dbzh_t1_fullvolume.zarr": "t1_fullvolume\n(strategy C)",
        "dbzh_t6_fullvolume.zarr": "t6_fullvolume\n(strategy D)",
    }
    return [mapping[s] for s in df["store"]]


def make_figures(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fontsize = 18
    dpi = 300
    plot_df = df.copy()
    plot_df["strategy"] = build_plot_labels(plot_df)

    # Figure 1
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=dpi)
    x = np.arange(len(plot_df))
    w = 0.32

    ax1.bar(
        x - w / 2,
        plot_df["seconds"],
        width=w,
        color="steelblue",
        edgecolor="black",
        label="Duration / Latency",
        zorder=1,
    )
    ax1.bar(
        x + w / 2,
        plot_df["compute_time_s"],
        width=w,
        color="lightgreen",
        edgecolor="black",
        label="Compute time",
        zorder=1,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["strategy"])
    ax1.set_ylabel("Seconds", fontsize=fontsize)
    ax1.set_xlabel("Chunking strategy", fontsize=fontsize)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        plot_df["transfer_time_ms"],
        marker="o",
        linewidth=2.5,
        color="tab:red",
        label="Transfer time",
        zorder=3,
    )
    ax2.set_ylabel("Transfer time (ms)", fontsize=fontsize)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    plt.title("Dask performance", fontsize=fontsize + 5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Task_2_dask_performance.png", dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close()

    # Figure 2
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=dpi)
    x = range(len(plot_df))

    ax1.bar(
        x,
        plot_df["n_tasks"],
        width=0.55,
        color="lightgray",
        edgecolor="black",
        label="Number of tasks",
        zorder=1,
    )
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(plot_df["strategy"])
    ax1.set_ylabel("Number of tasks", fontsize=fontsize)
    ax1.set_xlabel("Chunking strategy", fontsize=fontsize)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        plot_df["transfer_time_ms"],
        marker="o",
        linewidth=2.5,
        color="tab:red",
        label="Transfer time",
        zorder=3,
    )
    ax2.set_ylabel("Transfer time (ms)", fontsize=fontsize)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False)

    plt.title("Task granularity and transfer overhead", fontsize=fontsize + 5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Task_2_task_granularity.png", dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(dask.__version__)
    print(distributed.__version__)

    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        dashboard_address=DASHBOARD_ADDRESS,
    )
    client = Client(cluster)

    try:
        df_station = load_selected_stations(STATION_CSV)
        print(f"Selected stations (KNN_val == 1): {len(df_station)}")

        results = []
        cfad_outputs = {}

        for store in STORES:
            out = run_cfad_benchmark(
                ROOT / store,
                df_station,
                DBZ_EDGES,
                report_dir=REPORT_DIR,
                station_batch_size=STATION_BATCH_SIZE,
            )

            cfad_outputs[store] = out
            results.append({
                "store": store,
                "seconds": out["seconds"],
                "throughput_MBps": out["throughput_MBps"],
                "report_duration_s": out["report_duration_s"],
                "n_tasks": out["n_tasks"],
                "compute_time_s": out["compute_time_s"],
                "transfer_time_ms": out["transfer_time_s"],
                "n_time": out["n_time"],
                "n_z": out["n_z"],
                "n_station": out["n_station"],
                "report_path": out["report_path"],
            })

            print(
                f"{store}: "
                f"{out['seconds']:.3f} s | "
                f"tasks={out['n_tasks']} | "
                f"compute={out['compute_time_s']:.3f} s | "
                f"transfer={out['transfer_time_s']:.3f} ms"
            )

        df_bench_cfad = pd.DataFrame(results).sort_values("seconds").reset_index(drop=True)
        print(df_bench_cfad)
        df_bench_cfad.to_csv(CSV_PATH, index=False)
        make_figures(df_bench_cfad)
        print(f"[OK] saved summary -> {CSV_PATH}")
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
