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
ROOT = Path(r"G:/EnviroBigData_Data")
STORES = [
    "maxdbz_pysteps.zarr",
    "maxdbz_balanced.zarr",
    "maxdbz_spatialtiles.zarr",
]

TIME_START_LOCAL = "2024-10-29 17:30:00"
TIME_END_LOCAL = "2024-11-01 14:30:00"

LAT_MIN, LAT_MAX = 21.9, 25.3
LON_MIN, LON_MAX = 120.0, 122.0

ZR_A = 32.5
ZR_B = 1.65
DT_HOURS = 10 / 60

N_WORKERS = 4
THREADS_PER_WORKER = 1
DASHBOARD_ADDRESS = ":8787"

REPORT_DIR = Path("./task1_reports")
FIG_DIR = Path("./Figures")
CSV_PATH = Path("./task1_benchmark_summary.csv")


# =========================================================
# Helpers
# =========================================================
def dbz_to_rainrate(dbz: xr.DataArray) -> xr.DataArray:
    z_lin = 10.0 ** (dbz / 10.0)
    rain = (z_lin / ZR_A) ** (1.0 / ZR_B)
    return rain


def attach_latlon_if_missing(ds: xr.Dataset) -> xr.Dataset:
    has_lat = ("lat" in ds.coords) or ("lat" in ds.data_vars)
    has_lon = ("lon" in ds.coords) or ("lon" in ds.data_vars)
    if has_lat and has_lon:
        return ds

    ny = ds.sizes["y"]
    nx = ds.sizes["x"]
    lat_1d = np.linspace(20.0, 27.0, ny)
    lon_1d = np.linspace(118.0, 123.5, nx)
    return ds.assign_coords(lat=("y", lat_1d), lon=("x", lon_1d))


def subset_taiwan_roi(ds: xr.Dataset, time_start_utc: pd.Timestamp, time_end_utc: pd.Timestamp) -> xr.Dataset:
    ds = attach_latlon_if_missing(ds)
    return ds.sel(
        time=slice(time_start_utc, time_end_utc),
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX),
    )


def open_zarr_auto(path: Path) -> xr.Dataset:
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


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


def benchmark_one_store(store: str, time_start_utc: pd.Timestamp, time_end_utc: pd.Timestamp) -> dict:
    path = ROOT / store
    report_path = REPORT_DIR / f"report_{store.replace('.zarr', '')}.html"

    with performance_report(filename=str(report_path)):
        t0 = time.perf_counter()

        ds = open_zarr_auto(path)
        ds_roi = subset_taiwan_roi(ds, time_start_utc, time_end_utc)

        maxdbz = ds_roi["MAXDBZ"]
        rainrate = dbz_to_rainrate(maxdbz)
        accum_mm = (rainrate * DT_HOURS).sum(dim="time").compute()

        elapsed = time.perf_counter() - t0

    data_mb = maxdbz.nbytes / 1e6
    throughput_mb_s = data_mb / elapsed

    report = parse_dask_report_summary(report_path)

    return {
        "store": store,
        "latency_s": elapsed,
        "throughput_MBps": throughput_mb_s,
        "shape": tuple(accum_mm.shape),
        "mean_accum_mm": float(accum_mm.mean().values),
        "max_accum_mm": float(accum_mm.max().values),
        "n_tasks": report["n_tasks"],
        "compute_time_s": report["compute_time_s"],
        "transfer_time_ms": report["transfer_time_s"] * 1000 if pd.notna(report["transfer_time_s"]) else np.nan,
        "report_duration_s": report["report_duration_s"],
        "report_path": str(report_path),
    }


def build_plot_labels(df: pd.DataFrame) -> list[str]:
    mapping = {
        "maxdbz_pysteps.zarr": "nowcasting\n(strategy A)",
        "maxdbz_balanced.zarr": "balanced\n(strategy B)",
        "maxdbz_spatialtiles.zarr": "spatialtiles\n(strategy C)",
    }
    return [mapping[s] for s in df["store"]]


def make_figures(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fontsize = 18
    dpi = 300
    plot_df = df.copy()
    plot_df["strategy"] = build_plot_labels(plot_df)

    # Figure 1
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=dpi)
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
    plt.savefig(FIG_DIR / "Task_1_task_granularity.png", dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close()

    # Figure 2
    x = np.arange(len(plot_df))
    w = 0.32
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=dpi)

    ax1.bar(
        x - w / 2,
        plot_df["latency_s"],
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
    plt.savefig(FIG_DIR / "Task_1_dask_performance.png", dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    time_start_utc = pd.Timestamp(TIME_START_LOCAL) - pd.Timedelta(hours=8)
    time_end_utc = pd.Timestamp(TIME_END_LOCAL) - pd.Timedelta(hours=8)

    print(dask.__version__)
    print(distributed.__version__)
    print("local :", TIME_START_LOCAL, "->", time_start_utc)
    print("local :", TIME_END_LOCAL, "->", time_end_utc)

    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        dashboard_address=DASHBOARD_ADDRESS,
    )
    client = Client(cluster)

    try:
        results = []
        for store in STORES:
            out = benchmark_one_store(store, time_start_utc, time_end_utc)
            results.append(out)
            print(f"{store}: {out['latency_s']:.3f} s | {out['throughput_MBps']:.2f} MB/s")

        df_bench = pd.DataFrame(results).sort_values("latency_s").reset_index(drop=True)
        print(df_bench)
        df_bench.to_csv(CSV_PATH, index=False)
        make_figures(df_bench)
        print(f"[OK] saved summary -> {CSV_PATH}")
    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
