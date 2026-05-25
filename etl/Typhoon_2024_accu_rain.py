from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# =========================================================
# User settings
# =========================================================
input_zarr = Path(r"G:\EnviroBigData_Data\maxdbz_balanced.zarr")
out_dir = Path(r"G:\EnviroBigData_Data\Typhoon_MAXDBZ_Accum")
out_dir.mkdir(parents=True, exist_ok=True)

# dataset time is assumed to be UTC
# warning periods below are in UTC+8
events = [
    {
        "id": "202425",
        "name_en": "USAGI",
        "name_zh": "天兔",
        "start_local": "2024-11-14 05:30",
        "end_local":   "2024-11-16 11:30",
    },
    {
        "id": "202421",
        "name_en": "KONG-REY",
        "name_zh": "康芮",
        "start_local": "2024-10-29 17:30",
        "end_local":   "2024-11-01 14:30",
    },
    {
        "id": "202418",
        "name_en": "KRATHON",
        "name_zh": "山陀兒",
        "start_local": "2024-09-29 08:30",
        "end_local":   "2024-10-04 05:30",
    },
    {
        "id": "202403",
        "name_en": "GAEMI",
        "name_zh": "凱米",
        "start_local": "2024-07-22 23:30",
        "end_local":   "2024-07-26 08:30",
    },
]

overwrite = True

# fixed Z-R relationship
ZR_A = 32.5
ZR_B = 1.65


# =========================================================
# Helpers
# =========================================================
def open_zarr_auto(path: Path) -> xr.Dataset:
    """
    Try consolidated=True first, then fallback to consolidated=False.
    """
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def dbz_to_rainrate(dbz: xr.DataArray, a: float = 32.5, b: float = 1.65) -> xr.DataArray:
    """
    Convert reflectivity in dBZ to rain rate in mm/hr.

    Z = 10^(dBZ/10)
    R = (Z / a)^(1/b)
    """
    z_lin = 10.0 ** (dbz / 10.0)
    rain = (z_lin / a) ** (1.0 / b)
    rain = rain.astype(np.float32)
    rain.attrs["units"] = "mm hr-1"
    rain.attrs["long_name"] = f"Rain rate converted from MAXDBZ using Z={a}R^{b}"
    return rain


def get_dt_hours(time_coord: xr.DataArray) -> float:
    """
    Infer time step in hours from time coordinate.
    Assumes constant time interval.
    """
    if time_coord.size < 2:
        raise ValueError("Need at least 2 time steps to infer dt_hours.")

    dt = pd.to_datetime(time_coord.values[1]) - pd.to_datetime(time_coord.values[0])
    return dt.total_seconds() / 3600.0


def local_to_utc(ts_str: str) -> pd.Timestamp:
    """
    Convert UTC+8 naive timestamp string to UTC naive timestamp.
    """
    return pd.Timestamp(ts_str) - pd.Timedelta(hours=8)


# =========================================================
# Main
# =========================================================
ds = open_zarr_auto(input_zarr)

if "MAXDBZ" not in ds.data_vars:
    raise KeyError("Variable 'MAXDBZ' not found in input zarr.")

da = ds["MAXDBZ"]  # expected dims: (time, y, x)

if "time" not in da.dims:
    raise ValueError("Input MAXDBZ must contain 'time' dimension.")

dt_hours = get_dt_hours(da["time"])
print(f"[INFO] inferred dt_hours = {dt_hours:.6f}")

for ev in events:
    start_utc = local_to_utc(ev["start_local"])
    end_utc = local_to_utc(ev["end_local"])

    # time subset over full spatial range
    da_event = da.sel(time=slice(start_utc, end_utc))

    if da_event.sizes.get("time", 0) == 0:
        print(f"[WARN] No data found for {ev['name_en']} ({ev['id']})")
        continue

    rain_rate = dbz_to_rainrate(da_event, a=ZR_A, b=ZR_B)

    # accumulation in mm
    accum = (rain_rate * dt_hours).sum(dim="time").astype(np.float32)
    accum.name = "accum_rainfall"
    accum.attrs["units"] = "mm"
    accum.attrs["long_name"] = "Event-total accumulated rainfall"
    accum.attrs["event_id"] = ev["id"]
    accum.attrs["event_name_en"] = ev["name_en"]
    accum.attrs["event_name_zh"] = ev["name_zh"]
    accum.attrs["warning_start_local_utc8"] = ev["start_local"]
    accum.attrs["warning_end_local_utc8"] = ev["end_local"]
    accum.attrs["warning_start_utc"] = str(start_utc)
    accum.attrs["warning_end_utc"] = str(end_utc)

    out_ds = xr.Dataset(
        data_vars={
            "MAXDBZ": da_event.astype(np.float32),
            "rain_rate": rain_rate,
            "accum_rainfall": accum,
        },
        coords={
            "time": da_event["time"],
            "y": da_event["y"],
            "x": da_event["x"],
        },
        attrs={
            "source_zarr": str(input_zarr),
            "event_id": ev["id"],
            "event_name_en": ev["name_en"],
            "event_name_zh": ev["name_zh"],
            "warning_start_local_utc8": ev["start_local"],
            "warning_end_local_utc8": ev["end_local"],
            "warning_start_utc": str(start_utc),
            "warning_end_utc": str(end_utc),
            "time_step_hours": dt_hours,
            "zr_relationship": f"Z={ZR_A}R^{ZR_B}",
            "description": "MAXDBZ subset, rain rate, and event-total accumulated rainfall during typhoon warning period.",
        },
    )

    out_ds = out_ds.chunk({
    "time": 48,
    "y": 256,
    "x": 256,
    })

    out_path = out_dir / f"maxdbz_{ev['id']}_{ev['name_en'].lower()}_warning_accum.zarr"

    if overwrite and out_path.exists():
        import shutil
        shutil.rmtree(out_path)

    # keep chunking simple and stable
    encoding = {
        "MAXDBZ": {"dtype": "float32"},
        "rain_rate": {"dtype": "float32"},
        "accum_rainfall": {"dtype": "float32"},
    }

    out_ds.to_zarr(
        out_path,
        mode="w",
        consolidated=True,
        encoding=encoding,
    )

    print(
        f"[OK] {ev['name_en']} ({ev['id']}) "
        f"| time steps = {da_event.sizes['time']} "
        f"| saved -> {out_path}"
    )