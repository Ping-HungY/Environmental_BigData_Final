import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_STAC_ROOT = Path("catalog/catalog.json")
LEGACY_STAC_ROOT = Path("stac_catalog/catalog.json")
FALLBACK_STAC_ROOT = Path("catalog.json")
VECTOR_ASSET_NAME = "landslides"
ASSET_ALIASES = {
    "rain-rate": "rain_rate",
    "rainfall": "accum_rainfall",
    "accum-rainfall": "accum_rainfall",
    "echotop-18": "echotop_18",
    "echotop-45": "echotop_45",
    "landslide": VECTOR_ASSET_NAME,
}


def get_stac_root() -> Path:
    stac_root = Path(os.getenv("STAC_ROOT", DEFAULT_STAC_ROOT))
    if stac_root.exists():
        return stac_root
    if LEGACY_STAC_ROOT.exists():
        return LEGACY_STAC_ROOT
    if FALLBACK_STAC_ROOT.exists():
        return FALLBACK_STAC_ROOT
    raise FileNotFoundError(
        "STAC root catalog not found. Expected './STAC_catalog/catalog.json', "
        "'./stac_catalog/catalog.json', or './catalog.json'."
    )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_href(base: Path, href: str) -> Path:
    href = href.strip()
    if href.startswith("file://"):
        href = href[7:]
    href_path = Path(href)
    if href_path.is_absolute():
        return href_path
    return (base.parent / href_path).resolve()


def collect_items(catalog_path: Path) -> List[Dict[str, Any]]:
    root_catalog = load_json(catalog_path)
    items: List[Dict[str, Any]] = []
    visited: List[Path] = []
    _walk_catalog(catalog_path, root_catalog, visited, items)
    return items


def _walk_catalog(
    catalog_path: Path,
    catalog: Dict[str, Any],
    visited: List[Path],
    items: List[Dict[str, Any]],
) -> None:
    if catalog_path in visited:
        return
    visited.append(catalog_path)

    for link in catalog.get("links", []):
        rel = link.get("rel")
        href = link.get("href")
        if not href or not rel:
            continue

        target_path = resolve_href(catalog_path, href)
        if rel == "item":
            if target_path.exists():
                item = load_json(target_path)
                item["_source_path"] = str(target_path)
                items.append(item)
        elif rel in {"child", "collection"}:
            if target_path.exists():
                target_catalog = load_json(target_path)
                _walk_catalog(target_path, target_catalog, visited, items)


def item_asset_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    assets = item.get("assets", {})
    summary: Dict[str, Any] = {}
    for key, asset in assets.items():
        normalized = normalize_asset_key(item, key, asset)
        summary[normalized] = {
            "key": key,
            "href": asset.get("href"),
            "title": asset.get("title"),
            "type": asset.get("type"),
            "roles": asset.get("roles", []),
            "shape": asset.get("shape"),
        }
    return summary


def normalize_asset_key(item: Dict[str, Any], key: str, asset: Dict[str, Any]) -> str:
    tokens = _asset_tokens(item, key, asset)

    if _contains_token(tokens, "landslide"):
        return VECTOR_ASSET_NAME
    if asset.get("type", "").startswith("application/geopackage"):
        return VECTOR_ASSET_NAME
    if asset.get("type", "").startswith("application/geo+json") and _contains_token(tokens, "landslide"):
        return VECTOR_ASSET_NAME

    if _contains_token(tokens, "maxdbz"):
        return "maxdbz"
    if _contains_token(tokens, "rain-rate") or _contains_token(tokens, "rain_rate"):
        return "rain_rate"
    if _contains_token(tokens, "accum-rainfall") or _contains_token(tokens, "accum_rainfall"):
        return "accum_rainfall"
    if _contains_token(tokens, "echotop-18") or _contains_token(tokens, "echotop_18"):
        return "echotop_18"
    if _contains_token(tokens, "echotop-45") or _contains_token(tokens, "echotop_45"):
        return "echotop_45"
    if _contains_token(tokens, "vil"):
        return "vil"

    return _canonicalize_asset_name(key.lower())


def find_item(items: List[Dict[str, Any]], item_id: str) -> Optional[Dict[str, Any]]:
    for item in items:
        if item.get("id") == item_id:
            return item
    return None


def find_landslide_asset(item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    assets = item.get("assets", {})
    for key, asset in assets.items():
        if normalize_asset_key(item, key, asset) == VECTOR_ASSET_NAME:
            return key, asset

    for key, asset in assets.items():
        if asset.get("type", "").startswith("application/geo+json"):
            return key, asset

    raise KeyError("Landslide asset not found for item")


def find_raster_asset(item: Dict[str, Any], asset_name: str) -> Tuple[str, Dict[str, Any]]:
    assets = item.get("assets", {})
    normalized = _canonicalize_asset_name(asset_name.lower())

    for key, asset in assets.items():
        if normalize_asset_key(item, key, asset) == normalized:
            return key, asset

    for key, asset in assets.items():
        if asset.get("type", "").startswith("application/vnd+zarr"):
            return key, asset

    raise KeyError(f"Raster asset '{asset_name}' not found for item")


def resolve_asset_path(item: Dict[str, Any], asset: Dict[str, Any]) -> Path:
    href = asset.get("href")
    if not href:
        raise ValueError("Asset href is missing")
    source_path = Path(item.get("_source_path", "."))
    return resolve_href(source_path, href)


def choose_data_variable(dataset: Any, asset_name: str) -> str:
    candidates = list(dataset.data_vars)
    if not candidates:
        raise ValueError("No data variables found in the dataset")

    if asset_name in dataset.data_vars:
        return asset_name

    asset_name_lower = asset_name.lower()
    for name in candidates:
        if asset_name_lower in name.lower():
            return name

    return candidates[0]


def get_time_coordinates(dataset: Any, data_var: str) -> List[str]:
    """Extract time coordinates from xarray dataset and return as ISO strings."""
    if "time" not in dataset.coords:
        raise ValueError("No time coordinate found in dataset")

    time_coord = dataset.coords["time"]
    timestamps = []
    for t in time_coord.values:
        # Convert to datetime if it's not already
        if hasattr(t, 'isoformat'):
            timestamps.append(t.isoformat())
        else:
            # Assume it's already a string or can be converted
            timestamps.append(str(t))

    return timestamps


def get_time_coordinates_from_item(item: Dict[str, Any]) -> List[str]:
    # Time-series rasters should expose their real temporal axis through
    # cube:dimensions.time. Static rasters (for example accum_rainfall) do not
    # have a time coordinate, so we fall back to one representative timestamp
    # derived from the item metadata.
    time_dimension = item.get("properties", {}).get("cube:dimensions", {}).get("time", {})
    extent = time_dimension.get("extent") or []
    if len(extent) != 2:
        static_timestamps = _get_static_item_timestamps(item)
        if static_timestamps:
            return static_timestamps
        raise ValueError("No temporal extent found in STAC item metadata")

    start = _parse_iso_datetime(extent[0])
    end = _parse_iso_datetime(extent[1])
    step = _parse_iso8601_duration(time_dimension.get("step"))
    time_count = _get_time_count_from_item(item)
    if step is None or step.total_seconds() <= 0:
        if time_count <= 1:
            return [_format_iso_datetime(start), _format_iso_datetime(end)]
        return _interpolate_timestamps(start, end, time_count)

    timestamps: List[str] = []
    current = start
    while current <= end:
        timestamps.append(_format_iso_datetime(current))
        current += step

    if time_count > 1 and len(timestamps) != time_count:
        return _interpolate_timestamps(start, end, time_count)

    if not timestamps or timestamps[-1] != _format_iso_datetime(end):
        timestamps.append(_format_iso_datetime(end))
    return timestamps


def collect_events(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    events: Dict[str, Dict[str, Any]] = {}

    for item in items:
        event_name = get_event_name(item)
        if not event_name:
            continue

        event_id = slugify_event_name(event_name)
        event = events.setdefault(
            event_id,
            {
                "id": event_id,
                "title": event_name,
                "bbox": item.get("bbox"),
                "geometry": item.get("geometry"),
                "properties": {
                    "event:name": event_name,
                    "event:year": item.get("properties", {}).get("event:year"),
                    "event:type": item.get("properties", {}).get("event:type"),
                    "event:country": item.get("properties", {}).get("event:country"),
                },
                "raster_assets": {},
                "vector_asset": None,
            },
        )

        event["bbox"] = _merge_bbox(event.get("bbox"), item.get("bbox"))
        if event.get("geometry") is None and item.get("geometry") is not None:
            event["geometry"] = item.get("geometry")

        try:
            asset_key, asset = find_landslide_asset(item)
            event["vector_asset"] = build_event_asset_entry(item, asset_key, asset, VECTOR_ASSET_NAME)
            continue
        except KeyError:
            pass

        try:
            asset_name, raster_asset = find_raster_asset(item, _infer_item_asset_name(item))
        except KeyError:
            continue

        canonical_name = normalize_asset_key(item, asset_name, raster_asset)
        event["raster_assets"][canonical_name] = build_event_asset_entry(
            item,
            asset_name,
            raster_asset,
            canonical_name,
        )

    return events


def get_event_name(item: Dict[str, Any]) -> Optional[str]:
    properties = item.get("properties", {})
    event_name = properties.get("event:name")
    if event_name:
        return str(event_name)
    title = properties.get("title")
    if title:
        return str(title).split()[0]
    return None


def slugify_event_name(event_name: str) -> str:
    return event_name.strip().lower().replace(" ", "-")


def build_event_asset_entry(
    item: Dict[str, Any],
    asset_key: str,
    asset: Dict[str, Any],
    asset_name: str,
) -> Dict[str, Any]:
    return {
        "name": asset_name,
        "asset_key": asset_key,
        "item_id": item.get("id"),
        "collection": item.get("collection"),
        "href": asset.get("href"),
        "title": asset.get("title"),
        "type": asset.get("type"),
        "roles": asset.get("roles", []),
        "item": item,
        "asset": asset,
    }


def get_event(events: Dict[str, Dict[str, Any]], item_id: str) -> Optional[Dict[str, Any]]:
    return events.get(item_id)


def get_event_vector_asset(event: Dict[str, Any], asset_name: str) -> Dict[str, Any]:
    normalized = _canonicalize_asset_name(asset_name.lower())
    vector_asset = event.get("vector_asset")
    if normalized != VECTOR_ASSET_NAME or vector_asset is None:
        raise KeyError(f"Vector asset '{asset_name}' not found")
    return vector_asset


def get_event_raster_asset(event: Dict[str, Any], asset_name: str) -> Dict[str, Any]:
    normalized = _canonicalize_asset_name(asset_name.lower())
    raster_assets = event.get("raster_assets", {})
    if normalized not in raster_assets:
        raise KeyError(f"Raster asset '{asset_name}' not found")
    return raster_assets[normalized]


def summarize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    vector_asset = event.get("vector_asset")
    return {
        "id": event.get("id"),
        "title": event.get("title"),
        "bbox": event.get("bbox"),
        "geometry": event.get("geometry"),
        "properties": event.get("properties", {}),
        "assets": {
            "rasters": {
                name: _public_asset_summary(asset_info)
                for name, asset_info in sorted(event.get("raster_assets", {}).items())
            },
            VECTOR_ASSET_NAME: _public_asset_summary(vector_asset) if vector_asset else None,
        },
    }


def _public_asset_summary(asset_info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if asset_info is None:
        return None
    return {
        "item_id": asset_info.get("item_id"),
        "title": asset_info.get("title"),
        "type": asset_info.get("type"),
        "roles": asset_info.get("roles", []),
    }


def _asset_tokens(item: Dict[str, Any], key: str, asset: Dict[str, Any]) -> List[str]:
    properties = item.get("properties", {})
    return [
        str(key).lower(),
        str(asset.get("title", "")).lower(),
        str(asset.get("href", "")).lower(),
        str(item.get("id", "")).lower(),
        str(properties.get("product_name", "")).lower(),
    ]


def _contains_token(tokens: List[str], needle: str) -> bool:
    return any(needle in token for token in tokens)


def _canonicalize_asset_name(name: str) -> str:
    canonical = name.strip().lower().replace(" ", "_")
    canonical = ASSET_ALIASES.get(canonical, canonical)
    return canonical.replace("-", "_")


def _infer_item_asset_name(item: Dict[str, Any]) -> str:
    product_name = str(item.get("properties", {}).get("product_name", "")).strip().lower()
    if product_name:
        return _canonicalize_asset_name(product_name)
    return item.get("id", "")


def _merge_bbox(
    current_bbox: Optional[List[float]],
    incoming_bbox: Optional[List[float]],
) -> Optional[List[float]]:
    if incoming_bbox is None:
        return current_bbox
    if current_bbox is None:
        return incoming_bbox
    return [
        min(current_bbox[0], incoming_bbox[0]),
        min(current_bbox[1], incoming_bbox[1]),
        max(current_bbox[2], incoming_bbox[2]),
        max(current_bbox[3], incoming_bbox[3]),
    ]


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _format_iso_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso8601_duration(value: Optional[str]) -> Optional[timedelta]:
    if not value:
        return None
    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?",
        value,
    )
    if match is None:
        return None
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _get_static_item_timestamps(item: Dict[str, Any]) -> List[str]:
    properties = item.get("properties", {})
    product_name = str(properties.get("product_name", "")).lower()

    # Static accumulated products represent a finished summary over a period, so
    # the most meaningful single timestamp is the period end, not the start.
    if "accum" in product_name or "rainfall" in product_name:
        candidates = [
            properties.get("end_datetime"),
            properties.get("datetime"),
            properties.get("start_datetime"),
        ]
    else:
        candidates = [
            properties.get("datetime"),
            properties.get("end_datetime"),
            properties.get("start_datetime"),
        ]

    timestamps = []
    for value in candidates:
        if not value:
            continue
        dt = _parse_iso_datetime(str(value))
        formatted = _format_iso_datetime(dt)
        if formatted not in timestamps:
            timestamps.append(formatted)
    return timestamps[:1]


def _get_time_count_from_item(item: Dict[str, Any]) -> int:
    properties = item.get("properties", {})
    time_info = properties.get("cube:dimensions", {}).get("time", {})
    for key in ("values", "shape"):
      value = time_info.get(key)
      if isinstance(value, int):
          return value
    variables = properties.get("cube:variables", {})
    for variable in variables.values():
        dimensions = variable.get("dimensions", [])
        shape = variable.get("shape", [])
        if dimensions and shape and dimensions[0] == "time":
            return int(shape[0])
    assets = item.get("assets", {})
    for asset in assets.values():
        shape = asset.get("shape", {})
        if isinstance(shape, dict) and "time" in shape:
            return int(shape["time"])
    return 0


def _interpolate_timestamps(start: datetime, end: datetime, time_count: int) -> List[str]:
    if time_count <= 1:
        return [_format_iso_datetime(start)]
    total_seconds = (end - start).total_seconds()
    step_seconds = total_seconds / (time_count - 1)
    timestamps = []
    for index in range(time_count):
        timestamps.append(_format_iso_datetime(start + timedelta(seconds=step_seconds * index)))
    timestamps[-1] = _format_iso_datetime(end)
    return timestamps
