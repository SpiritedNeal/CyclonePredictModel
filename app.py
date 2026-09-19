import os
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict, Any

import joblib
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import cfgrib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "cyclone_multimodal.pth")
SCALER_PATH = os.getenv("SCALER_PATH", "cyclone_scalers.pkl")

INPUT_STEPS = 8
OUTPUT_STEPS = 4
TRACK_FEATURES = 4
ENV_FEATURES = 96
THREE_D_CHANNELS = 13
GRID_SIZE = 81

PRESSURE_LEVELS = [200, 500, 850, 925]
GFS_TIMEOUT = int(os.getenv("GFS_TIMEOUT", "60"))
GFS_MAX_WORKERS = int(os.getenv("GFS_MAX_WORKERS", "8"))
CODE_VERSION = "2026-09-19-tcnd-aligned-v7-fast-gfs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Cyclone Forecast API",
    description="8-observation cyclone track input -> 24-hour forecast",
    version="1.0.0",
)


# ============================================================
# Model
# ============================================================

class GridEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(THREE_D_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        x = self.network(x)
        x = x.flatten(1)
        return self.fc(x)


class CycloneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid_encoder = GridEncoder()

        lstm_input = TRACK_FEATURES + ENV_FEATURES + 64 + 1

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_STEPS * TRACK_FEATURES),
        )

    def forward(self, track, env, three_d, mask):
        # track:    (B, T, 4)
        # env:      (B, T, 96)
        # three_d:  (B, T, 13, 81, 81)
        # mask:     (B, T)

        batch_size, time_steps = track.shape[:2]

        grid_flat = three_d.reshape(
            batch_size * time_steps,
            THREE_D_CHANNELS,
            GRID_SIZE,
            GRID_SIZE,
        )

        grid_encoded = self.grid_encoder(grid_flat)
        grid_encoded = grid_encoded.reshape(batch_size, time_steps, 64)

        mask_feature = mask.unsqueeze(-1).float()

        x = torch.cat(
            [track, env, grid_encoded, mask_feature],
            dim=-1,
        )

        lstm_out, _ = self.lstm(x)
        final_hidden = lstm_out[:, -1, :]
        output = self.decoder(final_hidden)

        return output.reshape(batch_size, OUTPUT_STEPS, TRACK_FEATURES)


# ============================================================
# Load model + scalers
# ============================================================

model = CycloneModel().to(DEVICE)
model.eval()


def load_model_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError("Unsupported model checkpoint format")

    # Remove DataParallel prefix if the training machine used nn.DataParallel.
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    model.load_state_dict(cleaned, strict=True)
    model.eval()


track_scaler = None
env_scaler = None
target_scaler = None


def load_scalers(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found: {path}")

    loaded = joblib.load(path)

    if isinstance(loaded, dict):
        t_scaler = loaded.get("track") or loaded.get("track_scaler")
        e_scaler = loaded.get("env") or loaded.get("env_scaler")
        y_scaler = loaded.get("target") or loaded.get("target_scaler")
    elif isinstance(loaded, (list, tuple)) and len(loaded) >= 3:
        t_scaler, e_scaler, y_scaler = loaded[:3]
    else:
        raise RuntimeError(
            "Unsupported cyclone_scalers.pkl format. Expected a dict containing "
            "track, env and target scalers."
        )

    if t_scaler is None or e_scaler is None or y_scaler is None:
        raise RuntimeError(
            "cyclone_scalers.pkl is missing track, env or target scaler"
        )

    if getattr(t_scaler, "n_features_in_", TRACK_FEATURES) != TRACK_FEATURES:
        raise RuntimeError("track_scaler does not contain 4 features")
    if getattr(e_scaler, "n_features_in_", ENV_FEATURES) != ENV_FEATURES:
        raise RuntimeError("env_scaler does not contain 96 features")
    if getattr(y_scaler, "n_features_in_", TRACK_FEATURES) != TRACK_FEATURES:
        raise RuntimeError("target_scaler does not contain 4 features")

    print("Target scaler mean:", getattr(y_scaler, "mean_", None))
    print("Target scaler scale:", getattr(y_scaler, "scale_", None))

    return t_scaler, e_scaler, y_scaler


try:
    load_model_checkpoint(MODEL_PATH)
    track_scaler, env_scaler, target_scaler = load_scalers(SCALER_PATH)
    LOAD_ERROR = None
except Exception as exc:
    LOAD_ERROR = str(exc)
    print("LOAD ERROR:", LOAD_ERROR)


# ============================================================
# Request schemas
# ============================================================

class Observation(BaseModel):
    timestamp: str = Field(..., description="UTC timestamp in YYYYMMDDHH format")
    longitude: float
    latitude: float
    pressure: float = Field(..., description="Central pressure in hPa")
    wind: float = Field(..., description="Maximum sustained wind in m/s")


class PredictionRequest(BaseModel):
    observations: List[Observation]


# ============================================================
# General helpers
# ============================================================

def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%d%H")
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid timestamp '{value}'. Expected YYYYMMDDHH.",
        ) from exc


def normalize_longitude(lon: float) -> float:
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon


def longitude_0_360(lon: float) -> float:
    return lon % 360.0


def validate_observations(observations: List[Observation]) -> List[Observation]:
    if len(observations) != INPUT_STEPS:
        raise HTTPException(
            status_code=422,
            detail=f"Exactly {INPUT_STEPS} observations are required.",
        )

    parsed = [parse_timestamp(o.timestamp) for o in observations]

    for i in range(1, len(parsed)):
        delta = parsed[i] - parsed[i - 1]
        if delta != timedelta(hours=6):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Observations must be chronological and exactly 6 hours apart. "
                    f"Observation {i - 1} -> {i} differs by {delta}."
                ),
            )

    for o in observations:
        if not (-180 <= o.longitude <= 360):
            raise HTTPException(status_code=422, detail="Longitude is out of range.")
        if not (-90 <= o.latitude <= 90):
            raise HTTPException(status_code=422, detail="Latitude is out of range.")
        if not (800 <= o.pressure <= 1100):
            raise HTTPException(
                status_code=422,
                detail="Pressure must be between 800 and 1100 hPa.",
            )
        if o.wind < 0:
            raise HTTPException(status_code=422, detail="Wind cannot be negative.")

    return observations


def one_hot(index: int, size: int) -> List[float]:
    result = [0.0] * size
    if 0 <= index < size:
        result[index] = 1.0
    return result


# ============================================================
# Environment feature reconstruction
# ============================================================

# NOTE:
# The trained model expects the original 96-dimensional environmental feature
# vector. This function reconstructs those 96 dimensions from the supplied
# track so the API does not require a cyclone name or a TCND folder.
#
# The production GFS data supplies the atmospheric 3-D input separately.


def classify_basin(latitude: float, longitude: float) -> int:
    """Approximate the six TCND basin classes used by the environment vector."""
    lon = normalize_longitude(longitude)

    # 0 = EP, 1 = NA, 2 = NI, 3 = SI, 4 = SP, 5 = WP
    if lon >= -140 and lon < -80 and latitude >= -10:
        return 0  # EP
    if lon >= -100 and lon <= -10 and latitude >= 0:
        return 1  # NA
    if 20 <= lon <= 100 and -5 <= latitude <= 30:
        return 2  # NI
    if 20 <= lon <= 120 and latitude < -5:
        return 3  # SI
    if 120 <= lon <= 180 and latitude < 0:
        return 4  # SP
    return 5      # WP / fallback


def intensity_class_index(wind_mps: float) -> int:
    """Exact 6-class TCND environment intensity bins."""
    if wind_mps < 17.1:
        return 0
    if wind_mps < 24.4:
        return 1
    if wind_mps < 32.6:
        return 2
    if wind_mps < 41.4:
        return 3
    if wind_mps < 50.9:
        return 4
    return 5


def direction_onehot_from_raw(raw_lons: List[float], raw_lats: List[float]) -> np.ndarray:
    """Reproduce the direction encoding used by the TCND environment script."""
    raw_lons = list(raw_lons)
    raw_lats = list(raw_lats)

    if len(raw_lons) < 2 or len(raw_lats) < 2:
        return np.zeros(8, dtype=np.float64)

    long_rel = raw_lons[-1] - raw_lons[0]
    lat_rel = raw_lats[-1] - raw_lats[0]

    # Original TCND implementation uses 0.1-degree coordinate units.
    long_distance = (long_rel / 10.0) * 111.0 * math.cos(
        (raw_lats[0] + raw_lats[1]) / 2.0 / 10.0 * math.pi / 180.0
    )
    lat_distance = (lat_rel / 10.0) * 111.0

    velocity = math.sqrt(long_distance ** 2 + lat_distance ** 2)
    if velocity == 0:
        result = np.zeros(8, dtype=np.float64)
        result[0] = 1.0
        return result

    sin_angle = lat_distance / velocity
    cos_angle = long_distance / velocity
    sin_angle = max(-1.0, min(1.0, sin_angle))

    if sin_angle >= 0 and cos_angle >= 0:
        angle = math.asin(sin_angle)
    elif sin_angle >= 0 and cos_angle <= 0:
        angle = math.pi - math.asin(sin_angle)
    elif sin_angle <= 0 and cos_angle <= 0:
        angle = math.pi - math.asin(sin_angle)
    else:
        angle = 2.0 * math.pi + math.asin(sin_angle)

    angle_lists = [
        (math.pi * (1 / 8), 2 * math.pi - math.pi * (1 / 8)),
        (2 * math.pi - math.pi * (1 / 8), 2 * math.pi - math.pi * (3 / 8)),
        (2 * math.pi - math.pi * (3 / 8), 2 * math.pi - math.pi * (5 / 8)),
        (2 * math.pi - math.pi * (5 / 8), 2 * math.pi - math.pi * (7 / 8)),
        (2 * math.pi - math.pi * (7 / 8), 2 * math.pi - math.pi * (9 / 8)),
        (2 * math.pi - math.pi * (9 / 8), 2 * math.pi - math.pi * (11 / 8)),
        (2 * math.pi - math.pi * (11 / 8), 2 * math.pi - math.pi * (13 / 8)),
        (2 * math.pi - math.pi * (13 / 8), 2 * math.pi - math.pi * (15 / 8)),
    ]

    angle_class = 0
    for class_id, (low, high) in enumerate(angle_lists):
        if class_id == 0:
            if angle > high or angle <= low:
                angle_class = class_id
                break
        else:
            if angle > high and angle < low:
                angle_class = class_id
                break

    result = np.zeros(8, dtype=np.float64)
    result[angle_class] = 1.0
    return result


def intensity_change_onehot(winds_mps: List[float]) -> np.ndarray:
    """Reproduce TCND's 24-hour intensity-change classification."""
    winds = list(winds_mps)
    if len(winds) < 2:
        return np.zeros(4, dtype=np.float64)

    grad = [winds[i + 1] - winds[i] for i in range(len(winds) - 1)]
    g = np.asarray(grad, dtype=np.float64)

    if np.all(g == 0):
        cls = 3
    elif np.all(g >= 0):
        cls = 0
    elif np.all(g <= 0):
        cls = 2
    else:
        nonzero = np.flatnonzero(g)
        if len(nonzero) and g[nonzero[0]] > 0 and g[nonzero[-1]] < 0:
            cls = 1
        elif np.sum(g) > 0:
            cls = 0
        elif np.sum(g) < 0:
            cls = 2
        else:
            cls = 3

    result = np.zeros(4, dtype=np.float64)
    result[cls] = 1.0
    return result


def physical_to_tcnd_observation(longitude: float, latitude: float, pressure: float, wind: float) -> List[float]:
    """Convert physical units to the exact Data1D representation used by TCND."""
    lon360 = longitude_0_360(longitude)
    lon_raw_01deg = lon360 * 10.0
    lat_raw_01deg = latitude * 10.0

    return [
        (lon_raw_01deg - 1800.0) / 50.0,
        lat_raw_01deg / 50.0,
        (pressure - 960.0) / 50.0,
        (wind - 40.0) / 25.0,
    ]


def tcnd_to_physical_prediction(values: np.ndarray) -> np.ndarray:
    """Convert [LONG, LAT, PRES, WND] from TCND normalized units to physical units."""
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)

    lon_raw = values[..., 0] * 50.0 + 1800.0
    lat_raw = values[..., 1] * 50.0

    out[..., 0] = lon_raw / 10.0
    out[..., 1] = lat_raw / 10.0
    out[..., 2] = values[..., 2] * 50.0 + 960.0
    out[..., 3] = values[..., 3] * 25.0 + 40.0

    return out.astype(np.float32)

def build_environment_raw(observations: List[Observation], index: int) -> np.ndarray:
    """Build the exact 96-D TCND Env-Data vector for one history timestep."""
    current = observations[index]

    # TCND environment processing uses longitude in 0.1-degree E units.
    raw_lons = [longitude_0_360(o.longitude) * 10.0 for o in observations]
    raw_lats = [o.latitude * 10.0 for o in observations]
    winds = [float(o.wind) for o in observations]

    raw_lon = raw_lons[index]
    raw_lat = raw_lats[index]

    raw: List[float] = []

    # 6 - area one-hot. Basin boundaries are necessarily inferred because the
    # API intentionally does not require a basin/cyclone identifier.
    basin_idx = classify_basin(current.latitude, current.longitude)
    raw.extend(one_hot(basin_idx, 6))

    # 1 - environment wind is WND / 110 in the original TCND env-data code.
    raw.append(float(current.wind) / 110.0)

    # 6 - intensity class uses the original m/s thresholds.
    raw.extend(one_hot(intensity_class_index(float(current.wind)), 6))

    # 1 - movement velocity. First timestep is zero.
    if index == 0:
        move_velocity = 0.0
    else:
        long_rel = raw_lons[index] - raw_lons[index - 1]
        lat_rel = raw_lats[index] - raw_lats[index - 1]
        long_distance = (long_rel / 10.0) * 111.0 * math.cos(
            (raw_lats[index - 1] + raw_lats[index]) / 2.0 / 10.0 * math.pi / 180.0
        )
        lat_distance = (lat_rel / 10.0) * 111.0
        velocity_km = math.sqrt(long_distance ** 2 + lat_distance ** 2)
        move_velocity = velocity_km / 1219.8387650082498
    raw.append(float(move_velocity))

    # 12 - month one-hot
    raw.extend(one_hot(parse_timestamp(current.timestamp).month - 1, 12))

    # 2 - original environment location coordinates in 0.1-degree units.
    raw.extend([float(raw_lon), float(raw_lat)])

    # 36 + 12 - global location one-hot encoding from TCND.
    lon_bin = int((raw_lon // 10.0) // 10.0)
    lat_bin = int((raw_lat + 600.0) // 100.0)
    lon_bin = max(0, min(35, lon_bin))
    lat_bin = max(0, min(11, lat_bin))
    raw.extend(one_hot(lon_bin, 36))
    raw.extend(one_hot(lat_bin, 12))

    # 8 - 12h movement direction. TCND uses the previous 2 intervals.
    if index < 2:
        raw.extend(np.zeros(8, dtype=np.float64).tolist())
    else:
        raw.extend(direction_onehot_from_raw(raw_lons[index - 2:index + 1], raw_lats[index - 2:index + 1]).tolist())

    # 8 - 24h movement direction. TCND uses the previous 4 intervals.
    if index < 4:
        raw.extend(np.zeros(8, dtype=np.float64).tolist())
    else:
        raw.extend(direction_onehot_from_raw(raw_lons[index - 4:index + 1], raw_lats[index - 4:index + 1]).tolist())

    # 4 - 24h intensity change.
    if index < 4:
        raw.extend(np.zeros(4, dtype=np.float64).tolist())
    else:
        raw.extend(intensity_change_onehot(winds[index - 4:index + 1]).tolist())

    arr = np.asarray(raw, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.shape != (ENV_FEATURES,):
        raise RuntimeError(f"Environment vector has shape {arr.shape}, expected (96,)")

    return arr


def build_environment_sequence(observations: List[Observation]) -> np.ndarray:
    """Build one exact 96-D Env-Data vector for every historical timestep."""
    vectors = [build_environment_raw(observations, i) for i in range(len(observations))]
    return np.stack(vectors, axis=0).astype(np.float64)


# ============================================================
# GFS 0.25-degree GRIB Filter
# ============================================================


def gfs_cycle_candidates(valid_time: datetime):
    """Return (cycle_datetime, forecast_hour) candidates newest-first."""
    base = valid_time.replace(minute=0, second=0, microsecond=0)
    cycle_hour = (base.hour // 6) * 6
    cycle_dt = base.replace(hour=cycle_hour)

    candidates = []
    # The primary choice is the nominal cycle. If f000 is temporarily
    # unavailable, fall back to the previous cycle with a +6h forecast.
    for i in range(4):
        cdt = cycle_dt - timedelta(hours=6 * i)
        fhour = int((base - cdt).total_seconds() // 3600)
        candidates.append((cdt, fhour))
    return candidates


def gfs_url(cycle_dt: datetime, forecast_hour: int, latitude: float, longitude: float) -> str:
    date = cycle_dt.strftime("%Y%m%d")
    cycle = cycle_dt.strftime("%H")

    lon = normalize_longitude(longitude)
    lon0 = max(-180.0, lon - 10.0)
    lon1 = min(180.0, lon + 10.0)
    lat0 = max(-90.0, latitude - 10.0)
    lat1 = min(90.0, latitude + 10.0)

    params = [
        ("file", f"gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}"),
        ("lev_200_mb", "on"),
        ("lev_500_mb", "on"),
        ("lev_850_mb", "on"),
        ("lev_925_mb", "on"),
        ("lev_2_m_above_ground", "on"),
        ("var_HGT", "on"),
        ("var_TMP", "on"),
        ("var_UGRD", "on"),
        ("var_VGRD", "on"),
        ("subregion", ""),
        ("leftlon", f"{lon0:.2f}"),
        ("rightlon", f"{lon1:.2f}"),
        ("toplat", f"{lat1:.2f}"),
        ("bottomlat", f"{lat0:.2f}"),
        ("dir", f"/gfs.{date}/{cycle}/atmos"),
    ]

    from urllib.parse import urlencode
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?" + urlencode(params)


def download_gfs(timestamp: datetime, latitude: float, longitude: float):
    errors = []

    for cycle_dt, forecast_hour in gfs_cycle_candidates(timestamp):
        url = gfs_url(cycle_dt, forecast_hour, latitude, longitude)
        try:
            response = requests.get(url, timeout=GFS_TIMEOUT)

            if response.status_code != 200:
                errors.append(
                    f"{cycle_dt.strftime('%Y%m%d%H')} f{forecast_hour:03d}: HTTP {response.status_code}"
                )
                continue

            content = response.content
            content_type = response.headers.get("content-type", "").lower()
            if not content or b"<html" in content[:300].lower() or "text/html" in content_type:
                errors.append(
                    f"{cycle_dt.strftime('%Y%m%d%H')} f{forecast_hour:03d}: non-GRIB response"
                )
                continue

            return content, cycle_dt, forecast_hour

        except Exception as exc:
            errors.append(
                f"{cycle_dt.strftime('%Y%m%d%H')} f{forecast_hour:03d}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError("NOAA GFS lookup failed; " + " | ".join(errors[:4]))


def open_grib_group(grib_path: str, type_of_level: str):
    """Open one GRIB group containing all variables needed at a level type.

    Opening the same GRIB file once per variable/level is extremely expensive
    because cfgrib repeatedly parses the index. We therefore open the complete
    pressure-level group once and the 2-m group once per frame.
    """
    backend_kwargs = {
        "indexpath": "",
        "filter_by_keys": {
            "typeOfLevel": type_of_level,
        },
    }

    try:
        return cfgrib.open_dataset(grib_path, backend_kwargs=backend_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open GRIB group {type_of_level}: {exc}"
        ) from exc


def get_grib_variable(ds, preferred_names):
    """Return the first available variable from a list of GRIB/xarray names."""
    for name in preferred_names:
        if name in ds.data_vars:
            return ds[name]

    raise KeyError(
        f"None of {preferred_names} found in GRIB dataset; "
        f"available variables: {list(ds.data_vars)}"
    )


def to_2d_numpy(da) -> np.ndarray:
    """Convert a GRIB/xarray field to a 2D float32 array."""
    arr = np.asarray(da.values, dtype=np.float32)

    # Remove singleton/time dimensions until only latitude/longitude remain.
    while arr.ndim > 2:
        arr = arr[0]

    if arr.ndim != 2:
        raise RuntimeError(f"Expected a 2D atmospheric field, got shape {arr.shape}")

    return np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def standardize_channel(channel: np.ndarray) -> np.ndarray:
    """Standardize one atmospheric channel exactly as the inference pipeline expects."""
    channel = np.asarray(channel, dtype=np.float32)
    channel = np.nan_to_num(
        channel,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mean = float(np.mean(channel, dtype=np.float64))
    std = float(np.std(channel, dtype=np.float64))

    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0

    standardized = (channel - mean) / std
    return np.clip(standardized, -10.0, 10.0).astype(np.float32)


def resize_81x81(arr: np.ndarray) -> np.ndarray:
    """Resize a 2D atmospheric field to the model's 81x81 grid."""
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise RuntimeError(f"Expected a 2D field for resizing, got shape {arr.shape}")

    if arr.shape == (GRID_SIZE, GRID_SIZE):
        return arr

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=(GRID_SIZE, GRID_SIZE),
        mode="bilinear",
        align_corners=True,
    )
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


def extract_gfs_tensor(grib_bytes: bytes) -> np.ndarray:
    """Convert one filtered GFS GRIB2 payload into the 13-channel model input.

    Performance-critical implementation: cfgrib is opened only twice per frame
    rather than once for each of the 13 channels.
    """
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib_bytes)
        grib_path = tmp.name

    try:
        channels = []

        # Open the entire isobaric group once. It contains U, V and GH for all
        # four requested pressure levels.
        pressure_ds = open_grib_group(grib_path, "isobaricInhPa")

        for level in PRESSURE_LEVELS:
            da = pressure_ds["u"].sel(isobaricInhPa=level).load()
            channels.append(resize_81x81(standardize_channel(to_2d_numpy(da))))

        for level in PRESSURE_LEVELS:
            da = pressure_ds["v"].sel(isobaricInhPa=level).load()
            channels.append(resize_81x81(standardize_channel(to_2d_numpy(da))))

        # cfgrib may expose geopotential height as "gh". Keep a small fallback
        # for datasets that expose an alternate name.
        gh_name = "gh" if "gh" in pressure_ds.data_vars else "z" if "z" in pressure_ds.data_vars else None
        if gh_name is None:
            raise KeyError(
                f"GFS pressure dataset contains no geopotential-height variable; "
                f"found {list(pressure_ds.data_vars)}"
            )

        for level in PRESSURE_LEVELS:
            da = pressure_ds[gh_name].sel(isobaricInhPa=level).load()
            channels.append(resize_81x81(standardize_channel(to_2d_numpy(da))))

        # The model was trained with SST as channel 13. Production GFS uses
        # 2-m temperature as the documented inference-time proxy.
        surface_ds = open_grib_group(grib_path, "heightAboveGround")
        temp_name = "t2m" if "t2m" in surface_ds.data_vars else "2t" if "2t" in surface_ds.data_vars else None
        if temp_name is None:
            raise KeyError(
                f"GFS surface dataset contains no 2-m temperature variable; "
                f"found {list(surface_ds.data_vars)}"
            )

        da = surface_ds[temp_name].sel(heightAboveGround=2).load()
        channels.append(resize_81x81(standardize_channel(to_2d_numpy(da))))

        tensor = np.stack(channels, axis=0).astype(np.float32)

        expected = (THREE_D_CHANNELS, GRID_SIZE, GRID_SIZE)
        if tensor.shape != expected:
            raise RuntimeError(
                f"GFS tensor shape is {tensor.shape}; expected {expected}"
            )

        return tensor

    finally:
        try:
            os.remove(grib_path)
        except OSError:
            pass


def fetch_one_gfs_frame(timestamp: datetime, latitude: float, longitude: float):
    data, cycle_dt, forecast_hour = download_gfs(timestamp, latitude, longitude)
    tensor = extract_gfs_tensor(data)
    return tensor, cycle_dt, forecast_hour


# ============================================================
# Inference
# ============================================================


def build_track_array(observations: List[Observation]) -> np.ndarray:
    """Convert user-facing physical observations to TCND Data1D normalized values."""
    track = np.array(
        [
            physical_to_tcnd_observation(
                float(o.longitude),
                float(o.latitude),
                float(o.pressure),
                float(o.wind),
            )
            for o in observations
        ],
        dtype=np.float64,
    )

    if track.shape != (INPUT_STEPS, TRACK_FEATURES):
        raise RuntimeError(f"Track array shape is {track.shape}; expected ({INPUT_STEPS}, {TRACK_FEATURES})")

    return track


def inverse_transform_predictions(pred_scaled: np.ndarray) -> np.ndarray:
    """Undo StandardScaler and then undo TCND Data1D normalization."""
    if target_scaler is None:
        raise RuntimeError("target_scaler is not loaded")

    original_shape = pred_scaled.shape
    flat = pred_scaled.reshape(-1, TRACK_FEATURES)

    # Step 1: StandardScaler inverse transform -> TCND normalized values.
    tcnd_values = target_scaler.inverse_transform(flat)

    # Step 2: TCND normalized values -> physical lon/lat/pressure/wind.
    physical = tcnd_to_physical_prediction(tcnd_values)
    return physical.reshape(original_shape).astype(np.float32)


def sanitize_physical_prediction(row: np.ndarray) -> Dict[str, float]:
    """Format model output that is already in physical units."""
    longitude = float(normalize_longitude(float(row[0])))
    latitude = float(row[1])
    pressure_hpa = float(row[2])
    wind_mps = float(max(0.0, float(row[3])))
    wind_knots = wind_mps * 1.94384449

    return {
        "latitude": latitude,
        "longitude": longitude,
        "pressure_hpa": pressure_hpa,
        "wind_mps": wind_mps,
        "wind_knots": wind_knots,
    }


def forecast_timestamp(last_timestamp: str, hours_ahead: int) -> str:
    dt = parse_timestamp(last_timestamp)
    return (dt + timedelta(hours=hours_ahead)).strftime("%Y%m%d%H")


# ============================================================
# API routes
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Cyclone Forecast API",
        "status": "ok" if LOAD_ERROR is None else "model_load_error",
        "device": str(DEVICE),
        "endpoint": "POST /predict",
        "required_observations": INPUT_STEPS,
        "code_version": CODE_VERSION,
        "tcnd_data1d_normalized": True,
    }


@app.get("/health")
def health():
    result = {
        "status": "healthy" if LOAD_ERROR is None else "unhealthy",
        "model_loaded": LOAD_ERROR is None,
        "device": str(DEVICE),
        "load_error": LOAD_ERROR,
        "code_version": CODE_VERSION,
    }

    if LOAD_ERROR is None and target_scaler is not None:
        result["target_scaler_mean"] = [float(x) for x in target_scaler.mean_]
        result["target_scaler_scale"] = [float(x) for x in target_scaler.scale_]

    result["tcnd_normalization"] = {
        "longitude": "(longitude_0_to_360_deg * 10 - 1800) / 50",
        "latitude": "(latitude_deg * 10) / 50",
        "pressure_hpa": "(pressure_hpa - 960) / 50",
        "wind_mps": "(wind_mps - 40) / 25",
        "target_output": "StandardScaler inverse_transform, then TCND denormalization",
    }

    return result


@app.post("/predict")
def predict(request: PredictionRequest):
    if LOAD_ERROR is not None:
        raise HTTPException(
            status_code=500,
            detail=f"Model/scaler loading failed: {LOAD_ERROR}",
        )

    observations = validate_observations(request.observations)
    request_started = time.perf_counter()

    try:
        # --------------------------------------------------------
        # 1. Track inputs
        # --------------------------------------------------------
        track_raw = build_track_array(observations)
        track_scaled = track_scaler.transform(track_raw).astype(np.float32)

        # --------------------------------------------------------
        # 2. Environment inputs
        # --------------------------------------------------------
        env_raw = build_environment_sequence(observations)
        env_scaled = env_scaler.transform(env_raw).astype(np.float32)

        # --------------------------------------------------------
        # 3. Atmospheric inputs from NOAA GFS
        # --------------------------------------------------------
        # The old implementation fetched/decoded all 8 GFS frames serially.
        # Each frame is independent, so this could easily take 2-3 minutes.
        # Run a small bounded worker pool instead. Eight workers allow all eight independent NOAA frames to be fetched/decoded concurrently.
        # This removes the previous two-wave bottleneck when eight observations are supplied.
        gfs_started = time.perf_counter()

        zero_frame = np.zeros(
            (THREE_D_CHANNELS, GRID_SIZE, GRID_SIZE),
            dtype=np.float32,
        )

        three_d_frames = [zero_frame.copy() for _ in observations]
        mask = [0.0] * len(observations)
        gfs_failures = []
        gfs_frames_available = 0
        gfs_frames_missing = 0

        def fetch_indexed(index, obs):
            ts = parse_timestamp(obs.timestamp)
            frame, cycle_dt, forecast_hour = fetch_one_gfs_frame(
                ts,
                float(obs.latitude),
                float(obs.longitude),
            )
            return index, frame, cycle_dt, forecast_hour

        worker_count = max(1, min(GFS_MAX_WORKERS, len(observations)))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(fetch_indexed, i, obs): i
                for i, obs in enumerate(observations)
            }

            for future in as_completed(futures):
                index = futures[future]
                obs = observations[index]

                try:
                    _, frame, cycle_dt, forecast_hour = future.result()
                    three_d_frames[index] = frame
                    mask[index] = 1.0
                    gfs_frames_available += 1
                    print(
                        f"GFS frame {index + 1}/{len(observations)} loaded for "
                        f"{obs.timestamp} using {cycle_dt.strftime('%Y%m%d%H')} f{forecast_hour:03d}"
                    )
                except Exception as exc:
                    print(
                        f"GFS failure for {obs.timestamp}: {type(exc).__name__}: {exc}"
                    )
                    mask[index] = 0.0
                    gfs_frames_missing += 1
                    gfs_failures.append(
                        {
                            "timestamp": obs.timestamp,
                            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        }
                    )

        gfs_elapsed_seconds = round(time.perf_counter() - gfs_started, 2)

        if gfs_frames_available == 0:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "No NOAA GFS atmospheric frames could be loaded; inference was aborted.",
                    "gfs_frames_available": 0,
                    "gfs_frames_missing": gfs_frames_missing,
                    "gfs_failures": gfs_failures,
                },
            )

        three_d_raw = np.stack(three_d_frames, axis=0).astype(np.float32)
        mask_array = np.asarray(mask, dtype=np.float32)

        # --------------------------------------------------------
        # 4. Torch tensors
        # --------------------------------------------------------
        track_tensor = torch.from_numpy(track_scaled).unsqueeze(0).to(DEVICE)
        env_tensor = torch.from_numpy(env_scaled).unsqueeze(0).to(DEVICE)
        three_d_tensor = torch.from_numpy(three_d_raw).unsqueeze(0).to(DEVICE)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).to(DEVICE)

        # --------------------------------------------------------
        # 5. Neural-network prediction
        # --------------------------------------------------------
        with torch.no_grad():
            pred_scaled = model(
                track_tensor,
                env_tensor,
                three_d_tensor,
                mask_tensor,
            )

        # Shape: (1, 4, 4) -> (4, 4)
        pred_scaled_np = pred_scaled.detach().cpu().numpy()[0]

        # --------------------------------------------------------
        # 6. IMPORTANT: inverse-transform target outputs
        # --------------------------------------------------------
        if pred_scaled_np.shape != (OUTPUT_STEPS, TRACK_FEATURES):
            raise RuntimeError(
                f"Unexpected model output shape {pred_scaled_np.shape}; "
                f"expected ({OUTPUT_STEPS}, {TRACK_FEATURES})"
            )

        pred_physical = inverse_transform_predictions(pred_scaled_np)

        forecast_hours = [6, 12, 18, 24]
        predictions = []

        for i, hours_ahead in enumerate(forecast_hours):
            physical = sanitize_physical_prediction(pred_physical[i])

            predictions.append(
                {
                    "timestamp": forecast_timestamp(
                        observations[-1].timestamp,
                        hours_ahead,
                    ),
                    **physical,
                }
            )

        total_elapsed_seconds = round(time.perf_counter() - request_started, 2)

        return {
            "last_observed_timestamp": observations[-1].timestamp,
            "forecast_hours": forecast_hours,
            "atmospheric_source": "NOAA GFS 0.25 degree (2m temperature proxy for training SST channel)",
            "performance": {
                "gfs_elapsed_seconds": gfs_elapsed_seconds,
                "total_elapsed_seconds": total_elapsed_seconds,
                "gfs_worker_count": worker_count,
            },
            "output_units": {
                "longitude": "degrees",
                "latitude": "degrees",
                "pressure_hpa": "hPa",
                "wind_mps": "m/s",
                "wind_knots": "kt",
            },
            "gfs_frames_available": gfs_frames_available,
            "gfs_frames_missing": gfs_frames_missing,
            "gfs_failures": gfs_failures,
            "predictions": predictions,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Prediction error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {str(exc)}",
        ) from exc
