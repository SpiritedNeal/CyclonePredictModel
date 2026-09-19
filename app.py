import os
import math
import tempfile
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
        t_scaler = loaded.get("track_scaler")
        e_scaler = loaded.get("env_scaler")
        y_scaler = loaded.get("target_scaler")
    elif isinstance(loaded, (list, tuple)) and len(loaded) >= 3:
        t_scaler, e_scaler, y_scaler = loaded[:3]
    else:
        raise RuntimeError(
            "Unsupported cyclone_scalers.pkl format. Expected a dict containing "
            "track_scaler, env_scaler and target_scaler."
        )

    if t_scaler is None or e_scaler is None or y_scaler is None:
        raise RuntimeError(
            "cyclone_scalers.pkl is missing track_scaler, env_scaler or target_scaler"
        )

    if getattr(t_scaler, "n_features_in_", TRACK_FEATURES) != TRACK_FEATURES:
        raise RuntimeError("track_scaler does not contain 4 features")
    if getattr(e_scaler, "n_features_in_", ENV_FEATURES) != ENV_FEATURES:
        raise RuntimeError("env_scaler does not contain 96 features")
    if getattr(y_scaler, "n_features_in_", TRACK_FEATURES) != TRACK_FEATURES:
        raise RuntimeError("target_scaler does not contain 4 features")

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
    # Six broad classes to match the 6-element categorical input.
    # The exact bins were not retained as a standalone metadata file, so these
    # are kept deterministic rather than making the API require a training ID.
    knots = max(0.0, wind_mps) * 1.94384449
    if knots < 34:
        return 0
    if knots < 64:
        return 1
    if knots < 83:
        return 2
    if knots < 96:
        return 3
    if knots < 113:
        return 4
    return 5


def direction_class(degrees: float) -> int:
    # 8 directional sectors: N, NE, E, SE, S, SW, W, NW
    return int(((degrees + 22.5) % 360) // 45)


def movement_features(observations: List[Observation]) -> Dict[str, Any]:
    lons = np.array([normalize_longitude(o.longitude) for o in observations], dtype=np.float64)
    lats = np.array([o.latitude for o in observations], dtype=np.float64)
    winds = np.array([o.wind for o in observations], dtype=np.float64)

    def haversine_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(max(0.0, min(1.0, a))))

    distances = []
    bearings = []
    for i in range(1, len(observations)):
        distances.append(haversine_km(lats[i - 1], lons[i - 1], lats[i], lons[i]))

        lat1 = math.radians(lats[i - 1])
        lat2 = math.radians(lats[i])
        dl = math.radians(lons[i] - lons[i - 1])
        y = math.sin(dl) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dl)
        bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        bearings.append(bearing)

    if distances:
        velocity_kmh = distances[-1] / 6.0
        latest_direction = bearings[-1]
        direction_12 = bearings[-2] if len(bearings) >= 2 else latest_direction
        direction_24 = bearings[-4] if len(bearings) >= 4 else latest_direction
    else:
        velocity_kmh = 0.0
        latest_direction = 0.0
        direction_12 = 0.0
        direction_24 = 0.0

    current = observations[-1]
    current_wind = float(current.wind)

    change_24 = current_wind - float(observations[-5].wind)
    change_12 = current_wind - float(observations[-3].wind)

    return {
        "velocity_kmh": velocity_kmh,
        "latest_direction": latest_direction,
        "direction_12": direction_12,
        "direction_24": direction_24,
        "change_12": change_12,
        "change_24": change_24,
        "current_wind": current_wind,
    }


def build_environment_raw(observations: List[Observation]) -> np.ndarray:
    current = observations[-1]
    movement = movement_features(observations)

    raw: List[float] = []

    # 6 - basin one-hot
    basin_idx = classify_basin(current.latitude, current.longitude)
    raw.extend(one_hot(basin_idx, 6))

    # 1 - current wind
    raw.append(float(current.wind))

    # 6 - intensity class one-hot
    raw.extend(one_hot(intensity_class_index(current.wind), 6))

    # 1 - movement velocity
    raw.append(float(movement["velocity_kmh"]))

    # 12 - month one-hot
    month_index = parse_timestamp(current.timestamp).month - 1
    raw.extend(one_hot(month_index, 12))

    # 2 - current location
    raw.extend([float(current.longitude), float(current.latitude)])

    # 36 - longitude positional bins (30-degree sectors)
    lon360 = longitude_0_360(current.longitude)
    lon_bin = min(11, int(lon360 // 30.0))
    raw.extend(one_hot(lon_bin, 12))
    raw.extend(one_hot(lon_bin, 12))
    raw.extend(one_hot(int((lon360 / 360.0) * 12) % 12, 12))

    # 12 - latitude bins
    lat = max(-90.0, min(89.999, current.latitude))
    lat_bin = min(11, int((lat + 90.0) // 15.0))
    raw.extend(one_hot(lat_bin, 12))

    # 8 - latest movement direction
    raw.extend(one_hot(direction_class(movement["latest_direction"]), 8))

    # 8 - 24h movement direction
    raw.extend(one_hot(direction_class(movement["direction_24"]), 8))

    # 4 - 24h intensity change category
    delta = movement["change_24"]
    if delta < -5:
        change_idx = 0
    elif delta < 0:
        change_idx = 1
    elif delta < 5:
        change_idx = 2
    else:
        change_idx = 3
    raw.extend(one_hot(change_idx, 4))

    arr = np.asarray(raw, dtype=np.float32)

    if arr.shape != (ENV_FEATURES,):
        raise RuntimeError(f"Environment vector has shape {arr.shape}, expected (96,)")

    return arr


def build_environment_sequence(observations: List[Observation]) -> np.ndarray:
    """Build one 96-D environment vector for every historical timestep."""
    vectors = []

    for i in range(len(observations)):
        prefix = observations[: i + 1]

        # For early points, pad with the earliest observation so derived
        # history-dependent features remain defined.
        if len(prefix) < 5:
            prefix = [prefix[0]] * (5 - len(prefix)) + prefix

        vectors.append(build_environment_raw(prefix))

    return np.stack(vectors, axis=0).astype(np.float32)


# ============================================================
# GFS 0.25-degree GRIB Filter
# ============================================================


def gfs_url(timestamp: datetime, latitude: float, longitude: float) -> str:
    date = timestamp.strftime("%Y%m%d")
    cycle = timestamp.strftime("%H")

    # GFS operational cycles are 00/06/12/18 UTC.
    if cycle not in {"00", "06", "12", "18"}:
        raise RuntimeError(f"Unsupported GFS cycle hour: {cycle}")

    lon = normalize_longitude(longitude)
    lon0 = lon - 10.0
    lon1 = lon + 10.0
    lat0 = latitude - 10.0
    lat1 = latitude + 10.0

    lon0 = max(-180.0, lon0)
    lon1 = min(180.0, lon1)
    lat0 = max(-90.0, lat0)
    lat1 = min(90.0, lat1)

    # NOAA NOMADS GFS 0.25-degree GRIB filter.
    params = [
        ("file", f"gfs.t{cycle}z.pgrb2.0p25.f000"),
        ("lev_200_mb", "on"),
        ("lev_500_mb", "on"),
        ("lev_850_mb", "on"),
        ("lev_925_mb", "on"),
        ("lev_surface", "on"),
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


def download_gfs(timestamp: datetime, latitude: float, longitude: float) -> bytes:
    url = gfs_url(timestamp, latitude, longitude)
    response = requests.get(url, timeout=GFS_TIMEOUT)

    if response.status_code != 200:
        raise RuntimeError(
            f"NOAA GFS returned HTTP {response.status_code}: {response.text[:300]}"
        )

    content_type = response.headers.get("content-type", "").lower()
    if not response.content or b"<html" in response.content[:200].lower():
        raise RuntimeError("NOAA returned a non-GRIB response")

    if "text/html" in content_type:
        raise RuntimeError("NOAA returned HTML instead of GRIB2 data")

    return response.content


def collect_cfgrib_datasets(grib_path: str):
    # open_datasets handles the multiple GRIB message groups that result when
    # several levels/parameters are selected from one filtered file.
    datasets = cfgrib.open_datasets(
        grib_path,
        backend_kwargs={"indexpath": ""},
    )
    return datasets


def find_field(datasets, short_name: str, level_type: str, level_value: float = None):
    candidates = []

    for ds in datasets:
        for var_name in ds.data_vars:
            var = ds[var_name]
            attrs = var.attrs
            short = attrs.get("GRIB_shortName", var_name)
            type_of_level = attrs.get("GRIB_typeOfLevel")
            level = attrs.get("GRIB_level")

            if short != short_name:
                continue
            if level_type is not None and type_of_level != level_type:
                continue
            if level_value is not None and level is not None:
                if float(level) != float(level_value):
                    continue

            candidates.append(var)

    if not candidates:
        raise KeyError(
            f"Could not find GRIB field {short_name} / {level_type} / {level_value}"
        )

    return candidates[0]


def to_2d_numpy(da) -> np.ndarray:
    arr = np.asarray(da.values, dtype=np.float32)

    while arr.ndim > 2:
        arr = arr[0]

    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D atmospheric field, got {arr.shape}")

    return arr


def resize_81x81(arr: np.ndarray) -> np.ndarray:
    if arr.shape == (GRID_SIZE, GRID_SIZE):
        return arr.astype(np.float32)

    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=(GRID_SIZE, GRID_SIZE),
        mode="bilinear",
        align_corners=True,
    )
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


def standardize_channel(channel: np.ndarray) -> np.ndarray:
    channel = np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(np.mean(channel))
    std = float(np.std(channel))

    if std < 1e-6:
        return np.zeros_like(channel, dtype=np.float32)

    standardized = (channel - mean) / std
    return np.clip(standardized, -10.0, 10.0).astype(np.float32)


def extract_gfs_tensor(grib_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib_bytes)
        grib_path = tmp.name

    try:
        datasets = collect_cfgrib_datasets(grib_path)

        channels = []

        # Training order: 4 U + 4 V + 4 Z/HGT + 1 surface field.
        for level in PRESSURE_LEVELS:
            u = to_2d_numpy(
                find_field(datasets, "u", "isobaricInhPa", level)
            )
            channels.append(resize_81x81(standardize_channel(u)))

        for level in PRESSURE_LEVELS:
            v = to_2d_numpy(
                find_field(datasets, "v", "isobaricInhPa", level)
            )
            channels.append(resize_81x81(standardize_channel(v)))

        for level in PRESSURE_LEVELS:
            z = to_2d_numpy(
                find_field(datasets, "gh", "isobaricInhPa", level)
            )
            channels.append(resize_81x81(standardize_channel(z)))

        # The training data's 13th channel was SST. GFS 0.25 operational
        # surface TMP is used here as the live-data proxy.
        surface_tmp = to_2d_numpy(
            find_field(datasets, "2t", "heightAboveGround", 2)
        )
        channels.append(resize_81x81(standardize_channel(surface_tmp)))

        tensor = np.stack(channels, axis=0).astype(np.float32)

        if tensor.shape != (THREE_D_CHANNELS, GRID_SIZE, GRID_SIZE):
            raise RuntimeError(
                f"GFS tensor shape is {tensor.shape}; expected "
                f"({THREE_D_CHANNELS}, {GRID_SIZE}, {GRID_SIZE})"
            )

        return tensor

    finally:
        try:
            os.remove(grib_path)
        except OSError:
            pass


def fetch_one_gfs_frame(timestamp: datetime, latitude: float, longitude: float):
    data = download_gfs(timestamp, latitude, longitude)
    return extract_gfs_tensor(data)


# ============================================================
# Inference
# ============================================================


def build_track_array(observations: List[Observation]) -> np.ndarray:
    track = np.array(
        [
            [
                normalize_longitude(o.longitude),
                o.latitude,
                o.pressure,
                o.wind,
            ]
            for o in observations
        ],
        dtype=np.float32,
    )
    return track


def inverse_transform_predictions(pred_scaled: np.ndarray) -> np.ndarray:
    """
    CRITICAL STEP:
    Model outputs are in target-scaled space. Convert them back to the original
    [longitude, latitude, pressure, wind] units using target_scaler.
    """
    if target_scaler is None:
        raise RuntimeError("target_scaler is not loaded")

    original_shape = pred_scaled.shape
    flat = pred_scaled.reshape(-1, TRACK_FEATURES)

    physical = target_scaler.inverse_transform(flat)
    return physical.reshape(original_shape).astype(np.float32)


def sanitize_physical_prediction(row: np.ndarray) -> Dict[str, float]:
    # Do not clamp pressure or latitude here. The purpose of this step is to
    # expose the actual inverse-transformed model output rather than hide a
    # model/scaler problem behind artificial physical limits.
    longitude = float(normalize_longitude(float(row[0])))
    latitude = float(row[1])
    pressure_hpa = float(row[2])

    # Wind cannot be physically negative, and the deployed API intentionally
    # treats the model output as absolute wind speed, not wind change.
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
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if LOAD_ERROR is None else "unhealthy",
        "model_loaded": LOAD_ERROR is None,
        "device": str(DEVICE),
        "load_error": LOAD_ERROR,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if LOAD_ERROR is not None:
        raise HTTPException(
            status_code=500,
            detail=f"Model/scaler loading failed: {LOAD_ERROR}",
        )

    observations = validate_observations(request.observations)

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
        three_d_frames = []
        mask = []
        gfs_failures = []
        gfs_frames_available = 0
        gfs_frames_missing = 0

        for obs in observations:
            ts = parse_timestamp(obs.timestamp)

            try:
                frame = fetch_one_gfs_frame(
                    ts,
                    float(obs.latitude),
                    float(obs.longitude),
                )
                three_d_frames.append(frame)
                mask.append(1.0)
                gfs_frames_available += 1

            except Exception as exc:
                print(
                    f"GFS failure for {obs.timestamp}: {type(exc).__name__}: {exc}"
                )
                three_d_frames.append(
                    np.zeros(
                        (THREE_D_CHANNELS, GRID_SIZE, GRID_SIZE),
                        dtype=np.float32,
                    )
                )
                mask.append(0.0)
                gfs_frames_missing += 1
                gfs_failures.append(
                    {
                        "timestamp": obs.timestamp,
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    }
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

        return {
            "last_observed_timestamp": observations[-1].timestamp,
            "forecast_hours": forecast_hours,
            "atmospheric_source": "NOAA GFS 0.25 degree",
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
