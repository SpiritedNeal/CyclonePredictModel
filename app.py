import os
import math
import time
import tempfile
from datetime import datetime, timedelta

import numpy as np
import requests
import xarray as xr
import pickle

import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "cyclone_multimodal.pth"
)

SCALER_PATH = os.getenv(
    "SCALER_PATH",
    "cyclone_scalers.pkl"
)

INPUT_STEPS = 8
OUTPUT_STEPS = 4

TRACK_FEATURES = 4
ENV_FEATURES = 96

THREE_D_CHANNELS = 13
GRID_FEATURES = 64
MASK_FEATURES = 1

GRID_SIZE = 81

PRESSURE_LEVELS = [
    200,
    500,
    850,
    925
]

# NOAA asks automated users to pause between requests.
GFS_REQUEST_DELAY = float(
    os.getenv(
        "GFS_REQUEST_DELAY",
        "10"
    )
)

GFS_TIMEOUT = int(
    os.getenv(
        "GFS_TIMEOUT",
        "90"
    )
)

GFS_MAX_LOOKBACK_HOURS = int(
    os.getenv(
        "GFS_MAX_LOOKBACK_HOURS",
        "48"
    )
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

AREA_ORDER = [
    "EP",
    "NA",
    "NI",
    "SI",
    "SP",
    "WP"
]

VELOCITY_DIVISOR = 1219.8387650082498


# ============================================================
# MODEL
# ============================================================

class GridEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                THREE_D_CHANNELS,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.fc = nn.Linear(
            128,
            GRID_FEATURES
        )


    def forward(self, x):

        x = self.network(x)

        x = x.flatten(
            start_dim=1
        )

        return self.fc(x)


class CycloneModel(nn.Module):

    def __init__(self):
        super().__init__()

        lstm_input = (
            TRACK_FEATURES
            + ENV_FEATURES
            + GRID_FEATURES
            + MASK_FEATURES
        )

        self.grid_encoder = GridEncoder()

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                OUTPUT_STEPS * 4
            )
        )


    def forward(
        self,
        track,
        env,
        three_d,
        mask
    ):

        batch_size = track.shape[0]
        time_steps = track.shape[1]

        three_d_flat = three_d.reshape(
            batch_size * time_steps,
            THREE_D_CHANNELS,
            GRID_SIZE,
            GRID_SIZE
        )

        grid_features = self.grid_encoder(
            three_d_flat
        )

        grid_features = grid_features.reshape(
            batch_size,
            time_steps,
            GRID_FEATURES
        )

        mask = mask.unsqueeze(-1)

        combined = torch.cat(
            [
                track,
                env,
                grid_features,
                mask
            ],
            dim=-1
        )

        lstm_output, _ = self.lstm(
            combined
        )

        last_output = lstm_output[
            :, -1, :
        ]

        output = self.decoder(
            last_output
        )

        return output.reshape(
            batch_size,
            OUTPUT_STEPS,
            4
        )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Cyclone Multimodal Prediction API",
    version="3.0.0",
    description=(
        "Predicts the next 24 hours from eight historical cyclone "
        "observations. Environmental features are reconstructed "
        "from the track and 3D atmospheric fields are retrieved "
        "from NOAA GFS."
    )
)

model = None
scalers = None
startup_error = None


# ============================================================
# LOAD MODEL
# ============================================================

@app.on_event("startup")
def load_artifacts():

    global model
    global scalers
    global startup_error

    try:

        if not os.path.isfile(
            MODEL_PATH
        ):

            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        if not os.path.isfile(
            SCALER_PATH
        ):

            raise FileNotFoundError(
                f"Scaler file not found: {SCALER_PATH}"
            )

        state_dict = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        model = CycloneModel().to(
            DEVICE
        )

        model.load_state_dict(
            state_dict
        )

        model.eval()

        with open(
            SCALER_PATH,
            "rb"
        ) as f:

            scalers = pickle.load(f)

        startup_error = None

        print("=" * 60)
        print("Cyclone API started")
        print(f"Device: {DEVICE}")
        print(f"Model: {MODEL_PATH}")
        print(f"Scalers: {SCALER_PATH}")
        print(
            f"GFS request delay: "
            f"{GFS_REQUEST_DELAY}s"
        )
        print("=" * 60)

    except Exception as e:

        model = None
        scalers = None
        startup_error = str(e)

        print(
            "STARTUP ERROR:",
            e
        )


# ============================================================
# INPUT SCHEMA
# ============================================================

class Observation(BaseModel):

    timestamp: str = Field(
        ...,
        description="UTC timestamp in YYYYMMDDHH format."
    )

    longitude: float

    latitude: float

    pressure: float = Field(
        ...,
        description="Central pressure in hPa."
    )

    wind: float = Field(
        ...,
        description="Maximum sustained wind in m/s."
    )


class PredictionRequest(BaseModel):

    observations: list[Observation] = Field(
        ...,
        min_length=8,
        max_length=8,
        description=(
            "Exactly 8 observations, "
            "6 hours apart."
        )
    )


# ============================================================
# BASIC HELPERS
# ============================================================

def parse_timestamp(timestamp):

    try:

        return datetime.strptime(
            str(timestamp),
            "%Y%m%d%H"
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid timestamp "
                f"'{timestamp}'. "
                "Use YYYYMMDDHH."
            )
        ) from exc


def normalize_longitude(longitude):

    value = longitude % 360.0

    if value < 0:
        value += 360.0

    return value


def validate_observations(
    observations
):

    if len(observations) != 8:

        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly 8 observations "
                "are required."
            )
        )

    dates = [
        parse_timestamp(
            x.timestamp
        )
        for x in observations
    ]

    if dates != sorted(dates):

        raise HTTPException(
            status_code=400,
            detail=(
                "Observations must be "
                "chronological."
            )
        )

    for i in range(7):

        if (
            dates[i + 1]
            - dates[i]
        ) != timedelta(hours=6):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Observations must be "
                    "exactly 6 hours apart."
                )
            )

    for observation in observations:

        if not (
            -180
            <= observation.longitude
            <= 360
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Longitude must be "
                    "between -180 and 360."
                )
            )

        if not (
            -90
            <= observation.latitude
            <= 90
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Latitude must be "
                    "between -90 and 90."
                )
            )

        if observation.pressure <= 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Pressure must be positive."
                )
            )

        if observation.wind < 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Wind must be non-negative "
                    "and in m/s."
                )
            )


# ============================================================
# TCND ENVIRONMENT FEATURES
# ============================================================

def get_intensity(wind):

    if wind < 17.1:

        intensity_class = 0

    elif wind < 24.4:

        intensity_class = 1

    elif wind < 32.6:

        intensity_class = 2

    elif wind < 41.4:

        intensity_class = 3

    elif wind < 50.9:

        intensity_class = 4

    else:

        intensity_class = 5

    result = np.zeros(
        6,
        dtype=np.float64
    )

    result[
        intensity_class
    ] = 1.0

    return result


def get_velocity(
    longitudes,
    latitudes
):

    long_rel = (
        longitudes[1]
        - longitudes[0]
    )

    lat_rel = (
        latitudes[1]
        - latitudes[0]
    )

    long_distance = (
        (long_rel / 10.0)
        * 111.0
        * math.cos(
            (
                (
                    latitudes[0]
                    + latitudes[1]
                )
                / 2.0
            )
            / 10.0
            * math.pi
            / 180.0
        )
    )

    lat_distance = (
        lat_rel
        / 10.0
        * 111.0
    )

    return math.sqrt(
        long_distance ** 2
        + lat_distance ** 2
    )


def get_location_for_all(
    longitude,
    latitude
):

    x = int(
        (longitude // 10) // 10
    )

    y = int(
        (latitude + 600) // 100
    )

    x = max(
        0,
        min(35, x)
    )

    y = max(
        0,
        min(11, y)
    )

    location_x = np.zeros(
        36,
        dtype=np.float64
    )

    location_y = np.zeros(
        12,
        dtype=np.float64
    )

    location_x[x] = 1.0
    location_y[y] = 1.0

    return (
        location_x,
        location_y
    )


def get_direction(
    longitudes,
    latitudes
):

    longitudes = list(
        longitudes
    )

    latitudes = list(
        latitudes
    )

    long_rel = (
        longitudes[-1]
        - longitudes[0]
    )

    lat_rel = (
        latitudes[-1]
        - latitudes[0]
    )

    long_distance = (
        (long_rel / 10.0)
        * 111.0
        * math.cos(
            (
                (
                    latitudes[0]
                    + latitudes[1]
                )
                / 2.0
            )
            / 10.0
            * math.pi
            / 180.0
        )
    )

    lat_distance = (
        lat_rel
        / 10.0
        * 111.0
    )

    velocity = math.sqrt(
        long_distance ** 2
        + lat_distance ** 2
    )

    if velocity == 0:

        return np.array(
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            dtype=np.float64
        )

    sin_angle = (
        lat_distance
        / velocity
    )

    cos_angle = (
        long_distance
        / velocity
    )

    sin_angle = max(
        -1.0,
        min(1.0, sin_angle)
    )

    if (
        sin_angle >= 0
        and cos_angle >= 0
    ):

        angle = math.asin(
            sin_angle
        )

    elif (
        sin_angle >= 0
        and cos_angle <= 0
    ):

        angle = (
            math.pi
            - math.asin(
                sin_angle
            )
        )

    elif (
        sin_angle <= 0
        and cos_angle <= 0
    ):

        angle = (
            math.pi
            - math.asin(
                sin_angle
            )
        )

    elif (
        sin_angle <= 0
        and cos_angle >= 0
    ):

        angle = (
            2 * math.pi
            + math.asin(
                sin_angle
            )
        )

    else:

        angle = 0.0

    angle_list = [

        (
            math.pi * (1 / 8),
            2 * math.pi
            - math.pi * (1 / 8)
        ),

        (
            2 * math.pi
            - math.pi * (1 / 8),
            2 * math.pi
            - math.pi * (3 / 8)
        ),

        (
            2 * math.pi
            - math.pi * (3 / 8),
            2 * math.pi
            - math.pi * (5 / 8)
        ),

        (
            2 * math.pi
            - math.pi * (5 / 8),
            2 * math.pi
            - math.pi * (7 / 8)
        ),

        (
            2 * math.pi
            - math.pi * (7 / 8),
            2 * math.pi
            - math.pi * (9 / 8)
        ),

        (
            2 * math.pi
            - math.pi * (9 / 8),
            2 * math.pi
            - math.pi * (11 / 8)
        ),

        (
            2 * math.pi
            - math.pi * (11 / 8),
            2 * math.pi
            - math.pi * (13 / 8)
        ),

        (
            2 * math.pi
            - math.pi * (13 / 8),
            2 * math.pi
            - math.pi * (15 / 8)
        )
    ]

    angle_class = 0

    for class_id, angle_range in enumerate(
        angle_list
    ):

        lower = angle_range[0]
        upper = angle_range[1]

        if class_id == 0:

            if (
                angle > upper
                or angle <= lower
            ):

                angle_class = class_id
                break

        else:

            if (
                angle > upper
                and angle < lower
            ):

                angle_class = class_id
                break

    result = np.zeros(
        8,
        dtype=np.float64
    )

    result[
        angle_class
    ] = 1.0

    return result


def get_intensity_change(
    winds
):

    winds = list(
        winds
    )

    gradients = []

    for i in range(
        len(winds) - 1
    ):

        gradients.append(
            winds[i + 1]
            - winds[i]
        )

    gradients = np.asarray(
        gradients,
        dtype=np.float64
    )

    if np.all(
        gradients == 0
    ):

        intensity_class = 3

    elif np.all(
        gradients >= 0
    ):

        intensity_class = 0

    elif np.all(
        gradients <= 0
    ):

        intensity_class = 2

    else:

        nonzero = gradients[
            gradients != 0
        ]

        if (
            len(nonzero) > 0
            and nonzero[0] > 0
            and nonzero[-1] < 0
        ):

            intensity_class = 1

        else:

            total = np.sum(
                gradients
            )

            if total > 0:

                intensity_class = 0

            elif total < 0:

                intensity_class = 2

            else:

                intensity_class = 3

    result = np.zeros(
        4,
        dtype=np.float64
    )

    result[
        intensity_class
    ] = 1.0

    return result


def build_environment_vector(
    observations,
    index
):

    current = observations[
        index
    ]

    parts = []

    # Area = NI.
    area = np.zeros(
        6,
        dtype=np.float64
    )

    area[
        AREA_ORDER.index("NI")
    ] = 1.0

    parts.append(
        area
    )

    # Wind.
    parts.append(
        np.array(
            [
                current.wind / 110.0
            ],
            dtype=np.float64
        )
    )

    # Intensity class.
    parts.append(
        get_intensity(
            current.wind
        )
    )

    # Movement velocity.
    if index == 0:

        movement_velocity = 0.0

    else:

        previous = observations[
            index - 1
        ]

        movement_velocity = (
            get_velocity(
                [
                    previous.longitude,
                    current.longitude
                ],
                [
                    previous.latitude,
                    current.latitude
                ]
            )
            / VELOCITY_DIVISOR
        )

    parts.append(
        np.array(
            [movement_velocity],
            dtype=np.float64
        )
    )

    # Month.
    month = parse_timestamp(
        current.timestamp
    ).month

    month_onehot = np.zeros(
        12,
        dtype=np.float64
    )

    month_onehot[
        month - 1
    ] = 1.0

    parts.append(
        month_onehot
    )

    # Raw location.
    parts.append(
        np.array(
            [
                current.longitude,
                current.latitude
            ],
            dtype=np.float64
        )
    )

    # Location one-hot.
    longitude_360 = normalize_longitude(
        current.longitude
    )

    location_long, location_lat = (
        get_location_for_all(
            longitude_360,
            current.latitude
        )
    )

    parts.append(
        location_long
    )

    parts.append(
        location_lat
    )

    # 12-hour movement direction.
    if index < 2:

        direction12 = np.zeros(
            8,
            dtype=np.float64
        )

    else:

        window = observations[
            index - 2:
            index + 1
        ]

        direction12 = get_direction(
            [
                x.longitude
                for x in window
            ],
            [
                x.latitude
                for x in window
            ]
        )

    parts.append(
        direction12
    )

    # 24-hour movement direction.
    if index < 4:

        direction24 = np.zeros(
            8,
            dtype=np.float64
        )

    else:

        window = observations[
            index - 4:
            index + 1
        ]

        direction24 = get_direction(
            [
                x.longitude
                for x in window
            ],
            [
                x.latitude
                for x in window
            ]
        )

    parts.append(
        direction24
    )

    # 24-hour intensity change.
    if index < 4:

        intensity_change = np.zeros(
            4,
            dtype=np.float64
        )

    else:

        window = observations[
            index - 4:
            index + 1
        ]

        intensity_change = (
            get_intensity_change(
                [
                    x.wind
                    for x in window
                ]
            )
        )

    parts.append(
        intensity_change
    )

    vector = np.concatenate(
        parts
    )

    if vector.shape != (
        ENV_FEATURES,
    ):

        raise RuntimeError(
            f"Environment vector has "
            f"shape {vector.shape}; "
            f"expected "
            f"({ENV_FEATURES},)."
        )

    return vector


# ============================================================
# NOAA GFS GRIB2
# ============================================================

def build_gfs_url(
    cycle_dt,
    forecast_hour,
    latitude,
    longitude
):

    date_string = (
        cycle_dt.strftime(
            "%Y%m%d"
        )
    )

    cycle_hour = (
        cycle_dt.strftime(
            "%H"
        )
    )

    fhr = (
        f"{forecast_hour:03d}"
    )

    center_lon = normalize_longitude(
        longitude
    )

    left_lon = max(
        0.0,
        center_lon - 10.25
    )

    right_lon = min(
        359.75,
        center_lon + 10.25
    )

    bottom_lat = max(
        -90.0,
        latitude - 10.25
    )

    top_lat = min(
        90.0,
        latitude + 10.25
    )

    params = {

        "file":
            f"gfs.t{cycle_hour}z."
            f"pgrb2.0p25.f{fhr}",

        "lev_200_mb":
            "on",

        "lev_500_mb":
            "on",

        "lev_850_mb":
            "on",

        "lev_925_mb":
            "on",

        "lev_surface":
            "on",

        "var_HGT":
            "on",

        "var_TMP":
            "on",

        "var_UGRD":
            "on",

        "var_VGRD":
            "on",

        "subregion":
            "",

        "leftlon":
            f"{left_lon:.2f}",

        "rightlon":
            f"{right_lon:.2f}",

        "toplat":
            f"{top_lat:.2f}",

        "bottomlat":
            f"{bottom_lat:.2f}",

        "dir":
            f"/gfs.{date_string}/"
            f"{cycle_hour}/atmos"
    }

    request = requests.Request(
        "GET",
        (
            "https://nomads.ncep.noaa.gov/"
            "cgi-bin/filter_gfs_0p25.pl"
        ),
        params=params
    ).prepare()

    return request.url


def choose_gfs_candidate(
    target_dt
):

    candidates = []

    # Exact analysis.
    candidates.append(
        (
            target_dt,
            0
        )
    )

    # Older cycles + forecast hours.
    for hours_back in range(
        6,
        GFS_MAX_LOOKBACK_HOURS + 1,
        6
    ):

        cycle_dt = (
            target_dt
            - timedelta(
                hours=hours_back
            )
        )

        if cycle_dt.hour % 6 != 0:

            continue

        candidates.append(
            (
                cycle_dt,
                hours_back
            )
        )

    return candidates


def download_gfs_file(
    target_timestamp,
    latitude,
    longitude
):

    target_dt = parse_timestamp(
        target_timestamp
    )

    last_error = None

    candidates = (
        choose_gfs_candidate(
            target_dt
        )
    )

    for candidate_index, (
        cycle_dt,
        forecast_hour
    ) in enumerate(candidates):

        if candidate_index > 0:

            time.sleep(
                GFS_REQUEST_DELAY
            )

        url = build_gfs_url(
            cycle_dt,
            forecast_hour,
            latitude,
            longitude
        )

        try:

            response = requests.get(
                url,
                timeout=GFS_TIMEOUT
            )

            if response.status_code != 200:

                last_error = (
                    f"HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:250]}"
                )

                continue

            if (
                response.content[:4]
                != b"GRIB"
            ):

                last_error = (
                    "NOAA returned a "
                    "non-GRIB response: "
                    f"{response.text[:250]}"
                )

                continue

            temp_file = (
                tempfile.NamedTemporaryFile(
                    suffix=".grib2",
                    delete=False
                )
            )

            temp_path = (
                temp_file.name
            )

            try:

                temp_file.write(
                    response.content
                )

            finally:

                temp_file.close()

            return (
                temp_path,
                url
            )

        except Exception as exc:

            last_error = str(exc)

    raise RuntimeError(
        "Could not obtain GFS data "
        f"for {target_timestamp}. "
        f"Last error: {last_error}"
    )


def open_cfgrib_dataset(
    path,
    filter_by_keys
):

    return xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys":
                filter_by_keys,
            "indexpath":
                ""
        }
    )


def get_variable(
    dataset,
    names
):

    for name in names:

        if name in dataset.data_vars:

            return dataset[name]

    raise KeyError(
        "Could not find variables "
        f"{names}. Available: "
        f"{list(dataset.data_vars)}"
    )


def centered_indices(
    values,
    center_value
):

    values = np.asarray(
        values
    )

    nearest = int(
        np.argmin(
            np.abs(
                values
                - center_value
            )
        )
    )

    start = (
        nearest - 40
    )

    end = (
        nearest + 41
    )

    if start < 0:

        end -= start
        start = 0

    if end > len(values):

        start -= (
            end - len(values)
        )

        end = len(values)

    start = max(
        0,
        start
    )

    end = min(
        len(values),
        end
    )

    return np.arange(
        start,
        end
    )


def force_81x81(
    data,
    lat_values,
    lon_values,
    center_lat,
    center_lon
):

    lat_idx = centered_indices(
        lat_values,
        center_lat
    )

    lon_idx = centered_indices(
        lon_values,
        center_lon
    )

    result = data[
        ...,
        lat_idx,
        :
    ]

    result = result[
        ...,
        :,
        lon_idx
    ]

    if result.shape[-2] < GRID_SIZE:

        pad = (
            GRID_SIZE
            - result.shape[-2]
        )

        result = np.pad(
            result,
            [(0, 0)]
            * (result.ndim - 2)
            + [
                (0, pad),
                (0, 0)
            ],
            mode="edge"
        )

    if result.shape[-1] < GRID_SIZE:

        pad = (
            GRID_SIZE
            - result.shape[-1]
        )

        result = np.pad(
            result,
            [(0, 0)]
            * (result.ndim - 2)
            + [
                (0, 0),
                (0, pad)
            ],
            mode="edge"
        )

    return result[
        ...,
        :GRID_SIZE,
        :GRID_SIZE
    ]


def standardize_channel(
    channel
):

    channel = np.asarray(
        channel,
        dtype=np.float64
    )

    valid = (
        np.isfinite(channel)
        &
        (
            np.abs(channel)
            < 1e6
        )
    )

    output = np.zeros_like(
        channel,
        dtype=np.float64
    )

    if np.any(valid):

        values = channel[
            valid
        ]

        mean = np.mean(
            values,
            dtype=np.float64
        )

        std = np.std(
            values,
            dtype=np.float64
        )

        if not np.isfinite(
            mean
        ):

            mean = 0.0

        if (
            not np.isfinite(std)
            or std < 1e-8
        ):

            std = 1.0

        output[
            valid
        ] = (
            channel[valid]
            - mean
        ) / std

    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return np.clip(
        output,
        -10.0,
        10.0
    ).astype(
        np.float32
    )


def extract_gfs_tensor(
    grib_path,
    center_lat,
    center_lon
):

    # --------------------------------------------------------
    # Pressure-level data
    # --------------------------------------------------------

    pressure_ds = (
        open_cfgrib_dataset(
            grib_path,
            {
                "typeOfLevel":
                    "isobaricInhPa"
            }
        )
    )

    try:

        u = get_variable(
            pressure_ds,
            [
                "u",
                "ugrd"
            ]
        )

        v = get_variable(
            pressure_ds,
            [
                "v",
                "vgrd"
            ]
        )

        z = get_variable(
            pressure_ds,
            [
                "gh",
                "z",
                "hgt"
            ]
        )

        if (
            "isobaricInhPa"
            in u.coords
        ):

            level_coord = (
                "isobaricInhPa"
            )

        elif "level" in u.coords:

            level_coord = "level"

        else:

            raise RuntimeError(
                "Could not find pressure "
                "level coordinate."
            )

        u = u.sel(
            {
                level_coord:
                    PRESSURE_LEVELS
            },
            method="nearest"
        )

        v = v.sel(
            {
                level_coord:
                    PRESSURE_LEVELS
            },
            method="nearest"
        )

        z = z.sel(
            {
                level_coord:
                    PRESSURE_LEVELS
            },
            method="nearest"
        )

        u = u.transpose(
            level_coord,
            "latitude",
            "longitude"
        )

        v = v.transpose(
            level_coord,
            "latitude",
            "longitude"
        )

        z = z.transpose(
            level_coord,
            "latitude",
            "longitude"
        )

        lat_values = (
            u["latitude"].values
        )

        lon_values = (
            u["longitude"].values
        )

        u_data = np.asarray(
            u.values,
            dtype=np.float64
        )

        v_data = np.asarray(
            v.values,
            dtype=np.float64
        )

        z_data = np.asarray(
            z.values,
            dtype=np.float64
        )

        u_data = force_81x81(
            u_data,
            lat_values,
            lon_values,
            center_lat,
            center_lon
        )

        v_data = force_81x81(
            v_data,
            lat_values,
            lon_values,
            center_lat,
            center_lon
        )

        z_data = force_81x81(
            z_data,
            lat_values,
            lon_values,
            center_lat,
            center_lon
        )

    finally:

        pressure_ds.close()

    # --------------------------------------------------------
    # Surface temperature
    # --------------------------------------------------------

    surface_ds = (
        open_cfgrib_dataset(
            grib_path,
            {
                "typeOfLevel":
                    "surface",
                "shortName":
                    "t"
            }
        )
    )

    try:

        surface_t = get_variable(
            surface_ds,
            [
                "t",
                "tmp"
            ]
        )

        surface_t = (
            surface_t.transpose(
                "latitude",
                "longitude"
            )
        )

        lat_values = (
            surface_t[
                "latitude"
            ].values
        )

        lon_values = (
            surface_t[
                "longitude"
            ].values
        )

        surface_t_data = np.asarray(
            surface_t.values,
            dtype=np.float64
        )

        surface_t_data = force_81x81(
            surface_t_data,
            lat_values,
            lon_values,
            center_lat,
            center_lon
        )

    finally:

        surface_ds.close()

    # --------------------------------------------------------
    # 13 channels
    # --------------------------------------------------------

    processed = [

        standardize_channel(
            u_data[0]
        ),

        standardize_channel(
            u_data[1]
        ),

        standardize_channel(
            u_data[2]
        ),

        standardize_channel(
            u_data[3]
        ),

        standardize_channel(
            v_data[0]
        ),

        standardize_channel(
            v_data[1]
        ),

        standardize_channel(
            v_data[2]
        ),

        standardize_channel(
            v_data[3]
        ),

        standardize_channel(
            z_data[0]
        ),

        standardize_channel(
            z_data[1]
        ),

        standardize_channel(
            z_data[2]
        ),

        standardize_channel(
            z_data[3]
        ),

        standardize_channel(
            surface_t_data
        )
    ]

    tensor = np.stack(
        processed,
        axis=0
    ).astype(
        np.float32
    )

    expected = (
        THREE_D_CHANNELS,
        GRID_SIZE,
        GRID_SIZE
    )

    if tensor.shape != expected:

        raise RuntimeError(
            f"GFS tensor has shape "
            f"{tensor.shape}; "
            f"expected {expected}."
        )

    return tensor


def fetch_one_3d_frame(
    observation
):

    grib_path = None

    try:

        grib_path, url = (
            download_gfs_file(
                observation.timestamp,
                observation.latitude,
                observation.longitude
            )
        )

        tensor = extract_gfs_tensor(
            grib_path,
            observation.latitude,
            normalize_longitude(
                observation.longitude
            )
        )

        return (
            tensor,
            1.0,
            url
        )

    finally:

        if grib_path is not None:

            try:

                os.remove(
                    grib_path
                )

            except OSError:

                pass


# ============================================================
# INPUT CONSTRUCTION
# ============================================================

def build_track_array(
    observations
):

    return np.asarray(
        [
            [
                observation.longitude,
                observation.latitude,
                observation.pressure,
                observation.wind
            ]
            for observation in observations
        ],
        dtype=np.float64
    )


def build_environment_array(
    observations
):

    return np.asarray(
        [
            build_environment_vector(
                observations,
                i
            )
            for i in range(
                INPUT_STEPS
            )
        ],
        dtype=np.float64
    )


def build_3d_array(
    observations
):

    frames = []
    mask = []
    source_urls = []
    failures = []

    for observation in observations:

        try:

            frame, frame_mask, source_url = (
                fetch_one_3d_frame(
                    observation
                )
            )

            frames.append(
                frame
            )

            mask.append(
                frame_mask
            )

            source_urls.append(
                source_url
            )

        except Exception as exc:

            print(
                "GFS error for "
                f"{observation.timestamp}: "
                f"{exc}"
            )

            frames.append(
                np.zeros(
                    (
                        THREE_D_CHANNELS,
                        GRID_SIZE,
                        GRID_SIZE
                    ),
                    dtype=np.float32
                )
            )

            mask.append(
                0.0
            )

            source_urls.append(
                None
            )

            failures.append(
                {
                    "timestamp":
                        observation.timestamp,

                    "error":
                        str(exc)
                }
            )

    return (

        np.stack(
            frames,
            axis=0
        ),

        np.asarray(
            mask,
            dtype=np.float32
        ),

        source_urls,

        failures
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

def run_prediction(
    observations
):

    track = build_track_array(
        observations
    )

    env = build_environment_array(
        observations
    )

    (
        three_d,
        mask,
        source_urls,
        gfs_failures
    ) = build_3d_array(
        observations
    )

    if int(
        np.sum(mask)
    ) == 0:

        raise HTTPException(
            status_code=503,
            detail=(
                "NOAA GFS atmospheric data "
                "could not be retrieved for "
                "any of the eight observations."
            )
        )

    try:

        track_scaled = (
            scalers[
                "track"
            ].transform(
                track
            )
        )

        env_scaled = (
            scalers[
                "env"
            ].transform(
                env
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Input scaling failed: "
                f"{exc}"
            )
        ) from exc

    track_scaled = np.nan_to_num(
        track_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(
        np.float32
    )

    env_scaled = np.nan_to_num(
        env_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(
        np.float32
    )

    track_tensor = (
        torch.from_numpy(
            track_scaled
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    env_tensor = (
        torch.from_numpy(
            env_scaled
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    three_d_tensor = (
        torch.from_numpy(
            three_d
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    mask_tensor = (
        torch.from_numpy(
            mask
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    try:

        with torch.no_grad():

            prediction = model(
                track_tensor,
                env_tensor,
                three_d_tensor,
                mask_tensor
            )

        prediction = (
            prediction
            .cpu()
            .numpy()[0]
        )

        prediction_original = (
            scalers[
                "target"
            ].inverse_transform(
                prediction.reshape(
                    -1,
                    4
                )
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Model inference failed: "
                f"{exc}"
            )
        ) from exc

    last_timestamp = (
        observations[
            -1
        ].timestamp
    )

    predictions = []

    for i in range(
        OUTPUT_STEPS
    ):

        longitude = float(
            prediction_original[
                i,
                0
            ]
        )

        latitude = float(
            prediction_original[
                i,
                1
            ]
        )

        pressure = float(
            prediction_original[
                i,
                2
            ]
        )

        wind_mps = float(
            prediction_original[
                i,
                3
            ]
        )

        # Prevent physically impossible negative values.
        wind_mps = max(
            0.0,
            wind_mps
        )

        pressure = max(
            0.0,
            pressure
        )

        wind_knots = (
            wind_mps
            * 1.943844492
        )

        future_dt = (
            parse_timestamp(
                last_timestamp
            )
            + timedelta(
                hours=6 * (i + 1)
            )
        )

        predictions.append(
            {
                "timestamp":
                    future_dt.strftime(
                        "%Y%m%d%H"
                    ),

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "pressure_hpa":
                    pressure,

                "wind_mps":
                    wind_mps,

                "wind_knots":
                    wind_knots
            }
        )

    return {
        "last_observed_timestamp":
            last_timestamp,

        "forecast_hours":
            [
                6,
                12,
                18,
                24
            ],

        "atmospheric_source":
            "NOAA GFS 0.25 degree",

        "gfs_frames_available":
            int(
                np.sum(mask)
            ),

        "gfs_frames_missing":
            int(
                INPUT_STEPS
                - np.sum(mask)
            ),

        "gfs_failures":
            gfs_failures,

        "predictions":
            predictions
    }


# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
def root():

    return {
        "service":
            "Cyclone Multimodal Prediction API",

        "status":
            "running",

        "device":
            str(DEVICE),

        "model_loaded":
            model is not None,

        "input":
            "8 observations, exactly 6 hours apart",

        "wind_unit":
            "m/s",

        "forecast":
            [
                6,
                12,
                18,
                24
            ]
    }


@app.get("/health")
def health():

    if model is None:

        return {
            "status":
                "error",

            "model_loaded":
                False,

            "error":
                startup_error
        }

    return {
        "status":
            "ok",

        "model_loaded":
            True,

        "device":
            str(DEVICE)
    }


@app.post("/predict")
def predict(
    request: PredictionRequest
):

    if (
        model is None
        or scalers is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                f"{startup_error or ''}"
            )
        )

    validate_observations(
        request.observations
    )

    return run_prediction(
        request.observations
    )
