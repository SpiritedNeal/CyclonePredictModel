import os
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pickle
import xarray as xr

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
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

# TCND uses this exact normalization for movement velocity.
VELOCITY_DIVISOR = 1219.8387650082498

CLASS_NAMES = [
    "LOW",
    "MEDIUM",
    "HIGH"
]


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
    title="Cyclone Future Prediction API",
    version="2.0.0",
    description=(
        "Accepts 8 historical cyclone observations and "
        "automatically builds the environmental and "
        "3D atmospheric inputs."
    )
)


# ============================================================
# LOAD MODEL
# ============================================================

model = None
scalers = None
startup_error = None


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
        print("Device:", DEVICE)
        print("Model:", MODEL_PATH)
        print("Scalers:", SCALER_PATH)
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
# INPUT DATA MODEL
# ============================================================

class Observation(BaseModel):

    timestamp: str = Field(
        ...,
        description="UTC timestamp in YYYYMMDDHH format."
    )

    longitude: float = Field(
        ...,
        description="Cyclone longitude in degrees."
    )

    latitude: float = Field(
        ...,
        description="Cyclone latitude in degrees."
    )

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
        description="Exactly 8 observations spaced 6 hours apart."
    )


# ============================================================
# TIME FUNCTIONS
# ============================================================

def parse_timestamp(timestamp):

    try:

        return datetime.strptime(
            str(timestamp),
            "%Y%m%d%H"
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid timestamp '{timestamp}'. "
                "Use YYYYMMDDHH."
            )
        )


def validate_observations(
    observations
):

    if len(observations) != 8:

        raise HTTPException(
            status_code=400,
            detail="Exactly 8 observations are required."
        )

    dates = [
        parse_timestamp(
            observation.timestamp
        )
        for observation in observations
    ]

    for i in range(7):

        difference = (
            dates[i + 1]
            - dates[i]
        )

        if difference != timedelta(
            hours=6
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Observations must be exactly "
                    "6 hours apart."
                )
            )

    if dates != sorted(dates):

        raise HTTPException(
            status_code=400,
            detail="Observations must be chronological."
        )

    for observation in observations:

        if not -180 <= observation.longitude <= 360:

            raise HTTPException(
                status_code=400,
                detail="Invalid longitude."
            )

        if not -90 <= observation.latitude <= 90:

            raise HTTPException(
                status_code=400,
                detail="Invalid latitude."
            )

        if observation.pressure <= 0:

            raise HTTPException(
                status_code=400,
                detail="Pressure must be positive."
            )

        if observation.wind < 0:

            raise HTTPException(
                status_code=400,
                detail="Wind must be non-negative and in m/s."
            )


# ============================================================
# TCND ENVIRONMENT FEATURES
# ============================================================

def get_intensity(
    wind
):

    intensity_class = 0

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

    output = np.zeros(
        6,
        dtype=np.float64
    )

    output[
        intensity_class
    ] = 1.0

    return output


def get_velocity(
    lon1,
    lat1,
    lon2,
    lat2
):

    long_rel = lon2 - lon1

    lat_rel = lat2 - lat1

    long_distance = (
        long_rel / 10
    ) * 111 * math.cos(
        ((lat1 + lat2) / 2)
        / 10
        * math.pi
        / 180
    )

    lat_distance = (
        lat_rel / 10
    ) * 111

    velocity = math.sqrt(
        long_distance ** 2
        + lat_distance ** 2
    )

    return velocity


def get_location_for_all(
    longitude,
    latitude
):

    # Longitude: 0-360, 10 degree bins.
    x = int(
        (longitude // 10) // 10
    )

    # Latitude: -60 to +60, 10 degree bins.
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

    location_long = np.zeros(
        36,
        dtype=np.float64
    )

    location_lat = np.zeros(
        12,
        dtype=np.float64
    )

    location_long[x] = 1.0
    location_lat[y] = 1.0

    return (
        location_long,
        location_lat
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
        long_rel / 10
    ) * 111 * math.cos(
        (
            latitudes[0]
            + latitudes[1]
        )
        / 10
        * math.pi
        / 180
    )

    lat_distance = (
        lat_rel / 10
    ) * 111

    velocity = math.sqrt(
        long_distance ** 2
        + lat_distance ** 2
    )

    if velocity == 0:

        output = np.zeros(
            8,
            dtype=np.float64
        )

        output[0] = 1.0

        return output

    sin_angle = (
        lat_distance
        / velocity
    )

    cos_angle = (
        long_distance
        / velocity
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
            - math.asin(sin_angle)
        )

    elif (
        sin_angle <= 0
        and cos_angle <= 0
    ):

        angle = (
            math.pi
            - math.asin(sin_angle)
        )

    elif (
        sin_angle <= 0
        and cos_angle >= 0
    ):

        angle = (
            2 * math.pi
            + math.asin(sin_angle)
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

    output = np.zeros(
        8,
        dtype=np.float64
    )

    output[
        angle_class
    ] = 1.0

    return output


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

    output = np.zeros(
        4,
        dtype=np.float64
    )

    output[
        intensity_class
    ] = 1.0

    return output


def build_environment_vector(
    observations,
    index
):

    observation = observations[
        index
    ]

    longitude = observation.longitude
    latitude = observation.latitude
    wind = observation.wind

    parts = []

    # --------------------------------------------------------
    # Area: NI
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Wind
    # --------------------------------------------------------

    parts.append(
        np.array(
            [wind / 110.0],
            dtype=np.float64
        )
    )

    # --------------------------------------------------------
    # Intensity class
    # --------------------------------------------------------

    parts.append(
        get_intensity(
            wind
        )
    )

    # --------------------------------------------------------
    # Movement velocity
    # --------------------------------------------------------

    if index == 0:

        movement_velocity = 0.0

    else:

        previous = observations[
            index - 1
        ]

        movement_velocity = (
            get_velocity(
                previous.longitude,
                previous.latitude,
                longitude,
                latitude
            )
            / VELOCITY_DIVISOR
        )

    parts.append(
        np.array(
            [movement_velocity],
            dtype=np.float64
        )
    )

    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------

    month = parse_timestamp(
        observation.timestamp
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

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    parts.append(
        np.array(
            [
                longitude,
                latitude
            ],
            dtype=np.float64
        )
    )

    # --------------------------------------------------------
    # Location longitude / latitude
    # --------------------------------------------------------

    longitude_360 = (
        longitude
        if longitude >= 0
        else longitude + 360
    )

    location_long, location_lat = (
        get_location_for_all(
            longitude_360,
            latitude
        )
    )

    parts.append(
        location_long
    )

    parts.append(
        location_lat
    )

    # --------------------------------------------------------
    # History direction 12h
    # --------------------------------------------------------

    if index < 2:

        direction_12 = np.zeros(
            8,
            dtype=np.float64
        )

    else:

        history = observations[
            index - 2:
            index + 1
        ]

        direction_12 = get_direction(
            [
                item.longitude
                for item in history
            ],
            [
                item.latitude
                for item in history
            ]
        )

    parts.append(
        direction_12
    )

    # --------------------------------------------------------
    # History direction 24h
    # --------------------------------------------------------

    if index < 4:

        direction_24 = np.zeros(
            8,
            dtype=np.float64
        )

    else:

        history = observations[
            index - 4:
            index + 1
        ]

        direction_24 = get_direction(
            [
                item.longitude
                for item in history
            ],
            [
                item.latitude
                for item in history
            ]
        )

    parts.append(
        direction_24
    )

    # --------------------------------------------------------
    # Intensity change 24h
    # --------------------------------------------------------

    if index < 4:

        intensity_change = np.zeros(
            4,
            dtype=np.float64
        )

    else:

        history = observations[
            index - 4:
            index + 1
        ]

        intensity_change = (
            get_intensity_change(
                [
                    item.wind
                    for item in history
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
            f"Environment vector has shape "
            f"{vector.shape}; expected "
            f"({ENV_FEATURES},)."
        )

    return vector


# ============================================================
# GFS DATA ACCESS
# ============================================================

def timestamp_to_numpy_datetime(
    timestamp
):

    dt = parse_timestamp(
        timestamp
    )

    return np.datetime64(
        dt
    )


def open_gfs_dataset(
    target_datetime
):

    # Try the exact GFS cycle first, then older
    # cycles if the current one is unavailable.

    for hours_back in [
        0,
        6,
        12,
        18,
        24,
        30,
        36,
        42,
        48
    ]:

        cycle_datetime = (
            target_datetime
            - timedelta(
                hours=hours_back
            )
        )

        cycle_datetime = cycle_datetime.replace(
            minute=0,
            second=0,
            microsecond=0
        )

        cycle_datetime = cycle_datetime.replace(
            hour=(
                cycle_datetime.hour // 6
            ) * 6
        )

        date_string = (
            cycle_datetime
            .strftime("%Y%m%d")
        )

        hour_string = (
            cycle_datetime
            .strftime("%H")
        )

        url = (
            "https://nomads.ncep.noaa.gov/"
            "dods/gfs_0p25_1hr/"
            f"gfs{date_string}/"
            f"gfs_0p25_1hr_{hour_string}z"
        )

        try:

            ds = xr.open_dataset(
                url,
                engine="pydap"
            )

            target_np = np.datetime64(
                target_datetime
            )

            time_values = pd.to_datetime(
                ds["time"].values
            ).values

            differences = np.abs(
                time_values
                - target_np
            )

            time_index = int(
                np.argmin(
                    differences
                )
            )

            difference_hours = (
                abs(
                    (
                        time_values[
                            time_index
                        ]
                        - target_np
                    )
                    / np.timedelta64(
                        1,
                        "h"
                    )
                )
            )

            if difference_hours > 1.1:

                ds.close()

                continue

            return (
                ds,
                time_index,
                url
            )

        except Exception:

            continue

    raise RuntimeError(
        "Could not obtain GFS data for "
        f"{target_datetime:%Y-%m-%d %H:%M} UTC."
    )


def pad_to_81(
    data
):

    height = data.shape[-2]
    width = data.shape[-1]

    pad_height = (
        GRID_SIZE
        - height
    )

    pad_width = (
        GRID_SIZE
        - width
    )

    if pad_height < 0:

        start = (
            -pad_height
        ) // 2

        data = data[
            ...,
            start:
            start + GRID_SIZE,
            :
        ]

        pad_height = (
            GRID_SIZE
            - data.shape[-2]
        )

    if pad_width < 0:

        start = (
            -pad_width
        ) // 2

        data = data[
            ...,
            :,
            start:
            start + GRID_SIZE
        ]

        pad_width = (
            GRID_SIZE
            - data.shape[-1]
        )

    if (
        pad_height > 0
        or pad_width > 0
    ):

        data = np.pad(
            data,
            (
                (0, 0)
                if data.ndim == 3
                else (),
            )
        )

    # Explicit edge padding.

    if data.shape[-2] < GRID_SIZE:

        amount = (
            GRID_SIZE
            - data.shape[-2]
        )

        data = np.pad(
            data,
            (
                (0, 0),
                (0, amount),
                (0, 0)
            ),
            mode="edge"
        )

    if data.shape[-1] < GRID_SIZE:

        amount = (
            GRID_SIZE
            - data.shape[-1]
        )

        data = np.pad(
            data,
            (
                (0, 0),
                (0, 0),
                (0, amount)
            ),
            mode="edge"
        )

    return data


def standardize_channel(
    data
):

    data = np.asarray(
        data,
        dtype=np.float64
    )

    valid = (
        np.isfinite(data)
        &
        (
            np.abs(data)
            < 1e6
        )
    )

    output = np.zeros_like(
        data,
        dtype=np.float64
    )

    if np.any(valid):

        valid_values = data[
            valid
        ]

        mean = np.mean(
            valid_values,
            dtype=np.float64
        )

        std = np.std(
            valid_values,
            dtype=np.float64
        )

        if (
            not np.isfinite(mean)
        ):

            mean = 0.0

        if (
            not np.isfinite(std)
            or std < 1e-8
        ):

            std = 1.0

        output[valid] = (
            data[valid]
            - mean
        ) / std

    return np.clip(
        output,
        -10.0,
        10.0
    ).astype(
        np.float32
    )


def get_gfs_3d(
    timestamp,
    latitude,
    longitude
):

    target_datetime = parse_timestamp(
        timestamp
    )

    ds = None

    try:

        ds, time_index, source_url = (
            open_gfs_dataset(
                target_datetime
            )
        )

        # Convert longitude to GFS's 0-360 system.

        longitude_360 = (
            longitude
            if longitude >= 0
            else longitude + 360
        )

        lat_values = np.asarray(
            ds["lat"].values
        )

        lon_values = np.asarray(
            ds["lon"].values
        )

        lat_index = int(
            np.argmin(
                np.abs(
                    lat_values
                    - latitude
                )
            )
        )

        lon_index = int(
            np.argmin(
                np.abs(
                    lon_values
                    - longitude_360
                )
            )
        )

        lat_start = max(
            0,
            lat_index - 40
        )

        lat_end = min(
            len(lat_values),
            lat_index + 41
        )

        lon_start = max(
            0,
            lon_index - 40
        )

        lon_end = min(
            len(lon_values),
            lon_index + 41
        )

        level_values = np.asarray(
            ds["lev"].values
        )

        level_indices = []

        for pressure_level in (
            PRESSURE_LEVELS
        ):

            index = int(
                np.argmin(
                    np.abs(
                        level_values
                        - pressure_level
                    )
                )
            )

            level_indices.append(
                index
            )

        required_variables = [
            "ugrdprs",
            "vgrdprs",
            "hgtprs"
        ]

        # Surface temperature is used as the
        # GFS SST proxy for the final channel.

        if "tmpsfc" in ds:

            surface_variable = "tmpsfc"

        elif "tmp2m" in ds:

            surface_variable = "tmp2m"

        else:

            raise RuntimeError(
                "GFS surface temperature variable "
                "was not found."
            )

        subset = ds[
            required_variables
            + [surface_variable]
        ].isel(
            time=time_index,
            lat=slice(
                lat_start,
                lat_end
            ),
            lon=slice(
                lon_start,
                lon_end
            )
        )

        u = np.asarray(
            subset[
                "ugrdprs"
            ].isel(
                lev=level_indices
            ).values,
            dtype=np.float64
        )

        v = np.asarray(
            subset[
                "vgrdprs"
            ].isel(
                lev=level_indices
            ).values,
            dtype=np.float64
        )

        z = np.asarray(
            subset[
                "hgtprs"
            ].isel(
                lev=level_indices
            ).values,
            dtype=np.float64
        )

        surface_temperature = np.asarray(
            subset[
                surface_variable
            ].values,
            dtype=np.float64
        )

        # Remove an extra singleton dimension if necessary.

        u = np.squeeze(u)
        v = np.squeeze(v)
        z = np.squeeze(z)
        surface_temperature = np.squeeze(
            surface_temperature
        )

        u = pad_to_81(u)
        v = pad_to_81(v)
        z = pad_to_81(z)

        surface_temperature = np.asarray(
            surface_temperature
        )

        # Surface field is H x W.

        if surface_temperature.ndim != 2:

            surface_temperature = np.squeeze(
                surface_temperature
            )

        if (
            surface_temperature.shape[-2:]
            != (GRID_SIZE, GRID_SIZE)
        ):

            surface_temperature = np.pad(
                surface_temperature,
                (
                    (
                        0,
                        max(
                            0,
                            GRID_SIZE
                            - surface_temperature.shape[-2]
                        )
                    ),
                    (
                        0,
                        max(
                            0,
                            GRID_SIZE
                            - surface_temperature.shape[-1]
                        )
                    )
                ),
                mode="edge"
            )

            surface_temperature = (
                surface_temperature[
                    :GRID_SIZE,
                    :GRID_SIZE
                ]
            )

        # Standardize each channel independently.
        u = standardize_channel(u)
        v = standardize_channel(v)
        z = standardize_channel(z)
        surface_temperature = (
            standardize_channel(
                surface_temperature
            )
        )

        data = np.concatenate(
            [
                u,
                v,
                z,
                surface_temperature[
                    np.newaxis,
                    :, :
                ]
            ],
            axis=0
        )

        if data.shape != (
            THREE_D_CHANNELS,
            GRID_SIZE,
            GRID_SIZE
        ):

            raise RuntimeError(
                "Constructed GFS tensor has shape "
                f"{data.shape}; expected "
                f"({THREE_D_CHANNELS}, "
                f"{GRID_SIZE}, "
                f"{GRID_SIZE})."
            )

        return (
            data.astype(np.float32),
            1.0,
            source_url
        )

    finally:

        if ds is not None:

            try:
                ds.close()

            except Exception:
                pass


# ============================================================
# BUILD ENVIRONMENT INPUT
# ============================================================

def build_environment(
    observations
):

    vectors = []

    for index in range(
        INPUT_STEPS
    ):

        vectors.append(
            build_environment_vector(
                observations,
                index
            )
        )

    env = np.asarray(
        vectors,
        dtype=np.float64
    )

    if env.shape != (
        INPUT_STEPS,
        ENV_FEATURES
    ):

        raise RuntimeError(
            f"Environment shape is "
            f"{env.shape}; expected "
            f"({INPUT_STEPS}, "
            f"{ENV_FEATURES})."
        )

    return env


# ============================================================
# BUILD 3D INPUT
# ============================================================

def build_3d_input(
    observations
):

    frames = []
    masks = []

    sources = []

    for observation in observations:

        try:

            frame, mask, source = (
                get_gfs_3d(
                    observation.timestamp,
                    observation.latitude,
                    observation.longitude
                )
            )

            frames.append(
                frame
            )

            masks.append(
                mask
            )

            sources.append(
                source
            )

        except Exception as e:

            print(
                "GFS error for",
                observation.timestamp,
                ":",
                e
            )

            # Same missing-frame convention
            # used during TCND training.

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

            masks.append(
                0.0
            )

            sources.append(
                None
            )

    return (
        np.stack(
            frames,
            axis=0
        ),
        np.asarray(
            masks,
            dtype=np.float32
        ),
        sources
    )


# ============================================================
# PREDICTION
# ============================================================

def run_prediction(
    observations
):

    # --------------------------------------------------------
    # Track
    # --------------------------------------------------------

    track = np.asarray(
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

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = build_environment(
        observations
    )

    # --------------------------------------------------------
    # GFS 3D atmosphere
    # --------------------------------------------------------

    three_d, mask, sources = (
        build_3d_input(
            observations
        )
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    try:

        track_scaled = scalers[
            "track"
        ].transform(
            track
        )

        env_scaled = scalers[
            "env"
        ].transform(
            env
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Input scaling failed: {e}"
            )
        )

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

    # --------------------------------------------------------
    # Tensors
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Neural network
    # --------------------------------------------------------

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

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Model inference failed: {e}"
            )
        )

    # --------------------------------------------------------
    # Format output
    # --------------------------------------------------------

    last_timestamp = (
        observations[-1].timestamp
    )

    predictions = []

    for i in range(
        OUTPUT_STEPS
    ):

        raw_longitude = float(
            prediction_original[
                i,
                0
            ]
        )

        raw_latitude = float(
            prediction_original[
                i,
                1
            ]
        )

        raw_pressure = float(
            prediction_original[
                i,
                2
            ]
        )

        raw_wind = float(
            prediction_original[
                i,
                3
            ]
        )

        # Physical sanity guards.

        wind_mps = max(
            0.0,
            raw_wind
        )

        pressure = max(
            0.0,
            raw_pressure
        )

        wind_knots = (
            wind_mps
            * 1.943844492
        )

        forecast_time = (
            parse_timestamp(
                last_timestamp
            )
            + timedelta(
                hours=6 * (i + 1)
            )
        ).strftime(
            "%Y%m%d%H"
        )

        predictions.append(
            {
                "timestamp": forecast_time,

                "latitude": raw_latitude,

                "longitude": raw_longitude,

                "pressure_hpa": pressure,

                "wind_mps": wind_mps,

                "wind_knots": wind_knots
            }
        )

    return {
        "last_observed_timestamp":
            last_timestamp,

        "forecast_hours": [
            6,
            12,
            18,
            24
        ],

        "input_observations": 8,

        "atmospheric_source":
            "NOAA GFS 0.25 degree",

        "predictions":
            predictions,

        "gfs_frames_available":
            int(
                np.sum(mask)
            ),

        "gfs_frames_missing":
            int(
                8 - np.sum(mask)
            )
    }


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():

    return {
        "service":
            "Cyclone Future Prediction API",

        "status":
            "running",

        "device":
            str(DEVICE),

        "model_loaded":
            model is not None,

        "input":
            "8 observations spaced 6 hours apart",

        "wind_unit":
            "m/s",

        "forecast":
            "6, 12, 18 and 24 hours"
    }


@app.get("/health")
def health():

    if model is None:

        return {
            "status": "error",
            "model_loaded": False,
            "error": startup_error
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(DEVICE)
    }


@app.post("/predict")
def predict(
    request: PredictionRequest
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                f"{startup_error or ''}"
            )
        )

    if scalers is None:

        raise HTTPException(
            status_code=503,
            detail="Scalers are not loaded."
        )

    validate_observations(
        request.observations
    )

    return run_prediction(
        request.observations
    )
