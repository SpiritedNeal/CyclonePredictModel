import os
from datetime import datetime, timedelta
import pickle

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "cyclone_multimodal.pth")
SCALER_PATH = os.getenv("SCALER_PATH", "cyclone_scalers.pkl")

INPUT_STEPS = 8
OUTPUT_STEPS = 4
TRACK_FEATURES = 4
ENV_FEATURES = 96
THREE_D_CHANNELS = 13
GRID_FEATURES = 64
MASK_FEATURES = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.fc = nn.Linear(128, GRID_FEATURES)

    def forward(self, x):
        x = self.network(x)
        return self.fc(x.flatten(start_dim=1))


class CycloneModel(nn.Module):
    def __init__(self):
        super().__init__()

        lstm_input = (
            TRACK_FEATURES + ENV_FEATURES + GRID_FEATURES + MASK_FEATURES
        )

        self.grid_encoder = GridEncoder()

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
            nn.Linear(128, OUTPUT_STEPS * 4),
        )

    def forward(self, track, env, three_d, mask):
        batch_size = track.shape[0]
        time_steps = track.shape[1]

        three_d_flat = three_d.reshape(
            batch_size * time_steps,
            THREE_D_CHANNELS,
            81,
            81,
        )

        grid_features = self.grid_encoder(three_d_flat)
        grid_features = grid_features.reshape(
            batch_size, time_steps, GRID_FEATURES
        )

        mask = mask.unsqueeze(-1)

        combined = torch.cat(
            [track, env, grid_features, mask], dim=-1
        )

        lstm_output, _ = self.lstm(combined)
        last_output = lstm_output[:, -1, :]

        output = self.decoder(last_output)

        return output.reshape(batch_size, OUTPUT_STEPS, 4)


app = FastAPI(
    title="Cyclone Multimodal Prediction API",
    version="1.0.0",
    description=(
        "Predicts the next 24 hours of cyclone longitude, latitude, "
        "pressure and wind from 8 historical observations plus "
        "environmental and 3D atmospheric inputs."
    ),
)

model = None
scalers = None
startup_error = None


@app.on_event("startup")
def load_artifacts():
    global model, scalers, startup_error

    try:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        if not os.path.isfile(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

        state_dict = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

        model = CycloneModel().to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        with open(SCALER_PATH, "rb") as f:
            scalers = pickle.load(f)

        startup_error = None
        print("=" * 60)
        print("Cyclone API started")
        print(f"Device: {DEVICE}")
        print(f"Model: {MODEL_PATH}")
        print(f"Scalers: {SCALER_PATH}")
        print("=" * 60)

    except Exception as e:
        model = None
        scalers = None
        startup_error = str(e)
        print("STARTUP ERROR:", e)


class PredictionRequest(BaseModel):
    track: list[list[float]] = Field(
        ...,
        description="8 rows of [longitude, latitude, pressure, wind].",
    )
    env: list[list[float]] = Field(
        ...,
        description="8 rows, each containing 96 environment features.",
    )
    three_d: list = Field(
        ...,
        description="8 x 13 x 81 x 81 3D atmospheric tensor.",
    )
    mask: list[float] = Field(
        ...,
        description="8 values; 1 when a 3D frame exists, otherwise 0.",
    )
    last_timestamp: str = Field(
        ...,
        description="Last historical timestamp in YYYYMMDDHH format.",
    )


def validate_array(name, value, expected_shape):
    arr = np.asarray(value, dtype=np.float32)

    if arr.shape != expected_shape:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{name} has shape {arr.shape}; "
                f"expected {expected_shape}."
            ),
        )

    if not np.isfinite(arr).all():
        raise HTTPException(
            status_code=400,
            detail=f"{name} contains NaN or infinity.",
        )

    return arr


def future_timestamp(timestamp, hours):
    try:
        dt = datetime.strptime(str(timestamp), "%Y%m%d%H")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="last_timestamp must use YYYYMMDDHH format.",
        )

    return (dt + timedelta(hours=hours)).strftime("%Y%m%d%H")


@app.get("/")
def root():
    return {
        "service": "Cyclone Multimodal Prediction API",
        "status": "running",
        "device": str(DEVICE),
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    if model is None:
        return {
            "status": "error",
            "model_loaded": False,
            "error": startup_error,
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(DEVICE),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or scalers is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded. {startup_error or ''}",
        )

    track = validate_array(
        "track",
        request.track,
        (INPUT_STEPS, TRACK_FEATURES),
    )

    env = validate_array(
        "env",
        request.env,
        (INPUT_STEPS, ENV_FEATURES),
    )

    three_d = validate_array(
        "three_d",
        request.three_d,
        (INPUT_STEPS, THREE_D_CHANNELS, 81, 81),
    )

    mask = validate_array(
        "mask",
        request.mask,
        (INPUT_STEPS,),
    )

    try:
        track_scaled = scalers["track"].transform(
            track.astype(np.float64)
        )
        env_scaled = scalers["env"].transform(
            env.astype(np.float64)
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Scaling failed: {str(e)}",
        )

    track_scaled = np.nan_to_num(
        track_scaled, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    env_scaled = np.nan_to_num(
        env_scaled, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    track_tensor = torch.from_numpy(track_scaled).unsqueeze(0).to(DEVICE)
    env_tensor = torch.from_numpy(env_scaled).unsqueeze(0).to(DEVICE)
    three_d_tensor = torch.from_numpy(three_d).unsqueeze(0).to(DEVICE)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)

    try:
        with torch.no_grad():
            prediction = model(
                track_tensor,
                env_tensor,
                three_d_tensor,
                mask_tensor,
            )

        prediction = prediction.cpu().numpy()[0]

        prediction_original = scalers["target"].inverse_transform(
            prediction.reshape(-1, 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}",
        )

    predictions = []

    for i in range(OUTPUT_STEPS):
        predictions.append(
            {
                "timestamp": future_timestamp(
                    request.last_timestamp,
                    6 * (i + 1),
                ),
                "latitude": float(prediction_original[i, 1]),
                "longitude": float(prediction_original[i, 0]),
                "pressure": float(prediction_original[i, 2]),
                "wind": float(prediction_original[i, 3]),
            }
        )

    return {
        "last_observed_timestamp": request.last_timestamp,
        "forecast_hours": [6, 12, 18, 24],
        "predictions": predictions,
    }
