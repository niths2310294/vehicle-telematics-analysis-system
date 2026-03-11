from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import time

app = FastAPI()

model = joblib.load("driver_behavior_model_without_speed.pkl")

class SensorData(BaseModel):
    ax: float
    ay: float
    az: float
    speed: float
    lat: float
    lon: float


latest_data = {
    "lat": 13.0827,
    "lon": 80.2707,
    "ax": 0,
    "ay": 0,
    "az": 0,
    "speed": 0,
    "behavior": "NORMAL",
    "time": ""
}


@app.post("/predict")
def predict(data: SensorData):

    ax = data.ax
    ay = data.ay
    az = data.az
    speed = data.speed
    lat = data.lat
    lon = data.lon

    acc_mag = (ax**2 + ay**2 + az**2) ** 0.5

    # MODEL INPUT (WITHOUT SPEED)
    X = np.array([[ax, ay, az, acc_mag]])

    prediction = model.predict(X)[0]

    latest_data.update({
        "lat": lat,
        "lon": lon,
        "ax": ax,
        "ay": ay,
        "az": az,
        "speed": speed,
        "behavior": prediction,
        "time": time.strftime("%H:%M:%S")
    })

    return {"driver_behavior": prediction}


@app.get("/data")
def get_data():
    return latest_data


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open("dashboard.html", encoding="utf-8") as f:
        return f.read()