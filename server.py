from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import time
import os
import threading

from geopy.distance import geodesic

from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


app = FastAPI()

# ---------------- MODEL ----------------
model = joblib.load("driver_behavior_model_without_speed.pkl")

# ---------------- DATABASE ----------------
DATABASE_URL = os.getenv("postgresql://trip_details_user:IUHbaRAjgGEON0mgdfjiWDRjbYfxBktj@dpg-d75st094tr6s73ce0hd0-a/trip_details")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Trip(Base):
    __tablename__ = "trips"

    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(String)
    end_time = Column(String)
    distance = Column(Float)


Base.metadata.create_all(bind=engine)


# ---------------- INPUT ----------------
class SensorData(BaseModel):
    ax: float
    ay: float
    az: float
    speed: float
    lat: float
    lon: float


# ---------------- GLOBAL STATE ----------------
trip_active = False
start_time = ""
total_distance = 0
last_location = None
last_received_time = None

INACTIVITY_TIMEOUT = 10


# ---------------- LIVE DATA ----------------
latest_data = {
    "lat": 13.0827,
    "lon": 80.2707,
    "ax": 0,
    "ay": 0,
    "az": 0,
    "speed": 0,
    "behavior": "NORMAL",
    "time": "",
    "distance": 0,
    "trip_active": False,
    "trip_start": "",
    "trip_end": ""
}


# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(data: SensorData):

    global trip_active, start_time, total_distance
    global last_location, last_received_time

    db = SessionLocal()

    ax, ay, az = data.ax, data.ay, data.az
    speed, lat, lon = data.speed, data.lat, data.lon

    current_time_str = time.strftime("%H:%M:%S")
    current_timestamp = time.time()

    current_location = (lat, lon)

    # 🚀 START TRIP
    if not trip_active:
        trip_active = True
        start_time = current_time_str
        total_distance = 0
        last_location = current_location

        latest_data["trip_start"] = start_time
        latest_data["trip_end"] = ""

        print("Trip Started")

    # 🚗 DISTANCE
    if last_location is not None:
        dist = geodesic(last_location, current_location).meters
        total_distance += dist
        last_location = current_location

    # 🧠 ML
    acc_mag = (ax**2 + ay**2 + az**2) ** 0.5
    X = np.array([[ax, ay, az, acc_mag]])
    prediction = model.predict(X)[0]

    # ⏱ UPDATE TIME
    last_received_time = current_timestamp

    latest_data.update({
        "lat": lat,
        "lon": lon,
        "ax": ax,
        "ay": ay,
        "az": az,
        "speed": speed,
        "behavior": prediction,
        "time": current_time_str,
        "distance": total_distance,
        "trip_active": trip_active
    })

    return {
        "driver_behavior": prediction,
        "trip_active": trip_active,
        "distance": total_distance
    }


# ---------------- AUTO STOP ----------------
def monitor_inactivity():
    global trip_active, last_received_time, total_distance, start_time

    while True:
        if trip_active and last_received_time is not None:
            if time.time() - last_received_time > INACTIVITY_TIMEOUT:

                print("Trip Ended")

                db = SessionLocal()

                end_time = time.strftime("%H:%M:%S")

                # SAVE FINAL DISTANCE BEFORE RESET
                latest_data["distance"] = total_distance
                latest_data["trip_end"] = end_time

                trip = Trip(
                    start_time=start_time,
                    end_time=end_time,
                    distance=total_distance
                )

                db.add(trip)
                db.commit()

                # RESET
                trip_active = False
                total_distance = 0
                last_received_time = None

        time.sleep(2)


threading.Thread(target=monitor_inactivity, daemon=True).start()


# ---------------- API ----------------
@app.get("/data")
def get_data():
    return latest_data


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open("dashboard.html", encoding="utf-8") as f:
        return f.read()