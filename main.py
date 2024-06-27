from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
import base64
import json
from typing import List

class Location:
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude
        
class Detected:
    def __init__(self, image, location):
        self.image = image
        self.location = location

class Output:
    detected: List[Detected]

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "*"
]
 
path = os.path.join("fake-aws")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"])

def load_model(task_name):
    if task_name == "road":
        return YOLO(os.path.join("models", "road_inspection.pt"))
    elif task_name == "building":
        return YOLO(os.path.join("models", "building_cracks.pt"))
    elif task_name == "field":
        return YOLO(os.path.join("models", "nest_fawn_detector.pt"))

def get_predictions(model, task_name, task_id):
    results = model(os.path.join(path, str(task_id), "images"))
    taskid = str(task_name) + str(task_id)
    os.makedirs(taskid, exist_ok=True)

    for i, result in enumerate(results):
        filepath = os.path.join(taskid, os.path.basename(result.path))
        img = result.save(filename=filepath)
        
    return return_predictions(task_name, task_id)

def return_predictions(task_name, task_id):
    image_dir = str(task_name) + str(task_id)
    response = []
    jsons = []

    for files in os.listdir(os.path.join(path, str(task_id), "jsons")):
        file_path = os.path.join(path, str(task_id), "jsons", files)

        with open(file_path) as f:
            data = json.load(f)
            jsons.append(data)
            
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        if os.path.isfile(image_path):
            with open(image_path, "rb") as f:
                image_data = f.read()
                encoded_data = base64.b64encode(image_data).decode("utf-8")
                for j in jsons:
                    file = j["path"]
                    long = j["long"]
                    lat = j["lat"]
                    location = Location(long, lat)
                    if filename == file:
                        response.append(Detected(encoded_data, location))
                
    return response

@app.get("/task/{task_name}/{task_id}")
async def root(task_name: str, task_id: int):
    if os.path.exists(str(task_name) + str(task_id)):
        return return_predictions(task_name, task_id)
    else:
        model = load_model(task_name)
        return get_predictions(model, task_name, task_id)