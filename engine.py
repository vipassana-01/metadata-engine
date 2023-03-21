import cv2
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import json

class Engine:
    def __init__(self, video_path, metadata_dir):
        self.video_path = video_path
        self.metadata_dir = metadata_dir
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.vidcap = cv2.VideoCapture(self.video_path)
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    def detect_people(self):
        for i in tqdm(range(self.num_frames)):
            success, image = self.vidcap.read()
            if not success:
                break
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results = self.model(image, size=640)
            boxes = results.xyxy[0][:, :4].cpu().numpy().tolist()
            metadata = {
                "frame_number": i,
                "boxes": boxes,
                "num_object": len(boxes)
            }
            metadata_file = self.metadata_dir[:-5] + f"_frame_{i}.json"
            with open(metadata_file, 'w') as outfile:
                json.dump(metadata, outfile)

    def run(self):
        self.detect_people()

video_paths = ["BB_d2123119-032a-4ef6-856c-71e050948b71_preview.mp4", "BB_4bac8cdd-152c-402b-a78b-3c8b1675d8f0_preview.mp4", "BB_49c37660-d213-41e5-9ff5-c6ba88b98466_preview.mp4"]
metadata_dir = r"C:\Users\kinsh\Downloads"

for video_path in video_paths:
    engine = Engine(video_path, metadata_dir)
    engine.run()
