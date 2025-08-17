import json
import uuid
import numpy as np
import torch
import mss
import cv2
from termcolor import colored

class Aimbot:
    def __init__(self, box_constant=320, collect_data=False, debug=False):
        """Initialize YOLOv5s for player detection."""
        self.box_constant = box_constant
        self.collect_data = collect_data
        self.debug = debug
        self.screen = mss.mss()
        with open("lib/config/config.json") as f:
            self.sens_config = json.load(f)

        # Load YOLOv5s with FP16
        print("[INFO] Loading YOLOv5s model")
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='lib/best.pt').half()
            if torch.cuda.is_available():
                self.model.cuda()
                print(colored("CUDA ACCELERATION [ENABLED]", "green"))
            else:
                print(colored("[!] CUDA ACCELERATION UNAVAILABLE", "red"))
            self.model.conf = 0.45
            self.model.iou = 0.45
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise

    def log_detections(self, start_time):
        """Capture screen and log detection data."""
        detection_box = {
            'left': int(960 - self.box_constant // 2),
            'top': int(540 - self.box_constant // 2),
            'width': self.box_constant,
            'height': self.box_constant
        }
        try:
            frame = np.array(self.screen.grab(detection_box))
        except Exception as e:
            print(f"[ERROR] MSS capture failed: {e}")
            return None
        try:
            results = self.model(frame)
        except Exception as e:
            print(f"[ERROR] YOLOv5 inference failed: {e}")
            return None

        fps = int(1 / (time.perf_counter() - start_time))
        conf = results.xyxy[0][0][4].item() if len(results.xyxy[0]) > 0 else 0.0
        log_entry = {"timestamp": time.perf_counter(), "fps": fps, "conf": conf}

        if self.collect_data and len(results.xyxy[0]) > 0 and conf > 0.6:
            with open("lib/data/detections.json", "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
            frame_filename = f"lib/data/{str(uuid.uuid4())}.jpg"
            cv2.imwrite(frame_filename, frame)
            if self.debug:
                print(f"[DEBUG] Saved frame: {frame_filename}")
        return log_entry

    def clean_up(self):
        """Close MSS resources."""
        self.screen.close()