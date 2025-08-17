import json
import uuid
import time
import logging
import numpy as np
import torch
import mss
import cv2
import pyautogui
import keyboard
import pytesseract
from termcolor import colored
from datetime import datetime
import imageio

# Configure logging for traceability and fun output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class Aimbot:
    def __init__(self, box_constant=320, collect_data=False, debug=False, conf_threshold=0.45, iou_threshold=0.45, aim_fov=50, aim_smooth=0.5):
        """
        Initialize Aimbot with YOLOv5s for player detection and enhanced utilities.

        Args:
            box_constant (int): Detection box size (default: 320).
            collect_data (bool): Enable data collection for logs and frames.
            debug (bool): Enable debug mode with verbose output.
            conf_threshold (float): Confidence threshold for detections (default: 0.45).
            iou_threshold (float): IoU threshold for non-max suppression (default: 0.45).
            aim_fov (int): Field of view for auto-aim in pixels (default: 50).
            aim_smooth (float): Smoothing factor for aim movement (0-1, lower is smoother).
        """
        self.box_constant = box_constant
        self.collect_data = collect_data
        self.debug = debug
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.aim_fov = aim_fov
        self.aim_smooth = max(0.1, min(1.0, aim_smooth))
        self.screen = mss.mss()
        self.detections = []
        self.auto_aim = False
        self.trigger_bot = False
        self.trigger_delay = 0.05
        self.crosshair_enabled = False
        self.visuals_config = {"color": "red", "thickness": 2, "theme": "default", "show_labels": True}
        self.logs = []
        self.hotkeys = {"toggle_aimbot": "f3", "toggle_trigger": "f4", "snapshot": "f5"}  # Customizable hotkeys
        self.recording = False
        self.record_frames = []
        self.fov_scale = 1.0  # Dynamic FOV scaling
        pyautogui.FAILSAFE = True

        # Load sensitivity configuration
        self.config_path = "lib/config/config.json"
        try:
            with open(self.config_path, 'r') as f:
                self.sens_config = json.load(f)
            logger.info(colored("Sensitivity config loaded! 🎮", "cyan"))
        except Exception as e:
            logger.error(colored(f"Config load failed: {e}", "red"))
            raise

        # Load YOLOv5s model
        logger.info(colored("Loading YOLOv5s model... ready for action! 💥", "yellow"))
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='lib/best.pt').half()
            if torch.cuda.is_available():
                self.model.cuda()
                logger.info(colored("CUDA ACCELERATION [ENABLED] 🚀", "green"))
            else:
                logger.info(colored("CUDA ACCELERATION [UNAVAILABLE] 😢", "red"))
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
        except Exception as e:
            logger.error(colored(f"YOLOv5 load failed: {e}", "red"))
            raise

        # Get screen resolution
        self.screen_width, self.screen_height = pyautogui.size()
        logger.info(colored(f"Detected resolution: {self.screen_width}x{self.screen_height}", "cyan"))

    def update_config(self, xy_sens=None, targeting_sens=None, conf_threshold=None, iou_threshold=None, aim_fov=None, aim_smooth=None):
        """Update sensitivity, detection, or aim settings dynamically."""
        if xy_sens and targeting_sens:
            self.sens_config = {
                "xy_sens": xy_sens,
                "targeting_sens": targeting_sens,
                "xy_scale": 10 / xy_sens,
                "targeting_scale": 1000 / (targeting_sens * xy_sens)
            }
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(self.sens_config, f, indent=4)
                logger.info(colored("Sensitivity config updated! 🔧", "green"))
            except Exception as e:
                logger.error(colored(f"Config save failed: {e}", "red"))

        if conf_threshold:
            self.conf_threshold = conf_threshold
            self.model.conf = conf_threshold
        if iou_threshold:
            self.iou_threshold = iou_threshold
            self.model.iou = iou_threshold
        if aim_fov:
            self.aim_fov = aim_fov
        if aim_smooth:
            self.aim_smooth = max(0.1, min(1.0, aim_smooth))
        logger.info(colored(f"Settings updated: FOV={self.aim_fov}, Smooth={self.aim_smooth}", "cyan"))

    def set_hotkeys(self, toggle_aimbot=None, toggle_trigger=None, snapshot=None):
        """Set custom hotkeys for features."""
        if toggle_aimbot:
            self.hotkeys["toggle_aimbot"] = toggle_aimbot
        if toggle_trigger:
            self.hotkeys["toggle_trigger"] = toggle_trigger
        if snapshot:
            self.hotkeys["snapshot"] = snapshot
        logger.info(colored(f"Hotkeys updated: {self.hotkeys}", "cyan"))

    def toggle_auto_aim(self):
        """Toggle auto-aim functionality."""
        self.auto_aim = not self.auto_aim
        logger.info(colored(f"Auto-aim {'enabled' if self.auto_aim else 'disabled'}", "magenta"))

    def toggle_trigger_bot(self):
        """Toggle trigger bot functionality."""
        self.trigger_bot = not self.trigger_bot
        logger.info(colored(f"Trigger bot {'enabled' if self.trigger_bot else 'disabled'}", "magenta"))

    def toggle_crosshair(self):
        """Toggle custom crosshair overlay."""
        self.crosshair_enabled = not self.crosshair_enabled
        logger.info(colored(f"Crosshair {'enabled' if self.crosshair_enabled else 'disabled'}", "cyan"))

    def set_visuals_config(self, color="red", thickness=2, theme="default", show_labels=True):
        """Set configuration for ESP visuals."""
        theme_colors = {
            "default": color,
            "neon": "#00FFFF",
            "retro": "#FF00FF"
        }
        self.visuals_config = {
            "color": theme_colors.get(theme, color),
            "thickness": thickness,
            "theme": theme,
            "show_labels": show_labels
        }
        logger.info(colored(f"Visuals updated: color={self.visuals_config['color']}, theme={theme}", "yellow"))

    def adjust_fov(self, game_fov):
        """
        Adjust detection box size based on in-game FOV.

        Args:
            game_fov (float): In-game field of view in degrees.
        """
        # Heuristic: Scale box size inversely with FOV
        base_fov = 90  # Reference FOV
        self.fov_scale = base_fov / max(1, game_fov)
        self.box_constant = int(320 * self.fov_scale)
        logger.info(colored(f"FOV adjusted: scale={self.fov_scale:.2f}, box_constant={self.box_constant}", "cyan"))

    def start_recording(self, duration=5):
        """
        Start recording a gameplay clip for a specified duration.

        Args:
            duration (float): Recording duration in seconds.
        """
        self.recording = True
        self.record_frames = []
        logger.info(colored(f"Recording started for {duration}s! 🎥", "yellow"))
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < duration and self.recording:
            detection_box = {
                'left': int(self.screen_width / 2 - self.box_constant // 2),
                'top': int(self.screen_height / 2 - self.box_constant // 2),
                'width': self.box_constant,
                'height': self.box_constant
            }
            try:
                frame = np.array(self.screen.grab(detection_box))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                if len(self.detections) > 0:
                    frame = self.draw_bounding_boxes(frame, self.detections)
                self.record_frames.append(frame)
            except Exception as e:
                logger.error(colored(f"Recording frame error: {e}", "red"))
            time.sleep(0.03)  # ~30 FPS
        self.save_recording()

    def save_recording(self):
        """Save recorded frames as a video file."""
        if not self.record_frames:
            logger.warning(colored("No frames to save for recording!", "yellow"))
            return
        try:
            filename = f"lib/data/clip_{str(uuid.uuid4())}.mp4"
            writer = imageio.get_writer(filename, fps=30, codec='libx264')
            for frame in self.record_frames:
                writer.append_data(frame)
            writer.close()
            logger.info(colored(f"Video saved: {filename} 🎬", "green"))
        except Exception as e:
            logger.error(colored(f"Failed to save video: {e}", "red"))
        self.recording = False
        self.record_frames = []

    def log_game_state(self, region=None):
        """
        Log game state (e.g., health, ammo) using OCR on a specified screen region.

        Args:
            region (tuple): (x, y, w, h) region to capture for OCR.
        """
        if region is None:
            region = (50, 50, 200, 100)  # Default HUD region
        try:
            x, y, w, h = region
            frame = np.array(self.screen.grab({'left': x, 'top': y, 'width': w, 'height': h}))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            text = pytesseract.image_to_string(frame)
            log_entry = {"timestamp": time.perf_counter(), "game_state": text.strip()}
            self.logs.append(log_entry)
            logger.info(colored(f"Game state logged: {text.strip()}", "cyan"))
            return log_entry
        except Exception as e:
            logger.error(colored(f"Game state logging failed: {e}", "red"))
            return None

    def optimize_performance(self):
        """Adjust detection settings based on system performance."""
        avg_fps = np.mean([log["fps"] for log in self.logs[-10:]] if self.logs else [60])
        if avg_fps < 30:
            self.conf_threshold = min(0.6, self.conf_threshold + 0.05)
            self.box_constant = int(self.box_constant * 0.9)
            logger.info(colored(f"Optimizing: conf={self.conf_threshold:.2f}, box={self.box_constant}", "yellow"))
        elif avg_fps > 60:
            self.conf_threshold = max(0.3, self.conf_threshold - 0.05)
            self.box_constant = int(self.box_constant * 1.1)
            logger.info(colored(f"Optimizing: conf={self.conf_threshold:.2f}, box={self.box_constant}", "yellow"))
        self.model.conf = self.conf_threshold

    def log_detections(self, start_time):
        """
        Capture screen, perform detection, and log data.

        Returns:
            dict: Log entry with timestamp, FPS, confidence, and detection count.
        """
        detection_box = {
            'left': int(self.screen_width / 2 - self.box_constant // 2),
            'top': int(self.screen_height / 2 - self.box_constant // 2),
            'width': self.box_constant,
            'height': self.box_constant
        }

        try:
            frame = np.array(self.screen.grab(detection_box))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            results = self.model(frame)
            self.detections = results.xyxy[0].cpu().numpy()
        except Exception as e:
            logger.error(colored(f"Detection failed: {e}", "red"))
            return None

        current_time = time.perf_counter()
        fps = int(1 / (current_time - start_time)) if current_time > start_time else 0
        conf = self.detections[0][4].item() if len(self.detections) > 0 else 0.0
        det_count = len(self.detections)
        log_entry = {"timestamp": current_time, "fps": fps, "conf": conf, "det_count": det_count}
        self.logs.append(log_entry)

        if self.collect_data and det_count > 0 and conf > 0.6:
            try:
                with open("lib/data/detections.json", "a") as f:
                    json.dump(log_entry, f)
                    f.write("\n")
                frame_filename = f"lib/data/{str(uuid.uuid4())}.jpg"
                if self.debug:
                    annotated_frame = self.draw_bounding_boxes(frame, self.detections)
                    cv2.imwrite(frame_filename, annotated_frame)
                    logger.info(colored(f"Saved annotated frame: {frame_filename}", "green"))
                else:
                    cv2.imwrite(frame_filename, frame)
                    logger.info(colored(f"Saved frame: {frame_filename}", "green"))
            except Exception as e:
                logger.error(colored(f"Data save failed: {e}", "red"))

        if self.auto_aim and det_count > 0:
            self.perform_auto_aim(self.detections)
        if self.trigger_bot and det_count > 0:
            self.perform_trigger_bot(self.detections)
        self.optimize_performance()  # Adjust settings dynamically

        return log_entry

    def draw_bounding_boxes(self, frame, detections):
        """Draw bounding boxes and labels on the frame."""
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2, conf, _ = det
            if conf > self.conf_threshold:
                color = self.visuals_config["color"]
                if self.visuals_config["theme"] == "neon":
                    color = (0, 255, 255)
                elif self.visuals_config["theme"] == "retro":
                    color = (255, 0, 255)
                else:
                    color = (0, 255, 0) if color == "red" else color
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, self.visuals_config["thickness"])
                if self.visuals_config["show_labels"]:
                    label = f"Conf: {conf:.2f}"
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated_frame

    def get_detections(self):
        """Return detections for GUI rendering (ESP)."""
        screen_detections = []
        for det in self.detections:
            x1, y1, x2, y2, conf, _ = det
            screen_x = x1 + (self.screen_width / 2 - self.box_constant / 2)
            screen_y = y1 + (self.screen_height / 2 - self.box_constant / 2)
            screen_detections.append([screen_x, screen_y, x2 - x1, y2 - y1, conf])
        return screen_detections

    def draw_crosshair(self):
        """Return crosshair coordinates for GUI overlay."""
        if not self.crosshair_enabled:
            return None
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        size = 10
        color = self.visuals_config["color"]
        return (center_x, center_y, size, color)

    def perform_auto_aim(self, detections):
        """Move mouse to the center of the highest-confidence detection."""
        if len(detections) == 0:
            return
        best_det = max(detections, key=lambda x: x[4])
        x1, y1, x2, y2, conf, _ = best_det
        if conf < self.conf_threshold:
            return

        center_x = x1 + (x2 - x1) / 2
        center_y = y1 + (y2 - y1) / 2
        screen_x = center_x + (self.screen_width / 2 - self.box_constant / 2)
        screen_y = center_y + (self.screen_height / 2 - self.box_constant / 2)

        current_x, current_y = pyautogui.position()
        if abs(screen_x - current_x) > self.aim_fov or abs(screen_y - current_y) > self.aim_fov:
            return

        delta_x = (screen_x - current_x) * self.aim_smooth
        delta_y = (screen_y - current_y) * self.aim_smooth
        scaled_x = delta_x * self.sens_config["xy_scale"]
        scaled_y = delta_y * self.sens_config["xy_scale"]
        pyautogui.moveRel(scaled_x, scaled_y, duration=0.01)
        logger.debug(colored(f"Aiming at ({screen_x:.1f}, {screen_y:.1f})", "blue"))

    def perform_trigger_bot(self, detections):
        """Fire when a target is near the center."""
        for det in detections:
            x1, y1, x2, y2, conf, _ = det
            if conf < self.conf_threshold:
                continue
            center_x = x1 + (x2 - x1) / 2
            if abs(center_x - self.box_constant / 2) < 10:
                pyautogui.click()
                time.sleep(self.trigger_delay)
                logger.debug(colored("Trigger bot fired! 💥", "red"))
                break

    def auto_calibrate_sensitivity(self, test_frames=100, target_conf=0.7):
        """Automatically calibrate sensitivity based on detection stability."""
        logger.info(colored("Starting auto-calibration 🎯", "yellow"))
        total_conf = 0.0
        frame_count = 0
        for _ in range(test_frames):
            detection_box = {
                'left': int(self.screen_width / 2 - self.box_constant // 2),
                'top': int(self.screen_height / 2 - self.box_constant // 2),
                'width': self.box_constant,
                'height': self.box_constant
            }
            try:
                frame = np.array(self.screen.grab(detection_box))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                results = self.model(frame)
                if len(results.xyxy[0]) > 0:
                    total_conf += results.xyxy[0][0][4].item()
                    frame_count += 1
            except Exception as e:
                logger.error(colored(f"Calibration error: {e}", "red"))
            time.sleep(0.05)
        
        if frame_count > 0:
            avg_conf = total_conf / frame_count
            adjustment = target_conf / avg_conf
            new_xy_sens = self.sens_config["xy_sens"] * adjustment
            new_targeting_sens = self.sens_config["targeting_sens"] * adjustment
            self.update_config(xy_sens=new_xy_sens, targeting_sens=new_targeting_sens)
            logger.info(colored(f"Calibrated: xy_sens={new_xy_sens:.2f}, targeting_sens={new_targeting_sens:.2f}", "green"))
        else:
            logger.warning(colored("Calibration failed: no detections 😕", "yellow"))

    def capture_snapshot(self, annotate=True):
        """Capture a screenshot with optional annotations."""
        detection_box = {
            'left': int(self.screen_width / 2 - self.box_constant // 2),
            'top': int(self.screen_height / 2 - self.box_constant // 2),
            'width': self.box_constant,
            'height': self.box_constant
        }
        try:
            frame = np.array(self.screen.grab(detection_box))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            filename = f"lib/data/snapshot_{str(uuid.uuid4())}.jpg"
            if annotate and len(self.detections) > 0:
                frame = self.draw_bounding_boxes(frame, self.detections)
            cv2.imwrite(filename, frame)
            logger.info(colored(f"Snapshot saved: {filename} 📸", "cyan"))
        except Exception as e:
            logger.error(colored(f"Snapshot failed: {e}", "red"))

    def export_detections(self, output_dir="lib/export"):
        """Export all collected detections and frames with metadata."""
        os.makedirs(output_dir, exist_ok=True)
        try:
            metadata = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detections": self.logs,
                "config": self.sens_config,
                "visuals": self.visuals_config,
                "hotkeys": self.hotkeys
            }
            with open(os.path.join(output_dir, "exported_detections.json"), "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(colored(f"Detections exported to {output_dir} 🎉", "green"))
        except Exception as e:
            logger.error(colored(f"Export failed: {e}", "red"))

    def clean_up(self):
        """Close resources and save logs if needed."""
        logger.info(colored("Cleaning up Aimbot... Peace out! ✌️", "yellow"))
        if self.recording:
            self.save_recording()
        self.screen.close()
        if self.collect_data:
            self.export_detections()