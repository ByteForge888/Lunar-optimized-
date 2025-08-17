import json
import os
import sys
import time
import matplotlib.pyplot as plt
from termcolor import colored
from pynput import keyboard
from lib.aimbot import Aimbot

class LunarAnalyzer:
    def __init__(self, box_constant=320, collect_data=False, debug=False):
        """Initialize the analyzer with Aimbot."""
        self.box_constant = box_constant
        self.collect_data = collect_data
        self.debug = debug
        self.aimbot = Aimbot(box_constant, collect_data, debug)
        self.logs = []
        self.aimbot_status = colored("ENABLED", "green")
        print("\n[INFO] PRESS 'F1' TO TOGGLE ANALYSIS\n[INFO] PRESS 'F2' TO QUIT")

    def update_status(self):
        """Toggle analysis status."""
        self.aimbot_status = colored("DISABLED", "red") if self.aimbot_status == colored("ENABLED", "green") else colored("ENABLED", "green")
        sys.stdout.write("\033[K")
        print(f"[!] ANALYSIS IS [{self.aimbot_status}]", end="\r")

    def is_enabled(self):
        """Check if analysis is enabled."""
        return self.aimbot_status == colored("ENABLED", "green")

    def plot_logs(self):
        """Visualize FPS and confidence with matplotlib."""
        if not self.logs:
            return
        times = [log["timestamp"] for log in self.logs[-100:]]
        fps = [log["fps"] for log in self.logs[-100:]]
        conf = [log["conf"] for log in self.logs[-100:]]
        plt.clf()
        plt.plot(times, fps, label="FPS", color="#1f77b4")
        plt.plot(times, conf, label="Detection Confidence", color="#ff7f0e")
        plt.xlabel("Time (s)")
        plt.ylabel("Metrics")
        plt.legend()
        plt.pause(0.01)

    def start(self):
        """Run the analysis loop."""
        print("[INFO] Starting analysis")
        while True:
            if not self.is_enabled():
                time.sleep(0.1)
                continue
            start_time = time.perf_counter()
            log_entry = self.aimbot.log_detections(start_time)
            if log_entry:
                self.logs.append(log_entry)
                self.plot_logs()
            time.sleep(0.9)  # Optimized sleep time

    def clean_up(self):
        """Close resources and exit."""
        print("\n[INFO] F2 WAS PRESSED. QUITTING...")
        self.aimbot.clean_up()
        plt.close()
        os._exit(0)

def on_release(key):
    """Handle keyboard inputs."""
    try:
        if key == keyboard.Key.f1:
            analyzer.update_status()
        if key == keyboard.Key.f2:
            analyzer.clean_up()
    except NameError:
        pass

def setup():
    """Configure sensitivity settings."""
    path = "lib/config"
    if not os.path.exists(path):
        os.makedirs(path)
    print("[INFO] In-game X and Y axis sensitivity should be the same")
    def prompt(str):
        valid_input = False
        while not valid_input:
            try:
                number = float(input(str))
                valid_input = True
            except ValueError:
                print("[!] Invalid Input. Enter only the number (e.g., 6.9)")
        return number
    xy_sens = prompt("X-Axis and Y-Axis Sensitivity: ")
    targeting_sens = prompt("Targeting Sensitivity: ")
    sensitivity_settings = {
        "xy_sens": xy_sens,
        "targeting_sens": targeting_sens,
        "xy_scale": 10 / xy_sens,
        "targeting_scale": 1000 / (targeting_sens * xy_sens)
    }
    with open('lib/config/config.json', 'w') as outfile:
        json.dump(sensitivity_settings, outfile)
    print("[INFO] Sensitivity configuration complete")

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    print(colored('''
    | |
    | |    _   _ _ __   __ _ _ __
    | |   | | | | '_ \ / _` | '__|
    | |___| |_| | | | | (_| | |
    \_____/\__,_|_| |_|\__,_|_|
    (Neural Network Analyzer)''', "yellow"))

    path_exists = os.path.exists("lib/config/config.json")
    if not path_exists or ("setup" in sys.argv):
        if not path_exists:
            print("[!] Sensitivity configuration is not set")
        setup()
    if not os.path.exists("lib/data"):
        os.makedirs("lib/data")
    analyzer = LunarAnalyzer(collect_data="collect_data" in sys.argv)
    listener = keyboard.Listener(on_release=on_release)
    listener.start()
    analyzer.start()