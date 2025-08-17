import json
import os
import sys
import time
import logging
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from termcolor import colored
from pynput import keyboard
from lib.aimbot import Aimbot  # Assuming the custom Aimbot module is available
import psutil  # For performance monitoring

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class LunarAnalyzerGUI:
    def __init__(self, box_constant=320, collect_data=False, debug=False, plot_interval=100):
        """
        Initialize LunarAnalyzer with enhanced GUI for single-player modding.

        Args:
            box_constant (int): Detection box size (default: 320).
            collect_data (bool): Enable data collection.
            debug (bool): Enable debug mode.
            plot_interval (int): Number of logs to plot (default: 100).
        """
        self.box_constant = box_constant
        self.collect_data = collect_data
        self.debug = debug
        self.plot_interval = plot_interval
        self.aimbot = Aimbot(box_constant, collect_data, debug)
        self.logs = []
        self.aimbot_status = "ENABLED"
        self.running = True
        self.visuals_enabled = False

        # Setup Tkinter GUI
        self.root = tk.Tk()
        self.root.title("LunarAnalyzer - Single-Player Modding Suite")
        self.root.geometry("500x600")
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.95)
        self.root.configure(bg='#2e2e2e')  # Dark theme

        # Style for GUI
        style = ttk.Style()
        style.configure("TNotebook", background="#2e2e2e", foreground="white")
        style.configure("TButton", padding=5, font=("Arial", 10))
        style.configure("TLabel", background="#2e2e2e", foreground="white", font=("Arial", 10))

        # Create Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tabs
        self.aimbot_tab = ttk.Frame(self.notebook)
        self.visuals_tab = ttk.Frame(self.notebook)
        self.utilities_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.aimbot_tab, text='Aimbot')
        self.notebook.add(self.visuals_tab, text='Visuals')
        self.notebook.add(self.utilities_tab, text='Utilities')

        # Setup tabs
        self.setup_aimbot_tab()
        self.setup_visuals_tab()
        self.setup_utilities_tab()

        # Overlay for ESP and crosshair
        self.overlay = None
        self.create_overlay()

        logger.info(colored("PRESS 'F1' TO TOGGLE ANALYSIS", "cyan"))
        logger.info(colored("PRESS 'F2' TO QUIT", "cyan"))

    def setup_aimbot_tab(self):
        """Setup enhanced Aimbot tab with toggles and sliders."""
        frame = tk.Frame(self.aimbot_tab, bg='#2e2e2e')
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Status
        tk.Label(frame, text="Aimbot Status:", bg='#2e2e2e', fg='white', font=("Arial", 12)).pack(pady=5)
        self.status_var = tk.StringVar(value=self.aimbot_status)
        status_display = tk.Label(frame, textvariable=self.status_var, fg="green", bg='#2e2e2e', font=("Arial", 12, "bold"))
        status_display.pack()

        # Toggles
        tk.Button(frame, text="Toggle Aimbot", command=self.update_status).pack(pady=5)
        tk.Button(frame, text="Toggle Auto-Aim", command=self.aimbot.toggle_auto_aim).pack(pady=5)
        tk.Button(frame, text="Toggle Trigger Bot", command=self.aimbot.toggle_trigger_bot).pack(pady=5)
        tk.Button(frame, text="Toggle Crosshair", command=self.aimbot.toggle_crosshair).pack(pady=5)

        # Sliders
        tk.Label(frame, text="Aim FOV:", bg='#2e2e2e', fg='white').pack()
        self.fov_var = tk.DoubleVar(value=self.aimbot.aim_fov)
        tk.Scale(frame, from_=10, to=200, orient=tk.HORIZONTAL, variable=self.fov_var,
                 command=lambda _: self.aimbot.update_config(aim_fov=self.fov_var.get()), bg='#2e2e2e', fg='white').pack()

        tk.Label(frame, text="Aim Smoothness:", bg='#2e2e2e', fg='white').pack()
        self.smooth_var = tk.DoubleVar(value=self.aimbot.aim_smooth)
        tk.Scale(frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.smooth_var,
                 command=lambda _: self.aimbot.update_config(aim_smooth=self.smooth_var.get()), bg='#2e2e2e', fg='white').pack()

        # Stats display
        self.stats_var = tk.StringVar(value="FPS: 0 | Conf: 0.00 | Dets: 0")
        tk.Label(frame, textvariable=self.stats_var, bg='#2e2e2e', fg='cyan', font=("Arial", 10)).pack(pady=5)

        # Plot
        self.figure = Figure(figsize=(4, 3), dpi=100, facecolor='#2e2e2e')
        self.ax = self.figure.add_subplot(111, facecolor='#1e1e1e')
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_visuals_tab(self):
        """Setup enhanced Visuals tab with theme selection and color picker."""
        frame = tk.Frame(self.visuals_tab, bg='#2e2e2e')
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.visuals_var = tk.BooleanVar(value=self.visuals_enabled)
        tk.Checkbutton(frame, text="Enable Visuals (ESP)", variable=self.visuals_var, command=self.toggle_visuals,
                       bg='#2e2e2e', fg='white', selectcolor='#1e1e1e').pack(pady=5)

        # Theme selection
        tk.Label(frame, text="ESP Theme:", bg='#2e2e2e', fg='white').pack()
        self.theme_var = tk.StringVar(value="default")
        tk.Radiobutton(frame, text="Default", value="default", variable=self.theme_var, bg='#2e2e2e', fg='white').pack()
        tk.Radiobutton(frame, text="Neon", value="neon", variable=self.theme_var, bg='#2e2e2e', fg='white').pack()
        tk.Radiobutton(frame, text="Retro", value="retro", variable=self.theme_var, bg='#2e2e2e', fg='white').pack()

        # Color and thickness
        tk.Label(frame, text="ESP Color:", bg='#2e2e2e', fg='white').pack()
        self.color_entry = tk.Entry(frame, bg='#1e1e1e', fg='white')
        self.color_entry.insert(0, "red")
        self.color_entry.pack()

        tk.Label(frame, text="ESP Thickness:", bg='#2e2e2e', fg='white').pack()
        self.thickness_var = tk.DoubleVar(value=2)
        tk.Scale(frame, from_=1, to=5, resolution=1, orient=tk.HORIZONTAL, variable=self.thickness_var,
                 bg='#2e2e2e', fg='white').pack()

        self.labels_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text="Show Confidence Labels", variable=self.labels_var, bg='#2e2e2e', fg='white', selectcolor='#1e1e1e').pack()

        tk.Button(frame, text="Apply Visuals", command=self.apply_visuals).pack(pady=5)

    def setup_utilities_tab(self):
        """Setup enhanced Utilities tab with all Aimbot utilities."""
        frame = tk.Frame(self.utilities_tab, bg='#2e2e2e')
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        tk.Label(frame, text="Utilities", bg='#2e2e2e', fg='white', font=("Arial", 12, "bold")).pack(pady=5)

        # Sensitivity and calibration
        tk.Button(frame, text="Setup Sensitivity", command=setup_sensitivity).pack(pady=5)
        tk.Button(frame, text="Auto-Calibrate Sensitivity", command=self.aimbot.auto_calibrate_sensitivity).pack(pady=5)

        # Snapshots and recording
        tk.Button(frame, text="Capture Snapshot", command=self.aimbot.capture_snapshot).pack(pady=5)
        tk.Label(frame, text="Record Duration (s):", bg='#2e2e2e', fg='white').pack()
        self.record_duration = tk.Entry(frame, bg='#1e1e1e', fg='white')
        self.record_duration.insert(0, "5")
        self.record_duration.pack()
        tk.Button(frame, text="Start Recording", command=lambda: self.aimbot.start_recording(float(self.record_duration.get()))).pack(pady=5)

        # FOV adjustment
        tk.Label(frame, text="Game FOV (degrees):", bg='#2e2e2e', fg='white').pack()
        self.fov_entry = tk.Entry(frame, bg='#1e1e1e', fg='white')
        self.fov_entry.insert(0, "90")
        self.fov_entry.pack()
        tk.Button(frame, text="Adjust FOV", command=lambda: self.aimbot.adjust_fov(float(self.fov_entry.get()))).pack(pady=5)

        # Hotkeys
        tk.Label(frame, text="Hotkeys", bg='#2e2e2e', fg='white').pack()
        tk.Label(frame, text="Aimbot Hotkey:", bg='#2e2e2e', fg='white').pack()
        self.aim_hotkey = tk.Entry(frame, bg='#1e1e1e', fg='white')
        self.aim_hotkey.insert(0, self.aimbot.hotkeys["toggle_aimbot"])
        self.aim_hotkey.pack()
        tk.Label(frame, text="Trigger Hotkey:", bg='#2e2e2e', fg='white').pack()
        self.trigger_hotkey = tk.Entry(frame, bg='#1e1e1e', fg='white')
        self.trigger_hotkey.insert(0, self.aimbot.hotkeys["toggle_trigger"])
        self.trigger_hotkey.pack()
        tk.Label(frame, text="Snapshot Hotkey:", bg='#2e2e2e', fg='white').pack()
        self.snapshot_hotkey = tk.Entry(frame, bg='#1e1e1e', fg='white')
        self.snapshot_hotkey.insert(0, self.aimbot.hotkeys["snapshot"])
        self.snapshot_hotkey.pack()
        tk.Button(frame, text="Apply Hotkeys", command=self.apply_hotkeys).pack(pady=5)

        # Game state and export
        tk.Button(frame, text="Log Game State", command=self.aimbot.log_game_state).pack(pady=5)
        tk.Button(frame, text="Export Detections", command=self.aimbot.export_detections).pack(pady=5)

        # Performance monitor
        self.perf_var = tk.StringVar(value="CPU: 0% | FPS: 0")
        tk.Label(frame, textvariable=self.perf_var, bg='#2e2e2e', fg='cyan').pack(pady=5)

    def create_overlay(self):
        """Create a transparent overlay for ESP and crosshair."""
        self.overlay = tk.Toplevel(self.root)
        self.overlay.attributes('-fullscreen', True)
        self.overlay.attributes('-topmost', True)
        self.overlay.attributes('-transparentcolor', 'black')
        self.overlay.config(bg='black')
        self.overlay.overrideredirect(True)
        self.overlay_canvas = tk.Canvas(self.overlay, bg='black', highlightthickness=0)
        self.overlay_canvas.pack(fill=tk.BOTH, expand=True)
        self.overlay.withdraw()

    def toggle_visuals(self):
        """Toggle visuals and show/hide overlay."""
        self.visuals_enabled = self.visuals_var.get()
        if self.visuals_enabled:
            self.overlay.deiconify()
        else:
            self.overlay.withdraw()

    def apply_visuals(self):
        """Apply visuals settings to Aimbot."""
        try:
            self.aimbot.set_visuals_config(
                color=self.color_entry.get(),
                thickness=int(self.thickness_var.get()),
                theme=self.theme_var.get(),
                show_labels=self.labels_var.get()
            )
            logger.info(colored("Visuals settings applied!", "green"))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply visuals: {e}")

    def apply_hotkeys(self):
        """Apply custom hotkeys to Aimbot."""
        try:
            self.aimbot.set_hotkeys(
                toggle_aimbot=self.aim_hotkey.get(),
                toggle_trigger=self.trigger_hotkey.get(),
                snapshot=self.snapshot_hotkey.get()
            )
            messagebox.showinfo("Success", "Hotkeys updated!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply hotkeys: {e}")

    def draw_esp(self, detections):
        """Draw ESP boxes and crosshair on overlay."""
        self.overlay_canvas.delete("all")
        if self.visuals_enabled:
            color = self.aimbot.visuals_config["color"]
            for det in detections:
                x, y, w, h, conf = det
                if conf > self.aimbot.conf_threshold:
                    self.overlay_canvas.create_rectangle(x, y, x + w, y + h, outline=color, width=self.aimbot.visuals_config["thickness"])
                    if self.aimbot.visuals_config["show_labels"]:
                        self.overlay_canvas.create_text(x, y - 10, text=f"Conf: {conf:.2f}", fill=color, font=("Arial", 8))
            # Draw crosshair
            crosshair = self.aimbot.draw_crosshair()
            if crosshair:
                x, y, size, color = crosshair
                self.overlay_canvas.create_line(x - size, y, x + size, y, fill=color, width=2)
                self.overlay_canvas.create_line(x, y - size, x, y + size, fill=color, width=2)

    def update_status(self):
        """Toggle the analysis status."""
        if self.aimbot_status == "ENABLED":
            self.aimbot_status = "DISABLED"
            self.status_var.set("DISABLED")
            self.root.nametowidget(self.aimbot_tab).children['!label2'].config(fg="red")
        else:
            self.aimbot_status = "ENABLED"
            self.status_var.set("ENABLED")
            self.root.nametowidget(self.aimbot_tab).children['!label2'].config(fg="green")

    def is_enabled(self):
        """Check if analysis is enabled."""
        return self.aimbot_status == "ENABLED"

    def plot_logs(self):
        """Visualize the last N logs for FPS and confidence."""
        if not self.logs:
            return
        recent_logs = self.logs[-self.plot_interval:]
        times = [log["timestamp"] for log in recent_logs]
        fps = [log["fps"] for log in recent_logs]
        conf = [log["conf"] for log in recent_logs]
        
        self.ax.cla()
        self.ax.plot(times, fps, label="FPS", color="#1f77b4", marker='o')
        self.ax.plot(times, conf, label="Detection Confidence", color="#ff7f0e", marker='x')
        self.ax.set_xlabel("Time (s)", color='white')
        self.ax.set_ylabel("Metrics", color='white')
        self.ax.set_title("Real-Time Analysis Metrics", color='white')
        self.ax.legend(facecolor='#2e2e2e', edgecolor='white', labelcolor='white')
        self.ax.grid(True, color='#555555')
        self.ax.tick_params(colors='white')
        self.canvas.draw()

        # Update stats
        if recent_logs:
            latest = recent_logs[-1]
            self.stats_var.set(f"FPS: {latest['fps']} | Conf: {latest['conf']:.2f} | Dets: {latest['det_count']}")

    def update_performance_monitor(self):
        """Update CPU usage and FPS display."""
        cpu_usage = psutil.cpu_percent()
        fps = self.logs[-1]["fps"] if self.logs else 0
        self.perf_var.set(f"CPU: {cpu_usage:.1f}% | FPS: {fps}")

    def start(self):
        """Start the main analysis loop."""
        logger.info(colored("Starting analysis loop 🚀", "yellow"))
        while self.running:
            if not self.is_enabled():
                time.sleep(0.1)
                continue
            start_time = time.perf_counter()
            try:
                log_entry = self.aimbot.log_detections(start_time)
                if log_entry:
                    self.logs.append(log_entry)
                    self.plot_logs()
                    self.update_performance_monitor()
                detections = self.aimbot.get_detections()
                self.draw_esp(detections)
            except Exception as e:
                logger.error(colored(f"Detection error: {e}", "red"))
                messagebox.showerror("Error", f"Detection failed: {e}")
            time.sleep(0.9)
            self.root.update()  # Keep GUI responsive
        self.root.mainloop()

    def clean_up(self):
        """Clean up resources and exit."""
        logger.info(colored("F2 PRESSED. QUITTING... ✌️", "yellow"))
        self.running = False
        self.aimbot.clean_up()
        if self.collect_data:
            self.save_logs()
        self.root.quit()
        self.root.destroy()
        if self.overlay:
            self.overlay.destroy()
        os._exit(0)

    def save_logs(self):
        """Save collected logs to JSON."""
        log_file = "lib/data/analysis_logs.json"
        try:
            with open(log_file, 'w') as f:
                json.dump(self.logs, f, indent=4)
            logger.info(colored(f"Logs saved to {log_file}", "green"))
            messagebox.showinfo("Success", "Logs saved successfully!")
        except Exception as e:
            logger.error(colored(f"Failed to save logs: {e}", "red"))
            messagebox.showerror("Error", f"Failed to save logs: {e}")

def on_release(key, analyzer):
    """Handle keyboard release events for toggling and utilities."""
    try:
        key_str = str(key).replace("Key.", "")
        if key_str == analyzer.aimbot.hotkeys["toggle_aimbot"]:
            analyzer.aimbot.toggle_auto_aim()
            analyzer.update_status()
        elif key_str == analyzer.aimbot.hotkeys["toggle_trigger"]:
            analyzer.aimbot.toggle_trigger_bot()
        elif key_str == analyzer.aimbot.hotkeys["snapshot"]:
            analyzer.aimbot.capture_snapshot()
        elif key == keyboard.Key.f2:
            analyzer.clean_up()
    except NameError:
        pass

def setup_sensitivity():
    """Configure and save in-game sensitivity settings."""
    config_dir = "lib/config"
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.json")
    
    logger.info(colored("In-game X and Y axis sensitivity should be the same", "cyan"))
    
    def prompt_float(message):
        while True:
            try:
                return float(input(message))
            except ValueError:
                logger.warning(colored("Invalid Input. Enter only a number (e.g., 6.9)", "yellow"))
    
    xy_sens = prompt_float("X-Axis and Y-Axis Sensitivity: ")
    targeting_sens = prompt_float("Targeting Sensitivity: ")
    
    sensitivity_settings = {
        "xy_sens": xy_sens,
        "targeting_sens": targeting_sens,
        "xy_scale": 10 / xy_sens,
        "targeting_scale": 1000 / (targeting_sens * xy_sens)
    }
    
    try:
        with open(config_path, 'w') as outfile:
            json.dump(sensitivity_settings, outfile, indent=4)
        logger.info(colored("Sensitivity configuration saved successfully", "green"))
    except Exception as e:
        logger.error(colored(f"Failed to save configuration: {e}", "red"))

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LunarAnalyzer: Neural Network-Based Game Analyzer")
    parser.add_argument('--collect_data', action='store_true', help="Enable data collection mode")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    parser.add_argument('--setup', action='store_true', help="Force sensitivity setup")
    parser.add_argument('--box_constant', type=int, default=320, help="Detection box constant (default: 320)")
    parser.add_argument('--plot_interval', type=int, default=100, help="Number of logs to plot (default: 100)")
    return parser.parse_args()

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

    args = parse_arguments()
    
    config_path = "lib/config/config.json"
    if args.setup or not os.path.exists(config_path):
        if not os.path.exists(config_path):
            logger.warning(colored("Sensitivity configuration not found", "yellow"))
        setup_sensitivity()
    
    os.makedirs("lib/data", exist_ok=True)
    
    analyzer = LunarAnalyzerGUI(
        box_constant=args.box_constant,
        collect_data=args.collect_data,
        debug=args.debug,
        plot_interval=args.plot_interval
    )
    
    listener = keyboard.Listener(on_release=lambda key: on_release(key, analyzer))
    listener.start()
    
    try:
        analyzer.start()
    except KeyboardInterrupt:
        analyzer.clean_up()
    except Exception as e:
        logger.critical(colored(f"Unexpected error: {e}", "red"))
        analyzer.clean_up()