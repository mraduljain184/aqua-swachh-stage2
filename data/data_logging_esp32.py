"""
Aqua-Swachh ESP32 Serial Logger & Live Inference Bridge
======================================================
Reads sensor readings from the ESP32 via USB Serial, converts raw TDS probe
voltage to ppm, logs to CSV, and optionally runs real-time ML inference.

Expected serial line (firmware/esp32_sensor_node.ino):

    tds1_voltage,tds2_voltage,temperature[,flow_rate]
    e.g. 1.214,0.144,24.81          (no flow sensor fitted)
         1.214,0.144,24.81,1.85     (flow sensor fitted)

TDS Sensor 1 = feed (tds_in), TDS Sensor 2 = permeate (tds_out).
Use --input-units ppm if your firmware already sends ppm values.
Non-CSV lines such as "TDS1: 1.214 V  TDS2: 0.144 V  Temp: 24.81 C" are also
accepted: the first 3 (or 4) numbers on the line are used, in that order.
"""

import os
import re
import sys
import time
import argparse
from datetime import datetime
from typing import Optional, Tuple

# Allow `python data/data_logging_esp32.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.sensor_calibration import voltage_to_tds_ppm, ADC_MAX_VOLTAGE, DEFAULT_K_VALUE  # noqa: E402

# Optional pyserial import
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

CSV_FILE = "data/esp32_live_log.csv"
CSV_HEADER = "timestamp,sample_type,tds_in_voltage,tds_out_voltage,tds_in,tds_out,temperature,flow_rate\n"

# Standalone numbers only: skips digits glued to letters, e.g. the "1" in "TDS1"
# (but still accepts "1.214V" / "24.81°C").
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9.])[-+]?(?:\d+\.\d+|\.\d+|\d+)")


def parse_serial_line(line: str) -> Optional[Tuple[float, float, float, Optional[float]]]:
    """Returns (value1, value2, temperature, flow_or_None) or None if the line is not a reading."""
    parts = [p.strip() for p in line.split(',')]
    try:
        if len(parts) in (3, 4):
            nums = [float(p) for p in parts]
        else:
            raise ValueError
    except ValueError:
        nums = [float(n) for n in _NUMBER_RE.findall(line)]
        if len(nums) not in (3, 4):
            return None
    flow = nums[3] if len(nums) == 4 else None
    return nums[0], nums[1], nums[2], flow


def log_reading_to_csv(filepath: str, sample_type: str, v_in: Optional[float], v_out: Optional[float],
                       tds_in: float, tds_out: float, temp: float, flow: Optional[float],
                       timestamp: Optional[str] = None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def fmt(v, nd):
        return "" if v is None else f"{v:.{nd}f}"

    file_exists = os.path.exists(filepath)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(CSV_HEADER)
        f.write(f"{timestamp},{sample_type},{fmt(v_in, 4)},{fmt(v_out, 4)},{tds_in:.1f},{tds_out:.1f},"
                f"{temp:.2f},{fmt(flow, 2)}\n")


def run_serial_logger(port: str, baudrate: int = 115200, output_csv: str = CSV_FILE,
                      run_inference: bool = True, input_units: str = "voltage",
                      k_in: float = DEFAULT_K_VALUE, k_out: float = DEFAULT_K_VALUE,
                      sample_type: str = ""):
    if not SERIAL_AVAILABLE:
        print("[Error] 'pyserial' package is not installed. Run: pip install pyserial")
        sys.exit(1)

    predictor = None
    if run_inference:
        try:
            from inference.predict import FilterHealthPredictor
            predictor = FilterHealthPredictor()
            print("[INFO] ML Filter Health Predictor loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load ML model for live inference ({e}). Logging only.")

    print(f"[INFO] Connecting to ESP32 on port {port} at {baudrate} baud...")
    try:
        ser = serial.Serial(port, baudrate, timeout=2.0)
        time.sleep(2)  # Wait for ESP32 reboot on connection
        print("[INFO] Connected. Listening for sensor packets...")
        if input_units == "voltage":
            print(f"Expected: 'tds1_V,tds2_V,temp_C[,flow_Lpm]' (e.g. 1.214,0.144,24.81) | K1={k_in} K2={k_out}")
        else:
            print("Expected: 'tds_in_ppm,tds_out_ppm,temp_C[,flow_Lpm]' (e.g. 450.0,35.0,26.5)")

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            parsed = parse_serial_line(line)
            if parsed is None:
                print(f"[RAW ESP32]: {line}")
                continue

            a, b, temp, flow = parsed
            if input_units == "voltage":
                if not (0 <= a <= ADC_MAX_VOLTAGE and 0 <= b <= ADC_MAX_VOLTAGE):
                    print(f"[SKIP] Voltage out of ADC range (0-{ADC_MAX_VOLTAGE} V): {line}")
                    continue
                v_in, v_out = a, b
                tds_in = float(voltage_to_tds_ppm(v_in, temp, k_in))
                tds_out = float(voltage_to_tds_ppm(v_out, temp, k_out))
            else:
                v_in = v_out = None
                tds_in, tds_out = a, b

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_reading_to_csv(output_csv, sample_type, v_in, v_out, tds_in, tds_out, temp, flow, now_str)

            status_str = ""
            if predictor:
                pred = predictor.predict_sample(tds_in, tds_out, temp, flow)
                status_str = (f" | η={pred['physics_indicators']['salt_rejection_pct']:.1f}%"
                              f" | Status: {pred['predicted_status']} ({pred['confidence']*100:.1f}%, {pred['model_variant']})")

            volt_str = f"[{v_in:.3f} V / {v_out:.3f} V] " if v_in is not None else ""
            flow_str = f"{flow:.2f} L/min" if flow is not None else "n/a"
            print(f"[{now_str}] {volt_str}In: {tds_in:.1f} ppm | Out: {tds_out:.1f} ppm | Temp: {temp:.2f}°C | Flow: {flow_str}{status_str}")
    except KeyboardInterrupt:
        print("\n[INFO] Logging stopped by user.")
    except Exception as ex:
        print(f"[ERROR] Serial communication error: {ex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aqua-Swachh ESP32 Serial Logger")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (e.g. COM3, /dev/ttyUSB0, /dev/cu.usbserial-XXXX)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--output", type=str, default=CSV_FILE, help="CSV log path")
    parser.add_argument("--input-units", choices=["voltage", "ppm"], default="voltage",
                        help="Units of the two TDS values sent by the ESP32 (default: voltage)")
    parser.add_argument("--k-in", type=float, default=DEFAULT_K_VALUE, help="Calibration K-value for TDS Sensor 1 (feed)")
    parser.add_argument("--k-out", type=float, default=DEFAULT_K_VALUE, help="Calibration K-value for TDS Sensor 2 (permeate)")
    parser.add_argument("--sample-type", type=str, default="", help="Label written to every row (e.g. 'Normal water')")
    parser.add_argument("--no-inference", action="store_true", help="Disable real-time ML prediction")
    args = parser.parse_args()

    run_serial_logger(port=args.port, baudrate=args.baud, output_csv=args.output,
                      run_inference=not args.no_inference, input_units=args.input_units,
                      k_in=args.k_in, k_out=args.k_out, sample_type=args.sample_type)
