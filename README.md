# 💧 Aqua-Swachh: Smart Water Purifier Predictive Health Intelligence

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32--S3-E7352C?logo=espressif&logoColor=white)](https://www.espressif.com/)
[![Track](https://img.shields.io/badge/atomQUEST-Track%202%20%28RGIPT%29-0284c7)](https://rgipt.ac.in)
[![Tests](https://img.shields.io/badge/Tests-35%20Passing-10b981?logo=checkmarx&logoColor=white)](tests/test_backend.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Predictive Filter Health Monitoring for Smart Water Purifiers · Stage 3 Prototype**  
> An edge-to-cloud machine learning system that diagnoses RO membrane health, detects early scaling, and enforces deterministic safety guardrails on drinking water quality.
>
> 🌐 **Live Public Demo:** [https://c48cae0c1101b2.lhr.life](https://c48cae0c1101b2.lhr.life)  
> 📦 **GitHub Repository:** [https://github.com/Sargam-max/aqua-swachh-stage2](https://github.com/Sargam-max/aqua-swachh-stage2)

---

## 📌 The Problem We Are Solving

Millions of households and commercial facilities rely on Reverse Osmosis (RO) drinking water purifiers. Yet virtually all consumer purifiers operate on **blind, timer-based maintenance**:

1. **Premature Filter Swaps:** Technicians routinely replace functional cartridges every 6 to 12 months based purely on the calendar, wasting thousands of rupees and generating unnecessary plastic and activated carbon waste.
2. **Undetected Membrane Breakthrough:** If an RO membrane develops microscopic tears or scales up prematurely due to aggressive local borewell water, consumers continue drinking contaminated water without knowing it.
3. **Misleading Single-Sensor Alarms:** A simple TDS spike doesn't necessarily mean a filter is broken—if the city supply or tanker water becomes harder, a healthy membrane will naturally have a higher absolute permeate TDS while still rejecting 93%+ of impurities.

**Aqua-Swachh** solves this by pairing an **ESP32-S3 multi-sensor IoT edge array** with **physics-informed machine learning** and **deterministic safety guardrails** to continuously diagnose the true physical condition of the filtration membrane.

---

## 🔬 How It Works: The 5-Stage Purifier Architecture

Aqua-Swachh monitors the water journey across critical checkpoints in the filtration process:

```
[ 1. Raw Tap Inlet ] ────► [ 2. Sediment & Carbon ] ────► [ 3. RO Membrane Core ] ────► [ 4. Permeate Output ] ────► [ 5. Flow Meter ] ────► Pure Tap
   Xcluma TDS-1                5-Micron Spun Yarn              75 GPD Thin-Film             Xcluma TDS-2               YF-S201 Turbine
   GPIO 1 (ADC1)               Particulate Trap                Salt Separation η            GPIO 2 (ADC1)              GPIO 5 (Pulse ISR)
   Measures Feed TDS           Turbidity < 1 NTU               Fouling & Scaling            Measures Pure TDS          Measures Permeate Q
```

### Sensor Array Specification

| Sensor Channel | Hardware Component | Interface | Measurement Range | Role in Diagnostics |
| :--- | :--- | :--- | :--- | :--- |
| **Feed TDS (TDS Sensor 1)** | Xcluma Analog TDS V1.0 | GPIO 1 (ADC1) | $0 - 2.3\text{ V}$ analog (≈ $0 - 1000\text{ ppm}$) | Baseline feed mineral concentration |
| **Permeate TDS (TDS Sensor 2)** | Xcluma Analog TDS V1.0 | GPIO 2 (ADC1) | $0 - 2.3\text{ V}$ analog | Purified permeate solute concentration |
| **Water Temperature** | Dallas DS18B20 1-Wire | GPIO 4 | $-10 - 85^\circ\text{C}$ | Conductivity normalization to 25°C baseline |
| **Permeate Flow Rate** *(optional, not fitted yet)* | YF-S201 1/2" Hall Turbine | GPIO 5 (ISR) | $0.2 - 10\text{ L/min}$ | Detects particulate head-loss & flow choking |

> **Current hardware build:** the TDS probes report **raw voltage (V)**, and **no flow meter is fitted**.
> Voltage is converted to ppm on the PC side (see below), and every part of the pipeline treats `flow_rate` as optional.

### Probe Voltage → TDS (ppm)

The analog TDS boards output a voltage. `model/sensor_calibration.py` converts it with the Gravity/DFRobot reference curve:

$$V_{25} = \frac{V}{1 + 0.02\,(T - 25)}, \qquad \text{TDS} = \left(133.42\,V_{25}^3 - 255.86\,V_{25}^2 + 857.39\,V_{25}\right) \times 0.5 \times K$$

`K` is each probe's calibration factor (default `1.0` = uncalibrated). Calibrate by dipping a probe in a known standard (e.g. 342 ppm NaCl) and setting `K = known_ppm / measured_ppm`. K can be passed to the API, the dashboard, and the serial logger without reflashing the ESP32.

---

## 🧠 Physics-Informed Machine Learning (The Science)

Instead of treating machine learning like a black box, Aqua-Swachh embeds **electrochemical water treatment equations** into the feature pipeline (`model/features.py`):

### 1. Salt Rejection Efficiency ($\eta$)
$$\eta = 1 - \left(\frac{\text{TDS}_{out}}{\text{TDS}_{in}}\right)$$
*Why it matters:* Rejection isolates membrane health from input water swings. A feed TDS jump from 400 to 800 ppm filtered to 50 ppm represents a **healthy 93.8% rejection**, proving the membrane is working nominally.

### 2. 25°C Temperature Compensation
$$\text{TDS}_{\text{norm}} = \frac{\text{TDS}}{1.0 + 0.02 \times (T - 25.0)}$$
*Why it matters:* Water electrical conductivity drifts by approximately $2\%$ per $^\circ\text{C}$. This formula normalizes winter ($15^\circ\text{C}$) and summer ($35^\circ\text{C}$) readings to a shared 25°C calibration baseline.

### 3. Flow-Rejection Composite Index
$$\text{Index} = Q \times \max(\eta, 0)$$
*Why it matters:* Distinguishes **pre-filter clogging** (water remains clean, but flow plummets to a trickle) from **membrane rupture** (flow stays fast, but salt rejection crashes).

---

## 🎯 Model Benchmarks & Evaluation

We benchmarked three model tiers using stratified 5-fold cross-validation on 1,500 **synthetic**, physics-grounded operational samples (`model/train.py`). Two variants are trained on the same split:

**Full model** — TDS ×2 + temperature + flow (`random_forest_filter_model.joblib`):

| Model Architecture | Cross-Val F1 (Macro) | Test Accuracy | Role in Pipeline |
| :--- | :---: | :---: | :--- |
| **Dummy Classifier** (Majority Class) | $0.2366$ | $55.0\%$ | Baseline lower bound |
| **Logistic Regression** (Linear Baseline) | $0.9962 \pm 0.008$ | $99.7\%$ | Linear benchmark comparison |
| **Random Forest Classifier** (Primary) | **$1.0000 \pm 0.000$** | **$100.0\%$** | Used when a flow reading is present |

**No-flow model** — TDS ×2 + temperature only (`random_forest_filter_model_no_flow.joblib`):

| Model Architecture | Cross-Val F1 (Macro) | Test Accuracy | Role in Pipeline |
| :--- | :---: | :---: | :--- |
| **Dummy Classifier** (Majority Class) | $0.2366$ | $55.0\%$ | Baseline lower bound |
| **Logistic Regression** (Linear Baseline) | $0.9473 \pm 0.013$ | $95.0\%$ | Linear benchmark comparison |
| **Random Forest Classifier** (Primary) | **$0.9468 \pm 0.013$** | **$95.0\%$** | **Used for the current hardware (no flow meter)** |

The no-flow model is selected automatically whenever `flow_rate` is missing. Its lower score is expected: without flow it cannot tell flow-choke failures apart from early scaling.

### Three Diagnostic States:
* 🟢 **NORMAL:** Salt rejection $\eta > 88\%$, nominal permeate flow ($1.5 - 2.2\text{ L/min}$). Water quality is optimal.
* 🟡 **MONITOR:** Early mineral scaling or foulant deposition ($\eta = 74\% - 88\%$, flow $1.0 - 1.4\text{ L/min}$). Plan cartridge swap in upcoming cycles.
* 🔴 **SERVICE REQUIRED:** Membrane barrier compromised or particulate blockages ($Q < 0.7\text{ L/min}$ or $\text{TDS}_{out} > 200\text{ ppm}$).

---

## 🛡️ Deterministic Safety Guardrails

In consumer drinking water safety, statistical AI should never have unchecked authority. Aqua-Swachh implements **deterministic safety guardrails** (`inference/predict.py`) that run concurrently with the ML model:

$$\text{Trigger Condition} = (\text{TDS}_{out} \ge 250\text{ ppm}) \lor (\eta < 55\%) \lor (Q < 0.40\text{ L/min})$$

*(The flow condition is only evaluated when a flow reading exists.)*

*If any trigger condition is met, the system **automatically overrides** the ML prediction to `Service Required`, alerts the user, and triggers an emergency indicator.*

---

## 📁 Repository Structure

```
├── web/
│   ├── app.py                      # FastAPI REST API & static server
│   └── static/
│       ├── index.html              # Dashboard layout: Live Check, Batch Lab, ESP32 Node & Pinout, ML Methodology
│       └── app.js                  # Dashboard logic: checks, simulation stories, charts, batch upload
├── model/
│   ├── features.py                 # Physics feature extraction (η, Temp Comp, Delta TDS)
│   ├── sensor_calibration.py       # Probe voltage → ppm, K-value calibration, CSV header aliases
│   ├── train.py                    # Trains full + no-flow models, 5-fold CV, benchmark
│   └── saved_models/
│       ├── random_forest_filter_model.joblib          # Full model (with flow)
│       ├── random_forest_filter_model_no_flow.joblib  # No-flow model (TDS + temperature)
│       └── metrics_summary.json    # Exported evaluation metrics & confusion matrix
├── data/
│   ├── generate_dataset.py         # Physics-based synthetic dataset generator
│   ├── data_logging_esp32.py       # USB serial bridge logger (voltage → ppm, flow optional)
│   ├── real_sensor_readings.csv    # Real bench readings: raw TDS voltages + temperature (18 rows)
│   └── sensor_data.csv             # 1,500 labeled synthetic benchmark records
├── firmware/
│   └── esp32_sensor_node.ino       # ESP32-S3 firmware: streams raw probe voltage (+ optional flow)
├── inference/
│   └── predict.py                  # High-level inference engine + safety guardrails
├── tests/
│   └── test_backend.py             # Automated 35-test backend verification suite
├── demo.py                         # Interactive terminal test bench (5 scenarios)
├── run_dashboard.bat               # 1-Click Windows launcher (auto-recovers ports & venv)
├── stop_dashboard.bat              # 1-Click clean shutdown utility for Windows
├── Dockerfile                      # Production container image
├── Procfile                        # Cloud process definition (Render / Railway)
├── requirements.txt                # Cross-platform dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### Option 1: 1-Click Launch on Windows (Easiest)
Simply double-click:
👉 **`run_dashboard.bat`**

*The launcher automatically detects Python, verifies model weights, cleans up any occupied ports, and opens your default browser at `http://127.0.0.1:8000`.*

---

### Option 2: Terminal Setup (Windows / macOS / Linux)

```bash
# 1. Clone the repository
git clone https://github.com/Sargam-max/aqua-swachh-stage2.git
cd aqua-swachh-stage2

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the web dashboard
uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at **`http://localhost:8000`**.

---

### Option 3: Run the CLI Test Bench
```bash
python demo.py
```
*Evaluates the 5 real-world filter scenarios in your terminal with physics indicators, probabilities, and recommendations.*

---

### Option 4: Run the Automated Test Suite
```bash
python -m unittest tests/test_backend.py -v
```
*Runs all 35 verification tests across physics calculations, voltage conversion, ML inference (with and without flow), safety guardrails, and live REST API endpoints. The API tests need the server running on port 8000.*

---

## 🧪 Testing with Real Sensor Data

### Real bench readings (bundled)

`data/real_sensor_readings.csv` holds readings transcribed from the ESP32 Serial Monitor, as raw probe voltage:

| Sample type | Readings | TDS Sensor 1 (avg) | TDS Sensor 2 (avg) | Temperature | → Feed / Permeate (K=1.0) | Salt rejection | Model output |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Normal water | 8 | 1.213 V | 0.144 V | 24.2 – 24.8 °C | 456.8 / 60.0 ppm | 86.9 % | Monitor |
| Water cooler water | 10 | 1.214 V | 0.150 V | 31.1 – 32.7 °C | 391.3 / 54.3 ppm | 86.1 % | Monitor |

TDS Sensor 1 is treated as feed (`tds_in`) and TDS Sensor 2 as permeate (`tds_out`), as wired in the firmware. Treat these results as indicative only: the probes are not calibrated yet (K = 1.0) and there is no flow data. Both samples sit just under the 88 % "Normal" rejection boundary, so a small K correction could change the class.

### Demo file with simulated flow

`data/real_readings_with_simulated_flow.csv` contains the **same real voltages** plus **simulated** `flow_rate` values: healthy flow for Normal water, and progressive pre-filter clogging for Water cooler water. Every row is marked `flow_source = simulated`. It shows how the full model reacts to flow: 13 rows are Monitor (confidence 52.5–100%) and 5 are Service Required (72.5–92.5%), and the last row triggers the low-flow guardrail. The dashboard flags these values as simulated. They are not measurements.

### Using the dashboard

1. Open the **Batch Sensor Lab** tab in the web dashboard.
2. Click **"Analyze My Real Readings (Voltage)"** to run the bundled bench readings, with a per-sample-type summary. You can set probe K-values first.
3. Or drop any `.csv` file. Two formats are accepted, and `flow_rate` is optional in both:
   * **ppm:** `tds_in, tds_out, temperature[, flow_rate]`
   * **voltage:** `tds_in_voltage, tds_out_voltage, temperature[, flow_rate, sample_type, reading_no]`  
     Spreadsheet headers such as `TDS Sensor 1 (V)`, `TDS Sensor 2 (V)`, `Temperature (°C)`, `Sample Type`, `Reading No.` are recognised automatically.
4. On the **Live Diagnostics** tab, use the *Raw Probe Voltage Input* panel to diagnose a single Serial Monitor reading, or untick **sensor fitted** on the flow card to use the no-flow model.

### API

```bash
# Single reading from raw voltage (flow optional, K-values optional)
curl -X POST localhost:8000/api/predict-voltage -H 'Content-Type: application/json' \
     -d '{"tds_in_voltage": 1.214, "tds_out_voltage": 0.144, "temperature": 24.81}'

# Single reading in ppm without flow
curl -X POST localhost:8000/api/predict -H 'Content-Type: application/json' \
     -d '{"tds_in": 453.2, "tds_out": 59.5, "temperature": 24.81}'

# Bundled real readings / CSV upload with calibration
curl 'localhost:8000/api/load-real-data?k_in=1.0&k_out=1.0'
curl -F file=@my_readings.csv 'localhost:8000/api/upload-csv?k_in=1.05&k_out=0.98'
```

### Live logging from the ESP32

```bash
pip install pyserial
python data/data_logging_esp32.py --port /dev/cu.usbserial-XXXX --sample-type "Normal water" --k-in 1.0 --k-out 1.0
```

The logger accepts `tds1_V,tds2_V,temp_C[,flow]` lines (the format the firmware prints). It also accepts free-text lines such as `TDS1: 1.214 V  TDS2: 0.144 V  Temp: 24.81 C`. It writes voltage and converted ppm to `data/esp32_live_log.csv` and prints a live diagnosis.

### Getting labeled real data for training

The models are still trained only on the synthetic dataset, because real readings have no ground-truth filter condition. To train on real data, log readings from cartridges whose state you know (new, mid-life, or due for service). Add a `filter_status` column, calibrate the probes, and fit a flow meter if you want clogging detection.

---

## 🌐 Free Cloud Deployment (Render.com)

1. Fork or push this repository to GitHub.
2. Sign in to [Render.com](https://render.com) and click **New +** → **Web Service**.
3. Select your repository (`Sargam-max/aqua-swachh-stage2`).
4. Configure:
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt && python data/generate_dataset.py && python -m model.train`
   * **Start Command:** `uvicorn web.app:app --host 0.0.0.0 --port $PORT`
5. Click **Deploy**. Render provides a free, public 24/7 HTTPS URL.

---

## 👥 Team & Institutional Acknowledgments

* **Project:** Aqua-Swachh (Predictive Filter Health Monitoring for Smart Water Purifiers)
* **Team:** Team Aqua-Swachh
* **Competition:** **atomQUEST Track 2**
* **Institution:** **Rajiv Gandhi Institute of Petroleum Technology (RGIPT)**

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and build upon this work for academic, commercial, and research purposes.
