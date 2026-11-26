"""
Aqua-Swachh: Web Dashboard Backend & Real-World Sensor Analysis API
===================================================================
FastAPI application serving:
- Real-time ML inference for single sensor snapshots
- Raw TDS probe voltage -> ppm conversion (Gravity polynomial + temp compensation)
- Batch CSV processor for real sensor data logs (ppm OR voltage columns, flow optional)
- Physics indicators, safety guardrails & feature explainability
- System configuration and hardware telemetry
- Static human-designed frontend UI
"""

import os
import io
import json
import random
import pandas as pd
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from inference.predict import FilterHealthPredictor
from model.sensor_calibration import (
    prepare_sensor_frame, voltage_to_tds_ppm, SensorSchemaError, DEFAULT_K_VALUE
)

app = FastAPI(
    title="Aqua-Swachh Smart Purifier AI",
    description="Predictive Filter Health Monitoring API (atomQUEST Track 2 - RGIPT) - Stage 3 Prototype",
    version="3.0.0"
)

# Paths
MODEL_PATH = "model/saved_models/random_forest_filter_model.joblib"
METRICS_PATH = "model/saved_models/metrics_summary.json"
REAL_DATA_PATH = "data/real_sensor_readings.csv"
SIMULATED_FLOW_DATA_PATH = "data/real_readings_with_simulated_flow.csv"

try:
    predictor = FilterHealthPredictor(model_path=MODEL_PATH)
except Exception as e:
    print(f"[WARN] Predictor initialization deferred: {e}")
    predictor = None

class SensorInput(BaseModel):
    tds_in: float = Field(..., ge=0, le=2500, description="Feed water TDS in ppm")
    tds_out: float = Field(..., ge=0, le=2500, description="Permeate water TDS in ppm")
    temperature: float = Field(25.0, ge=0, le=60, description="Water temperature in °C")
    flow_rate: Optional[float] = Field(None, ge=0, le=10, description="Permeate flow rate in L/min (omit / null if no flow sensor)")

class VoltageSensorInput(BaseModel):
    tds_in_voltage: float = Field(..., ge=0, le=3.3, description="TDS Sensor 1 (feed) analog output in volts")
    tds_out_voltage: float = Field(..., ge=0, le=3.3, description="TDS Sensor 2 (permeate) analog output in volts")
    temperature: float = Field(25.0, ge=0, le=60, description="Water temperature in °C")
    flow_rate: Optional[float] = Field(None, ge=0, le=10, description="Permeate flow rate in L/min (omit / null if no flow sensor)")
    k_in: float = Field(DEFAULT_K_VALUE, gt=0, le=5, description="Calibration K-value for TDS Sensor 1")
    k_out: float = Field(DEFAULT_K_VALUE, gt=0, le=5, description="Calibration K-value for TDS Sensor 2")

PRESET_SCENARIOS = [
    {
        "id": "new_filter",
        "name": "Fresh RO Membrane",
        "category": "Normal",
        "description": "Standard municipal water (420 ppm) with high rejection (>93%) and strong permeate flow.",
        "tds_in": 420.0,
        "tds_out": 28.0,
        "temperature": 25.5,
        "flow_rate": 1.95
    },
    {
        "id": "hard_water",
        "name": "High-Hardness Tap Water",
        "category": "Normal",
        "description": "Challenging hard water feed (780 ppm) successfully handled by healthy membrane (48 ppm permeate).",
        "tds_in": 780.0,
        "tds_out": 48.0,
        "temperature": 27.0,
        "flow_rate": 1.80
    },
    {
        "id": "moderate_scaling",
        "name": "Early Membrane Scaling",
        "category": "Monitor",
        "description": "Mineral foulant build-up on polyamide pores. Salt rejection dropped to ~77.5%, flow restricted to 1.25 L/min.",
        "tds_in": 510.0,
        "tds_out": 115.0,
        "temperature": 26.8,
        "flow_rate": 1.25
    },
    {
        "id": "clogged_flow",
        "name": "Sediment / Carbon Clogging",
        "category": "Service Required",
        "description": "Pre-filters blocked by sediment; severe pressure drop choking permeate flow to 0.65 L/min.",
        "tds_in": 480.0,
        "tds_out": 120.0,
        "temperature": 26.0,
        "flow_rate": 0.65
    },
    {
        "id": "membrane_damage",
        "name": "Membrane Pore Breakthrough",
        "category": "Service Required",
        "description": "Membrane tear or chlorine chemical attack allowing raw dissolved solids to pass directly through.",
        "tds_in": 550.0,
        "tds_out": 230.0,
        "temperature": 28.2,
        "flow_rate": 1.40
    }
]

def _load_real_data_scenarios() -> List[Dict[str, Any]]:
    """Builds one preset per sample type from the user's real bench readings (mean voltage & temperature)."""
    if not os.path.exists(REAL_DATA_PATH):
        return []
    try:
        df = prepare_sensor_frame(pd.read_csv(REAL_DATA_PATH))
    except Exception as e:
        print(f"[WARN] Could not load real sensor readings: {e}")
        return []
    if 'sample_type' not in df.columns:
        df['sample_type'] = "Bench sample"
    scenarios = []
    for sample_type, grp in df.groupby('sample_type', sort=False):
        v_in = float(grp['tds_in_voltage'].mean())
        v_out = float(grp['tds_out_voltage'].mean())
        temp = float(grp['temperature'].mean())
        slug = "".join(ch if ch.isalnum() else "_" for ch in str(sample_type).lower()).strip("_")
        scenarios.append({
            "id": f"real_{slug}",
            "name": f"{sample_type} (real bench data)",
            "category": "Real Data",
            "description": (
                f"Mean of {len(grp)} Serial Monitor readings: TDS1 {v_in:.3f} V, TDS2 {v_out:.3f} V, "
                f"{temp:.2f} °C. No flow sensor fitted."
            ),
            "source": "real",
            "tds_in_voltage": round(v_in, 4),
            "tds_out_voltage": round(v_out, 4),
            "tds_in": round(float(voltage_to_tds_ppm(v_in, temp)), 1),
            "tds_out": round(float(voltage_to_tds_ppm(v_out, temp)), 1),
            "temperature": round(temp, 2),
            "flow_rate": None
        })
    return scenarios

REAL_DATA_SCENARIOS = _load_real_data_scenarios()

def explain_prediction(status: str, rejection: float, flow: Optional[float], tds_out: float) -> str:
    """Provides plain human explainability for water engineers & users."""
    flow_known = flow is not None
    flow_txt = f" with healthy permeate flow ({flow:.2f} L/min)" if flow_known else " (flow not measured)"
    if status == "Normal":
        return f"Nominal operation: Salt rejection is high at {rejection:.1f}% (benchmark >88%){flow_txt}. Permeate quality looks good."
    elif status == "Monitor":
        if flow_known:
            return f"Preventive alert: Salt rejection has slipped to {rejection:.1f}% and flow is down to {flow:.2f} L/min. This indicates early foulant deposition or mineral scaling. Plan cartridge replacement in 2-4 weeks."
        return f"Preventive alert: Salt rejection is {rejection:.1f}% (below the 88% healthy benchmark). This indicates early foulant deposition, mineral scaling, or uncalibrated TDS probes. Verify probe calibration, then plan cartridge replacement in 2-4 weeks."
    else: # Service Required
        if tds_out >= 200.0:
            return f"Critical failure: High salt passage ({tds_out:.1f} ppm) and low rejection ({rejection:.1f}%). Membrane barrier integrity is compromised. Replace RO cartridge immediately."
        elif flow_known and flow < 0.8:
            return f"Severe blockage: Flow rate has dropped to {flow:.2f} L/min (<0.8 L/min limit) due to particulate head loss. Sediment/carbon stage requires replacement."
        if flow_known:
            return f"Service required: Combined performance indicators show severe degradation ({rejection:.1f}% rejection, {flow:.2f} L/min flow)."
        return f"Service required: Salt rejection has degraded to {rejection:.1f}%. Membrane performance is below the acceptable threshold."

def _get_predictor() -> FilterHealthPredictor:
    global predictor
    if predictor is None:
        try:
            predictor = FilterHealthPredictor(model_path=MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model could not be loaded: {str(e)}")
    return predictor

def _attach_explanation(result: Dict[str, Any], tds_out: float) -> Dict[str, Any]:
    phys = result['physics_indicators']
    result['explanation'] = explain_prediction(
        result['predicted_status'],
        phys['salt_rejection_pct'],
        phys['flow_rate_lpm'],
        tds_out
    )
    return result

@app.post("/api/predict")
def predict_filter_condition(inputs: SensorInput) -> Dict[str, Any]:
    """Predict from calibrated ppm values. flow_rate is optional."""
    result = _get_predictor().predict_sample(
        tds_in=inputs.tds_in,
        tds_out=inputs.tds_out,
        temperature=inputs.temperature,
        flow_rate=inputs.flow_rate
    )
    return _attach_explanation(result, inputs.tds_out)

@app.post("/api/predict-voltage")
def predict_from_voltage(inputs: VoltageSensorInput) -> Dict[str, Any]:
    """Predict from raw TDS probe voltages (as printed on the ESP32 Serial Monitor)."""
    result = _get_predictor().predict_from_voltage(
        tds_in_voltage=inputs.tds_in_voltage,
        tds_out_voltage=inputs.tds_out_voltage,
        temperature=inputs.temperature,
        flow_rate=inputs.flow_rate,
        k_in=inputs.k_in,
        k_out=inputs.k_out
    )
    return _attach_explanation(result, result['sensor_conversion']['tds_out_ppm'])

@app.get("/api/convert-voltage")
def convert_voltage(voltage: float, temperature: float = 25.0, k_value: float = DEFAULT_K_VALUE):
    """Convert a single TDS probe voltage to ppm."""
    if not (0 <= voltage <= 3.3):
        raise HTTPException(status_code=400, detail="voltage must be between 0 and 3.3 V")
    return {
        "voltage": voltage,
        "temperature": temperature,
        "k_value": k_value,
        "tds_ppm": round(float(voltage_to_tds_ppm(voltage, temperature, k_value)), 1)
    }

@app.get("/api/scenarios")
def get_scenarios():
    return REAL_DATA_SCENARIOS + PRESET_SCENARIOS

@app.get("/api/metrics")
def get_model_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Metrics file not found. Train model first."}

@app.get("/api/system-telemetry")
def get_system_telemetry():
    """Hardware node status, water purifier telemetry & life counters."""
    return {
        "hardware": {
            "microcontroller": "ESP32-S3 DevKit-C (Xtensa dual-core LX7)",
            "firmware_version": "v3.0.0-stage3",
            "connection_mode": "Active (USB Serial / UART @ 115200 baud)",
            "serial_output": "tds1_voltage,tds2_voltage,temperature[,flow_rate]",
            "sensors": [
                {"name": "Feed TDS", "sensor": "Xcluma TDS Meter V1.0", "channel": "ADC1 / GPIO 1", "status": "ONLINE", "output_unit": "V (converted to ppm server-side)"},
                {"name": "Permeate TDS", "sensor": "Xcluma TDS Meter V1.0", "channel": "ADC1 / GPIO 2", "status": "ONLINE", "output_unit": "V (converted to ppm server-side)"},
                {"name": "Temperature", "sensor": "DS18B20 1-Wire Probe", "channel": "GPIO 4", "status": "ONLINE", "output_unit": "°C"},
                {"name": "Permeate Flow", "sensor": "YF-S201 Hall Effect", "channel": "GPIO 5 (Pulse ISR)", "status": "NOT FITTED", "output_unit": "L/min"}
            ],
            "calibration": {
                "tds_k_value_in": DEFAULT_K_VALUE,
                "tds_k_value_out": DEFAULT_K_VALUE,
                "calibrated": False
            }
        },
        "purifier_stats": {
            "installation_date": "2026-09-15",
            "cumulative_volume_liters": 1845.0,
            "estimated_remaining_days": 38,
            "cartridge_model": "AquaSwachh RO-75GPD Polyamide Thin-Film Composite",
            "nominal_rated_capacity_liters": 6000.0,
            "filter_health_pct": 82
        }
    }

def _row_label(row: pd.Series, idx: int) -> str:
    if 'sample_type' in row and pd.notna(row.get('sample_type')):
        label = str(row['sample_type'])
        if 'reading_no' in row and pd.notna(row.get('reading_no')):
            label += f" #{int(row['reading_no'])}" if float(row['reading_no']).is_integer() else f" #{row['reading_no']}"
        return label
    if 'timestamp' in row and pd.notna(row.get('timestamp')):
        return str(row['timestamp'])
    return f"Row #{idx+1}"

def process_dataframe_inference(
    df: pd.DataFrame,
    filename: str,
    k_in: float = DEFAULT_K_VALUE,
    k_out: float = DEFAULT_K_VALUE
) -> Dict[str, Any]:
    """
    Batch inference on a sensor log. Accepts ppm columns (tds_in, tds_out) or
    raw voltage columns (tds_in_voltage/tds_out_voltage, or 'TDS Sensor 1 (V)' /
    'TDS Sensor 2 (V)'). flow_rate is optional, per row.
    """
    pred_engine = _get_predictor()

    try:
        df = prepare_sensor_frame(df, k_in=k_in, k_out=k_out)
    except SensorSchemaError as e:
        raise HTTPException(status_code=400, detail=str(e))
    input_mode = df.attrs.get("input_mode", "ppm")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV contains no valid data rows.")

    if len(df) > 1000:
        df = df.iloc[:1000]

    predictions = []
    statuses = []
    rejections = []
    variants = []

    for idx, row in df.iterrows():
        flow = row['flow_rate']
        flow = None if pd.isna(flow) else float(flow)
        pred = pred_engine.predict_sample(
            tds_in=float(row['tds_in']),
            tds_out=float(row['tds_out']),
            temperature=float(row['temperature']),
            flow_rate=flow
        )
        statuses.append(pred['predicted_status'])
        rejections.append(pred['physics_indicators']['salt_rejection_pct'])
        variants.append(pred['model_variant'])

        if idx < 50:
            entry = {
                "row_index": idx + 1,
                "timestamp": _row_label(row, idx),
                "sample_type": str(row['sample_type']) if 'sample_type' in df.columns and pd.notna(row.get('sample_type')) else None,
                "tds_in": round(float(row['tds_in']), 1),
                "tds_out": round(float(row['tds_out']), 1),
                "temperature": round(float(row['temperature']), 2),
                "flow_rate": round(flow, 2) if flow is not None else None,
                "salt_rejection": pred['physics_indicators']['salt_rejection_pct'],
                "predicted_status": pred['predicted_status'],
                "confidence": round(pred['confidence'] * 100, 1),
                "guardrail_triggered": pred['guardrail_triggered'],
                "model_variant": pred['model_variant'],
                "flow_simulated": bool(
                    flow is not None and 'flow_source' in df.columns
                    and 'simulat' in str(row.get('flow_source', '')).lower()
                )
            }
            if input_mode == "voltage":
                entry["tds_in_voltage"] = round(float(row['tds_in_voltage']), 4)
                entry["tds_out_voltage"] = round(float(row['tds_out_voltage']), 4)
            predictions.append(entry)

    df = df.assign(predicted_status=statuses, salt_rejection_pct=rejections)
    status_counts = pd.Series(statuses).value_counts().to_dict()
    total_rows = len(statuses)
    flow_rows = int(df['flow_rate'].notna().sum())
    simulated_flow_rows = 0
    if 'flow_source' in df.columns:
        simulated_flow_rows = int(
            (df['flow_rate'].notna() & df['flow_source'].astype(str).str.lower().str.contains('simulat')).sum()
        )

    by_sample_type = []
    if 'sample_type' in df.columns:
        for sample_type, grp in df.groupby('sample_type', sort=False):
            group_counts = grp['predicted_status'].value_counts().to_dict()
            summary = {
                "sample_type": str(sample_type),
                "records": int(len(grp)),
                "avg_tds_in_ppm": round(float(grp['tds_in'].mean()), 1),
                "avg_tds_out_ppm": round(float(grp['tds_out'].mean()), 1),
                "avg_temperature_c": round(float(grp['temperature'].mean()), 2),
                "temperature_range_c": [round(float(grp['temperature'].min()), 2), round(float(grp['temperature'].max()), 2)],
                "avg_salt_rejection_pct": round(float(grp['salt_rejection_pct'].mean()), 2),
                "majority_status": max(group_counts, key=group_counts.get),
                "status_counts": {k: int(v) for k, v in group_counts.items()}
            }
            if input_mode == "voltage":
                summary["avg_tds_in_voltage"] = round(float(grp['tds_in_voltage'].mean()), 4)
                summary["avg_tds_out_voltage"] = round(float(grp['tds_out_voltage'].mean()), 4)
            by_sample_type.append(summary)

    return {
        "filename": filename,
        "input_mode": input_mode,
        "calibration": {"k_in": k_in, "k_out": k_out},
        "flow_available": flow_rows > 0,
        "flow_rows": flow_rows,
        "simulated_flow_rows": simulated_flow_rows,
        "model_variants_used": {k: int(v) for k, v in pd.Series(variants).value_counts().to_dict().items()},
        "by_sample_type": by_sample_type,
        "total_records": total_rows,
        "breakdown": {
            "Normal": status_counts.get("Normal", 0),
            "Monitor": status_counts.get("Monitor", 0),
            "Service Required": status_counts.get("Service Required", 0)
        },
        "percentages": {
            "Normal": round((status_counts.get("Normal", 0) / total_rows) * 100, 1),
            "Monitor": round((status_counts.get("Monitor", 0) / total_rows) * 100, 1),
            "Service Required": round((status_counts.get("Service Required", 0) / total_rows) * 100, 1)
        },
        "avg_salt_rejection_pct": round(float(pd.Series(rejections).mean()), 2),
        "avg_flow_rate_lpm": round(float(df['flow_rate'].mean()), 2) if flow_rows > 0 else None,
        "rows_preview": predictions
    }

@app.post("/api/upload-csv")
async def analyze_uploaded_csv(
    file: UploadFile = File(...),
    k_in: float = DEFAULT_K_VALUE,
    k_out: float = DEFAULT_K_VALUE
):
    """
    Analyzes real-world sensor logs uploaded as a CSV.
    Voltage CSVs are converted to ppm using calibration K-values k_in / k_out (query params).
    """
    if not (0 < k_in <= 5 and 0 < k_out <= 5):
        raise HTTPException(status_code=400, detail="k_in and k_out must be in (0, 5]")
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8-sig', errors='ignore')))
        return process_dataframe_inference(df, file.filename, k_in=k_in, k_out=k_out)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing CSV file: {str(e)}")

@app.get("/api/load-sample-csv")
def load_sample_csv():
    """Loads bundled sensor_data.csv and returns ML inference results instantly."""
    sample_path = "data/sensor_data.csv"
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample dataset data/sensor_data.csv not found")
    df = pd.read_csv(sample_path)
    return process_dataframe_inference(df, "sensor_data.csv (Synthetic Benchmark Dataset)")

@app.get("/api/load-real-data")
def load_real_data(
    k_in: float = DEFAULT_K_VALUE,
    k_out: float = DEFAULT_K_VALUE,
    with_simulated_flow: bool = False
):
    """
    Runs inference on the bundled real bench readings (raw voltages from the ESP32 Serial Monitor).
    with_simulated_flow=true uses the same readings plus SIMULATED flow values (demo only;
    no flow meter is fitted, so these flow values are not measurements).
    """
    path = SIMULATED_FLOW_DATA_PATH if with_simulated_flow else REAL_DATA_PATH
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Dataset not found at {path}")
    if not (0 < k_in <= 5 and 0 < k_out <= 5):
        raise HTTPException(status_code=400, detail="k_in and k_out must be in (0, 5]")
    df = pd.read_csv(path)
    label = (
        "real_readings_with_simulated_flow.csv (Real Voltages + SIMULATED Flow)"
        if with_simulated_flow else
        "real_sensor_readings.csv (Real Bench Readings, Voltage)"
    )
    return process_dataframe_inference(df, label, k_in=k_in, k_out=k_out)

@app.get("/api/download-real-csv")
def download_real_csv():
    """Downloads the real bench readings CSV (voltage format template)."""
    if not os.path.exists(REAL_DATA_PATH):
        raise HTTPException(status_code=404, detail="Real sensor readings not found")
    return FileResponse(REAL_DATA_PATH, filename="real_sensor_readings.csv", media_type="text/csv")

@app.get("/api/download-sample-csv")
def download_sample_csv():
    """Downloads sample sensor_data.csv."""
    sample_path = "data/sensor_data.csv"
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample dataset not found")
    return FileResponse(sample_path, filename="sensor_data.csv", media_type="text/csv")

@app.get("/api/real-readings")
def get_real_readings():
    """Returns the raw real bench readings (voltages + temperature) as JSON, e.g. for replay in the dashboard."""
    if not os.path.exists(REAL_DATA_PATH):
        raise HTTPException(status_code=404, detail=f"Real sensor readings not found at {REAL_DATA_PATH}")
    df = pd.read_csv(REAL_DATA_PATH)
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict(orient="records")


@app.get("/api/stream-sample")
def get_stream_sample(state: Optional[str] = "Normal"):
    """Returns realistic sensor readings with continuous physical drift."""
    if state == "Normal":
        t_in = 420 + random.uniform(-15, 25)
        rejection = random.uniform(0.92, 0.96)
        temp = 25.5 + random.uniform(-0.5, 0.5)
        flow = 1.90 + random.uniform(-0.08, 0.08)
        t_out = t_in * (1 - rejection)
    elif state == "Monitor":
        t_in = 470 + random.uniform(-20, 25)
        rejection = random.uniform(0.78, 0.84)
        temp = 26.2 + random.uniform(-0.6, 0.6)
        flow = 1.28 + random.uniform(-0.06, 0.06)
        t_out = t_in * (1 - rejection)
    else: # Service Required
        t_in = 520 + random.uniform(-20, 30)
        rejection = random.uniform(0.55, 0.68)
        temp = 27.5 + random.uniform(-0.8, 0.8)
        flow = 0.62 + random.uniform(-0.06, 0.06)
        t_out = t_in * (1 - rejection)
        
    return {
        "tds_in": round(t_in, 1),
        "tds_out": round(t_out, 1),
        "temperature": round(temp, 1),
        "flow_rate": round(flow, 2)
    }

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Aqua-Swachh API running. Static index.html not yet created."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("web.app:app", host="0.0.0.0", port=port, reload=False)
