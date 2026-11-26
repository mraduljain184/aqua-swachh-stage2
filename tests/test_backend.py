"""
Aqua-Swachh: Comprehensive Backend Verification Suite
======================================================
Tests:
1. Physics feature engineering (salt rejection, temp compensation, flow index, delta TDS)
2. TDS probe voltage -> ppm conversion & CSV column normalization (real bench data)
3. Predictor model loading, single-sample inference, probability distribution
4. No-flow inference path (no flow meter fitted)
5. Deterministic safety guardrail overrides (high permeate TDS & low salt rejection)
6. Live REST API endpoints via HTTP against running server
   (start it first: uvicorn web.app:app --port 8000)
"""

import unittest
import json
import urllib.request
import urllib.parse
import pandas as pd
from model.features import (
    compute_physics_features,
    extract_features,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_NO_FLOW
)
from model.sensor_calibration import (
    voltage_to_tds_ppm,
    prepare_sensor_frame,
    SensorSchemaError
)
from inference.predict import FilterHealthPredictor

class TestPhysicsFeatures(unittest.TestCase):
    def test_salt_rejection_calculation(self):
        raw_df = pd.DataFrame([{
            'tds_in': 420.0,
            'tds_out': 28.0,
            'temperature': 25.0,
            'flow_rate': 1.95
        }])
        phys = compute_physics_features(raw_df)
        rejection = phys['salt_rejection'].iloc[0]
        self.assertAlmostEqual(rejection, 1.0 - (28.0 / 420.0), places=4)

    def test_salt_rejection_zero_feed(self):
        # Guard against division by zero
        raw_df = pd.DataFrame([{
            'tds_in': 0.0,
            'tds_out': 0.0,
            'temperature': 25.0,
            'flow_rate': 1.0
        }])
        phys = compute_physics_features(raw_df)
        self.assertFalse(pd.isna(phys['salt_rejection'].iloc[0]))

    def test_temperature_compensation(self):
        raw_df = pd.DataFrame([{
            'tds_in': 500.0,
            'tds_out': 50.0,
            'temperature': 25.0,
            'flow_rate': 2.0
        }])
        phys = compute_physics_features(raw_df)
        self.assertAlmostEqual(phys['tds_out_norm_25c'].iloc[0], 50.0, places=4)

    def test_delta_tds(self):
        raw_df = pd.DataFrame([{
            'tds_in': 500.0,
            'tds_out': 50.0,
            'temperature': 25.0,
            'flow_rate': 2.0
        }])
        phys = compute_physics_features(raw_df)
        self.assertEqual(phys['delta_tds'].iloc[0], 450.0)

    def test_feature_dataframe_structure(self):
        raw_df = pd.DataFrame([{
            'tds_in': 420.0,
            'tds_out': 28.0,
            'temperature': 25.5,
            'flow_rate': 1.95
        }])
        feats = extract_features(raw_df)
        self.assertEqual(list(feats.columns), FEATURE_COLUMNS)
        self.assertEqual(len(feats), 1)

    def test_feature_dataframe_without_flow(self):
        raw_df = pd.DataFrame([{'tds_in': 453.2, 'tds_out': 59.5, 'temperature': 24.81}])
        feats = extract_features(raw_df, include_flow=False)
        self.assertEqual(list(feats.columns), FEATURE_COLUMNS_NO_FLOW)
        self.assertNotIn('flow_rate', feats.columns)
        self.assertFalse(feats.isnull().any().any())


class TestSensorCalibration(unittest.TestCase):
    def test_voltage_to_ppm_reference_point(self):
        # Real bench reading: TDS Sensor 1 = 1.214 V at 24.81 °C
        self.assertAlmostEqual(voltage_to_tds_ppm(1.214, 24.81), 453.2, delta=0.1)
        self.assertAlmostEqual(voltage_to_tds_ppm(0.144, 24.81), 59.5, delta=0.1)

    def test_voltage_to_ppm_temperature_compensation(self):
        # Same voltage at higher temperature -> lower 25 °C-equivalent TDS
        self.assertLess(voltage_to_tds_ppm(1.214, 32.0), voltage_to_tds_ppm(1.214, 25.0))

    def test_voltage_to_ppm_k_value_scales_linearly(self):
        base = voltage_to_tds_ppm(1.0, 25.0)
        self.assertAlmostEqual(voltage_to_tds_ppm(1.0, 25.0, k_value=1.1), base * 1.1, places=6)

    def test_zero_voltage_is_zero_ppm(self):
        self.assertEqual(voltage_to_tds_ppm(0.0, 25.0), 0.0)

    def test_prepare_frame_accepts_spreadsheet_headers(self):
        df = pd.DataFrame({
            'Sample Type': ['Normal water', 'Water cooler water'],
            'Reading No.': [1, 1],
            'TDS Sensor 1 (V)': [1.214, 1.213],
            'TDS Sensor 2 (V)': [0.144, 0.150],
            'Temperature (°C)': [24.81, 31.12],
            'Source': ['Serial Monitor screenshot'] * 2,
        })
        out = prepare_sensor_frame(df)
        self.assertEqual(out.attrs['input_mode'], 'voltage')
        for col in ('tds_in', 'tds_out', 'temperature', 'flow_rate', 'sample_type', 'reading_no'):
            self.assertIn(col, out.columns)
        self.assertTrue(out['flow_rate'].isna().all())
        self.assertAlmostEqual(out['tds_in'].iloc[0], 453.2, delta=0.1)

    def test_prepare_frame_ppm_passthrough(self):
        df = pd.DataFrame({'tds_in': [420.0], 'tds_out': [28.0], 'temperature': [25.5], 'flow_rate': [1.95]})
        out = prepare_sensor_frame(df)
        self.assertEqual(out.attrs['input_mode'], 'ppm')
        self.assertEqual(out['tds_in'].iloc[0], 420.0)
        self.assertEqual(out['flow_rate'].iloc[0], 1.95)

    def test_prepare_frame_rejects_unknown_schema(self):
        with self.assertRaises(SensorSchemaError):
            prepare_sensor_frame(pd.DataFrame({'foo': [1], 'temperature': [25]}))

    def test_real_readings_file_loads(self):
        df = prepare_sensor_frame(pd.read_csv("data/real_sensor_readings.csv"))
        self.assertEqual(len(df), 18)
        self.assertEqual(set(df['sample_type']), {'Normal water', 'Water cooler water'})


class TestInferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = FilterHealthPredictor(model_path="model/saved_models/random_forest_filter_model.joblib")

    def test_nominal_sample(self):
        result = self.predictor.predict_sample(tds_in=420.0, tds_out=28.0, temperature=25.5, flow_rate=1.95)
        self.assertEqual(result["predicted_status"], "Normal")
        self.assertGreaterEqual(result["confidence"], 0.80)
        self.assertIn("Normal", result["probabilities"])
        self.assertIn("Monitor", result["probabilities"])
        self.assertIn("Service Required", result["probabilities"])
        self.assertFalse(result["guardrail_triggered"])

    def test_degraded_sample(self):
        result = self.predictor.predict_sample(tds_in=510.0, tds_out=115.0, temperature=26.8, flow_rate=1.25)
        self.assertEqual(result["predicted_status"], "Monitor")

    def test_clogged_flow_sample(self):
        result = self.predictor.predict_sample(tds_in=480.0, tds_out=120.0, temperature=26.0, flow_rate=0.65)
        self.assertEqual(result["predicted_status"], "Service Required")

    def test_safety_guardrail_high_permeate_tds(self):
        # Permeate TDS >= 250 ppm triggers failsafe rule
        result = self.predictor.predict_sample(tds_in=600.0, tds_out=260.0, temperature=25.0, flow_rate=1.8)
        self.assertEqual(result["predicted_status"], "Service Required")
        self.assertTrue(result["guardrail_triggered"])
        self.assertIn("permeate tds", result["guardrail_reason"].lower())

    def test_safety_guardrail_low_salt_rejection(self):
        # Salt rejection < 55% triggers failsafe rule
        result = self.predictor.predict_sample(tds_in=300.0, tds_out=150.0, temperature=25.0, flow_rate=1.5)
        self.assertEqual(result["predicted_status"], "Service Required")
        self.assertTrue(result["guardrail_triggered"])

    def test_full_model_used_when_flow_present(self):
        result = self.predictor.predict_sample(tds_in=420.0, tds_out=28.0, temperature=25.5, flow_rate=1.95)
        self.assertTrue(result["flow_available"])
        self.assertEqual(result["model_variant"], "full")

    def test_no_flow_prediction(self):
        result = self.predictor.predict_sample(tds_in=420.0, tds_out=28.0, temperature=25.5, flow_rate=None)
        self.assertFalse(result["flow_available"])
        self.assertEqual(result["model_variant"], "no_flow")
        self.assertIsNone(result["physics_indicators"]["flow_rate_lpm"])
        self.assertEqual(result["predicted_status"], "Normal")
        self.assertTrue(result["notes"])

    def test_nan_flow_treated_as_missing(self):
        result = self.predictor.predict_sample(tds_in=420.0, tds_out=28.0, temperature=25.5, flow_rate=float('nan'))
        self.assertEqual(result["model_variant"], "no_flow")

    def test_no_flow_guardrail_still_active(self):
        result = self.predictor.predict_sample(tds_in=600.0, tds_out=260.0, temperature=25.0)
        self.assertEqual(result["predicted_status"], "Service Required")
        self.assertTrue(result["guardrail_triggered"])

    def test_predict_from_voltage(self):
        result = self.predictor.predict_from_voltage(tds_in_voltage=1.214, tds_out_voltage=0.144, temperature=24.81)
        conv = result["sensor_conversion"]
        self.assertAlmostEqual(conv["tds_in_ppm"], 453.2, delta=0.1)
        self.assertAlmostEqual(conv["tds_out_ppm"], 59.5, delta=0.1)
        self.assertFalse(conv["calibrated"])
        self.assertEqual(result["model_variant"], "no_flow")
        self.assertIn(result["predicted_status"], ["Normal", "Monitor", "Service Required"])

    def test_batch_prediction_on_real_readings(self):
        df = prepare_sensor_frame(pd.read_csv("data/real_sensor_readings.csv"))
        out = self.predictor.predict_batch(df)
        self.assertEqual(len(out), 18)
        self.assertTrue((out['model_variant'] == 'no_flow').all())


class TestLiveApiEndpoints(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:8000"

    def test_predict_api(self):
        url = f"{self.BASE_URL}/api/predict"
        payload = json.dumps({
            "tds_in": 420.0,
            "tds_out": 28.0,
            "temperature": 25.5,
            "flow_rate": 1.95
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["predicted_status"], "Normal")
            self.assertIn("physics_indicators", data)
            self.assertIn("explanation", data)

    def test_predict_api_without_flow(self):
        url = f"{self.BASE_URL}/api/predict"
        payload = json.dumps({"tds_in": 420.0, "tds_out": 28.0, "temperature": 25.5}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["model_variant"], "no_flow")
            self.assertIsNone(data["physics_indicators"]["flow_rate_lpm"])

    def test_predict_voltage_api(self):
        url = f"{self.BASE_URL}/api/predict-voltage"
        payload = json.dumps({"tds_in_voltage": 1.214, "tds_out_voltage": 0.144, "temperature": 24.81}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("sensor_conversion", data)
            self.assertAlmostEqual(data["sensor_conversion"]["tds_in_ppm"], 453.2, delta=0.1)
            self.assertIn("explanation", data)

    def test_load_real_data_api(self):
        url = f"{self.BASE_URL}/api/load-real-data"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["total_records"], 18)
            self.assertEqual(data["input_mode"], "voltage")
            self.assertFalse(data["flow_available"])
            self.assertEqual(len(data["by_sample_type"]), 2)
            self.assertIn("tds_in_voltage", data["rows_preview"][0])

    def test_load_real_data_with_simulated_flow_api(self):
        url = f"{self.BASE_URL}/api/load-real-data?with_simulated_flow=true"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["total_records"], 18)
            self.assertTrue(data["flow_available"])
            self.assertEqual(data["simulated_flow_rows"], 18)
            self.assertTrue(all(r["flow_simulated"] for r in data["rows_preview"]))
            # Flow drives the outcome: both Monitor and Service Required should appear
            self.assertGreater(data["breakdown"]["Monitor"], 0)
            self.assertGreater(data["breakdown"]["Service Required"], 0)
            confidences = {r["confidence"] for r in data["rows_preview"]}
            self.assertGreater(len(confidences), 2)

    def test_scenarios_api(self):
        url = f"{self.BASE_URL}/api/scenarios"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            scenarios = json.loads(resp.read().decode("utf-8"))
            self.assertGreaterEqual(len(scenarios), 5)

    def test_metrics_api(self):
        url = f"{self.BASE_URL}/api/metrics"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            metrics = json.loads(resp.read().decode("utf-8"))
            self.assertIn("test_accuracy", metrics)
            self.assertIn("comparison", metrics)
            self.assertIn("Random Forest Classifier (Primary Model)", metrics["comparison"])


    def test_system_telemetry_api(self):
        url = f"{self.BASE_URL}/api/system-telemetry"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("hardware", data)
            self.assertIn("purifier_stats", data)

    def test_load_sample_csv_api(self):
        url = f"{self.BASE_URL}/api/load-sample-csv"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertGreaterEqual(data["total_records"], 100)
            self.assertIn("Normal", data["breakdown"])
            self.assertIn("Monitor", data["breakdown"])
            self.assertIn("Service Required", data["breakdown"])
            self.assertIn("rows_preview", data)

    def test_stream_sample_api(self):
        url = f"{self.BASE_URL}/api/stream-sample?state=Normal"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("tds_in", data)
            self.assertIn("tds_out", data)
            self.assertIn("temperature", data)
            self.assertIn("flow_rate", data)

if __name__ == "__main__":
    unittest.main()
