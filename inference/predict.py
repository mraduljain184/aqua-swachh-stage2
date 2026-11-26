"""
Aqua-Swachh Real-Time Filter Condition Inference Engine
=======================================================
Provides high-level inference for single or streaming sensor readings.
Includes:
- Raw TDS probe voltage -> ppm conversion (see model/sensor_calibration.py)
- Physics feature extraction
- Model prediction with probability confidence scores
  (automatically uses the no-flow model when flow_rate is not available)
- Deterministic safety guardrails (failsafe rules)
- User-friendly diagnostic messaging
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any, Optional

from model.features import extract_features, compute_physics_features
from model.sensor_calibration import voltage_to_tds_ppm, is_missing, DEFAULT_K_VALUE

DEFAULT_MODEL_PATH = "model/saved_models/random_forest_filter_model.joblib"
DEFAULT_NO_FLOW_MODEL_PATH = "model/saved_models/random_forest_filter_model_no_flow.joblib"

FLOW_UNAVAILABLE_NOTE = (
    "No flow-meter reading: diagnosis is based on TDS and temperature only. "
    "Pre-filter clogging / flow choke cannot be detected without a flow sensor."
)


class FilterHealthPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, no_flow_model_path: Optional[str] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Please run `python -m model.train` to train and save the model first."
            )
        self.model = joblib.load(model_path)
        self.classes_ = list(self.model.classes_)

        if no_flow_model_path is None:
            no_flow_model_path = os.path.join(
                os.path.dirname(model_path), os.path.basename(DEFAULT_NO_FLOW_MODEL_PATH)
            )
        self.no_flow_model = joblib.load(no_flow_model_path) if os.path.exists(no_flow_model_path) else None
        self.no_flow_model_path = no_flow_model_path

    def predict_sample(
        self,
        tds_in: float,
        tds_out: float,
        temperature: float,
        flow_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Predicts filter health for a single sensor reading snapshot (ppm inputs).
        `flow_rate` is optional; pass None when no flow meter is fitted.
        """
        flow_available = not is_missing(flow_rate)
        flow_value = float(flow_rate) if flow_available else None

        if flow_available:
            model = self.model
        else:
            if self.no_flow_model is None:
                raise FileNotFoundError(
                    f"No-flow model not found at '{self.no_flow_model_path}'. "
                    "Run `python -m model.train` to train it."
                )
            model = self.no_flow_model

        raw_df = pd.DataFrame([{
            'tds_in': float(tds_in),
            'tds_out': float(tds_out),
            'temperature': float(temperature),
            'flow_rate': flow_value
        }])

        # 1. Physics Features
        physics_df = compute_physics_features(raw_df)
        rejection_pct = float(physics_df['salt_rejection'].iloc[0]) * 100.0
        temp_comp_rejection_pct = float(physics_df['temp_comp_rejection'].iloc[0]) * 100.0
        delta_tds = float(physics_df['delta_tds'].iloc[0])

        # 2. Extract model features and predict
        X = extract_features(raw_df, include_flow=flow_available)
        predicted_status = model.predict(X)[0]
        model_status = str(predicted_status)
        probabilities = model.predict_proba(X)[0]
        prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(model.classes_, probabilities)}

        # 3. Deterministic Safety Guardrails (Fail-Safe Checks)
        # Prevents any false-negative "Normal" if severe physical contamination or flow blockage is detected
        guardrail_triggered = False
        guardrail_reason = ""

        if tds_out >= 250.0:
            guardrail_triggered = True
            guardrail_reason = f"High permeate TDS detected ({tds_out:.1f} ppm >= 250 ppm safety ceiling)"
            predicted_status = "Service Required"
        elif rejection_pct < 55.0:
            guardrail_triggered = True
            guardrail_reason = f"Severe salt rejection loss ({rejection_pct:.1f}% < 55% threshold)"
            predicted_status = "Service Required"
        elif flow_available and flow_value < 0.40 and tds_in > 0:
            guardrail_triggered = True
            guardrail_reason = f"Severe flow choke / membrane clogging ({flow_value:.2f} L/min < 0.40 L/min)"
            predicted_status = "Service Required"

        # 4. User Action & Diagnosis Recommendation
        action_recommendation = self._get_recommendation(predicted_status, flow_available)

        flow_index = None
        if flow_available:
            flow_index = round(flow_value * max(rejection_pct / 100.0, 0.0), 3)

        return {
            'predicted_status': predicted_status,
            'confidence': float(max(probabilities)),
            # What the ML model said before any safety-rule override
            'model_status': model_status,
            'probabilities': prob_dict,
            'guardrail_triggered': guardrail_triggered,
            'guardrail_reason': guardrail_reason,
            'flow_available': flow_available,
            'model_variant': 'full' if flow_available else 'no_flow',
            'notes': [] if flow_available else [FLOW_UNAVAILABLE_NOTE],
            'physics_indicators': {
                'salt_rejection_pct': round(rejection_pct, 2),
                'temp_compensated_rejection_pct': round(temp_comp_rejection_pct, 2),
                'delta_tds_ppm': round(delta_tds, 1),
                'water_temperature_c': round(float(temperature), 1),
                'flow_rate_lpm': round(flow_value, 2) if flow_available else None,
                'flow_rejection_index': flow_index
            },
            'action_recommendation': action_recommendation
        }

    def predict_from_voltage(
        self,
        tds_in_voltage: float,
        tds_out_voltage: float,
        temperature: float,
        flow_rate: Optional[float] = None,
        k_in: float = DEFAULT_K_VALUE,
        k_out: float = DEFAULT_K_VALUE
    ) -> Dict[str, Any]:
        """
        Predicts filter health from raw analog TDS probe voltages
        (TDS Sensor 1 = feed / tds_in, TDS Sensor 2 = permeate / tds_out).
        """
        tds_in = round(float(voltage_to_tds_ppm(tds_in_voltage, temperature, k_in)), 1)
        tds_out = round(float(voltage_to_tds_ppm(tds_out_voltage, temperature, k_out)), 1)
        result = self.predict_sample(tds_in, tds_out, temperature, flow_rate)
        result['sensor_conversion'] = {
            'tds_in_voltage': round(float(tds_in_voltage), 4),
            'tds_out_voltage': round(float(tds_out_voltage), 4),
            'tds_in_ppm': tds_in,
            'tds_out_ppm': tds_out,
            'k_in': k_in,
            'k_out': k_out,
            'calibrated': not (k_in == DEFAULT_K_VALUE and k_out == DEFAULT_K_VALUE),
            'method': 'Gravity TDS polynomial with 2%/°C temperature compensation'
        }
        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs batch inference on a pandas DataFrame with ppm columns
        (tds_in, tds_out, temperature, optional flow_rate).
        """
        results = []
        for _, row in df.iterrows():
            res = self.predict_sample(
                tds_in=row['tds_in'],
                tds_out=row['tds_out'],
                temperature=row['temperature'],
                flow_rate=row.get('flow_rate')
            )
            results.append({
                'predicted_status': res['predicted_status'],
                'confidence': res['confidence'],
                'salt_rejection_pct': res['physics_indicators']['salt_rejection_pct'],
                'guardrail_triggered': res['guardrail_triggered'],
                'model_variant': res['model_variant']
            })
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    @staticmethod
    def _get_recommendation(status: str, flow_available: bool = True) -> str:
        if status == "Normal":
            if flow_available:
                return "Filter operating within nominal parameters. Water quality and flow are optimal."
            return "Salt rejection and permeate TDS are within nominal parameters. (Flow not measured.)"
        elif status == "Monitor":
            return "Early degradation or fouling detected. Purifier is functional; prepare for cartridge swap in coming cycles."
        elif status == "Service Required":
            return "Filter condition degraded beyond acceptable threshold. Cartridge service/replacement required."
        return "Unknown condition."


if __name__ == "__main__":
    predictor = FilterHealthPredictor()

    # Test cases representing three operational states
    test_samples = [
        {"name": "Fresh Filter", "tds_in": 450, "tds_out": 30, "temperature": 26.0, "flow_rate": 1.9},
        {"name": "Aging Filter", "tds_in": 480, "tds_out": 105, "temperature": 27.2, "flow_rate": 1.3},
        {"name": "Clogged / Damaged Filter", "tds_in": 520, "tds_out": 220, "temperature": 28.0, "flow_rate": 0.6}
    ]

    print("\n--- Model Inference Test Cases (ppm + flow) ---")
    for sample in test_samples:
        pred = predictor.predict_sample(
            tds_in=sample["tds_in"],
            tds_out=sample["tds_out"],
            temperature=sample["temperature"],
            flow_rate=sample["flow_rate"]
        )
        print(f"\nCase: {sample['name']}")
        print(f"Inputs: TDS In={sample['tds_in']}, TDS Out={sample['tds_out']}, Temp={sample['temperature']}°C, Flow={sample['flow_rate']} L/min")
        print(f"Prediction: {pred['predicted_status']} (Confidence: {pred['confidence']*100:.1f}%)")
        print(f"Rejection Efficiency: {pred['physics_indicators']['salt_rejection_pct']}%")
        print(f"Action: {pred['action_recommendation']}")

    print("\n--- Raw voltage reading (no flow sensor) ---")
    pred = predictor.predict_from_voltage(tds_in_voltage=1.214, tds_out_voltage=0.144, temperature=24.81)
    conv = pred['sensor_conversion']
    print(f"TDS1={conv['tds_in_voltage']} V -> {conv['tds_in_ppm']} ppm | TDS2={conv['tds_out_voltage']} V -> {conv['tds_out_ppm']} ppm")
    print(f"Prediction: {pred['predicted_status']} (Confidence: {pred['confidence']*100:.1f}%, model={pred['model_variant']})")
    print(f"Rejection Efficiency: {pred['physics_indicators']['salt_rejection_pct']}%")
