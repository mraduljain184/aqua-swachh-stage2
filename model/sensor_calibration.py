"""
Aqua-Swachh Sensor Calibration & Input Normalization
====================================================
The analog TDS boards (Xcluma / Gravity-compatible) output a voltage, not ppm.
This module converts raw sensor voltage into TDS ppm and normalizes CSV logs
coming from different sources (ESP32 serial logger, hand-transcribed Serial
Monitor readings, spreadsheets) into the schema the ML pipeline expects:

    tds_in (ppm), tds_out (ppm), temperature (°C), flow_rate (L/min, optional)

Conversion (identical to the Gravity/DFRobot reference used in the firmware):

    V_25   = V / (1 + 0.02 * (T - 25))                      # temperature compensation
    TDS    = (133.42*V_25^3 - 255.86*V_25^2 + 857.39*V_25) * 0.5 * K

K is the per-probe calibration factor ("K-value"). It defaults to 1.0, i.e.
uncalibrated. Calibrate each probe against a known standard solution
(e.g. 342 ppm NaCl buffer) and pass the resulting K to get accurate ppm.
"""

import re
from typing import Optional, Union

import numpy as np
import pandas as pd

ArrayLike = Union[float, np.ndarray, pd.Series]

# Gravity TDS cubic polynomial coefficients (ppm per V^n, before the 0.5 factor)
_POLY_A3 = 133.42
_POLY_A2 = -255.86
_POLY_A1 = 857.39

# Physical limits of the ESP32 ADC (11 dB attenuation, 3.3 V reference)
ADC_MAX_VOLTAGE = 3.3

DEFAULT_K_VALUE = 1.0


def voltage_to_tds_ppm(
    voltage: ArrayLike,
    temperature_c: ArrayLike = 25.0,
    k_value: float = DEFAULT_K_VALUE,
) -> ArrayLike:
    """Convert analog TDS probe voltage to temperature-compensated TDS in ppm."""
    v = np.asarray(voltage, dtype=float)
    t = np.asarray(temperature_c, dtype=float)
    comp = np.maximum(1.0 + 0.02 * (t - 25.0), 0.5)
    v25 = v / comp
    tds = (_POLY_A3 * v25 ** 3 + _POLY_A2 * v25 ** 2 + _POLY_A1 * v25) * 0.5 * k_value
    tds = np.maximum(tds, 0.0)
    if np.ndim(tds) == 0:
        return float(tds)
    if isinstance(voltage, pd.Series):
        return pd.Series(tds, index=voltage.index)
    return tds


# --------------------------------------------------------------------------- #
# Column normalization
# --------------------------------------------------------------------------- #

# Canonical name -> accepted aliases (already slugified, see _slug()).
COLUMN_ALIASES = {
    "tds_in_voltage": [
        "tds_in_voltage", "tds_in_v", "tds_in_volt", "tds1_v", "tds1_voltage",
        "tds_1_v", "tds_sensor_1_v", "tds_sensor_1_voltage", "tds_sensor_1",
        "feed_tds_v", "feed_voltage",
    ],
    "tds_out_voltage": [
        "tds_out_voltage", "tds_out_v", "tds_out_volt", "tds2_v", "tds2_voltage",
        "tds_2_v", "tds_sensor_2_v", "tds_sensor_2_voltage", "tds_sensor_2",
        "permeate_tds_v", "permeate_voltage",
    ],
    "tds_in": ["tds_in", "tds_in_ppm", "feed_tds", "feed_tds_ppm"],
    "tds_out": ["tds_out", "tds_out_ppm", "permeate_tds", "permeate_tds_ppm"],
    "temperature": [
        "temperature", "temperature_c", "temp", "temp_c", "water_temperature",
        "water_temp",
    ],
    "flow_rate": [
        "flow_rate", "flow", "flow_lpm", "flow_rate_lpm", "flow_rate_l_min",
        "flow_l_min",
    ],
    "sample_type": ["sample_type", "sample", "sample_name", "water_type"],
    "reading_no": ["reading_no", "reading", "reading_number", "reading_id"],
    "timestamp": ["timestamp", "time", "datetime", "date_time"],
}

_ALIAS_LOOKUP = {alias: canon for canon, aliases in COLUMN_ALIASES.items() for alias in aliases}


def _slug(name: str) -> str:
    """'TDS Sensor 1 (V)' -> 'tds_sensor_1_v', 'Temperature (°C)' -> 'temperature_c'."""
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column aliases to canonical names. Unknown columns are kept (slugified)."""
    renamed = {}
    used = set()
    for col in df.columns:
        slug = _slug(col)
        canon = _ALIAS_LOOKUP.get(slug, slug)
        if canon in used:  # keep the first occurrence only
            canon = slug + "_dup"
        used.add(canon)
        renamed[col] = canon
    return df.rename(columns=renamed)


class SensorSchemaError(ValueError):
    pass


def prepare_sensor_frame(
    df: pd.DataFrame,
    k_in: float = DEFAULT_K_VALUE,
    k_out: float = DEFAULT_K_VALUE,
) -> pd.DataFrame:
    """
    Normalize a raw sensor log into the model schema.

    Accepts either:
      * ppm columns  : tds_in, tds_out, temperature [, flow_rate]
      * voltage cols : tds_in_voltage, tds_out_voltage, temperature [, flow_rate]

    Returns a DataFrame that always has tds_in, tds_out, temperature and
    flow_rate (NaN where no flow sensor data exists), plus any metadata
    columns (sample_type, reading_no, timestamp, *_voltage) that were present.
    The attribute ``df.attrs['input_mode']`` is 'voltage' or 'ppm'.
    """
    data = normalize_columns(df.copy())

    if "temperature" not in data.columns:
        raise SensorSchemaError(
            f"CSV is missing a temperature column. Found: {list(data.columns)}"
        )

    has_ppm = {"tds_in", "tds_out"}.issubset(data.columns)
    has_volt = {"tds_in_voltage", "tds_out_voltage"}.issubset(data.columns)

    if not has_ppm and not has_volt:
        raise SensorSchemaError(
            "CSV needs either ppm columns (tds_in, tds_out) or voltage columns "
            "(tds_in_voltage, tds_out_voltage / 'TDS Sensor 1 (V)', 'TDS Sensor 2 (V)'). "
            f"Found: {list(data.columns)}"
        )

    numeric_cols = [c for c in ("tds_in", "tds_out", "tds_in_voltage", "tds_out_voltage",
                                "temperature", "flow_rate") if c in data.columns]
    for c in numeric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    if has_ppm:
        input_mode = "ppm"
        data = data.dropna(subset=["tds_in", "tds_out", "temperature"])
    else:
        input_mode = "voltage"
        data = data.dropna(subset=["tds_in_voltage", "tds_out_voltage", "temperature"])
        data["tds_in"] = np.round(voltage_to_tds_ppm(data["tds_in_voltage"], data["temperature"], k_in), 1)
        data["tds_out"] = np.round(voltage_to_tds_ppm(data["tds_out_voltage"], data["temperature"], k_out), 1)

    if "flow_rate" not in data.columns:
        data["flow_rate"] = np.nan

    data = data.reset_index(drop=True)
    data.attrs["input_mode"] = input_mode
    return data


def is_missing(value: Optional[float]) -> bool:
    """True for None / NaN flow readings."""
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return True
