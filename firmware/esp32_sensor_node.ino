/*
  Aqua-Swachh: ESP32-S3 Sensor Node Firmware
  =========================================
  Target: ESP32-S3 DevKit
  Sensors:
  - TDS Sensor 1 (Feed / Raw Water): Analog ADC (GPIO 1)
  - TDS Sensor 2 (Permeate):         Analog ADC (GPIO 2)
  - Temperature: DS18B20 1-Wire (GPIO 4)
  - Flow Sensor: YF-S201 Hall-Effect (GPIO 5) -- OPTIONAL, see FLOW_SENSOR_FITTED

  The TDS probes are sent as RAW VOLTAGE. Conversion to ppm (temperature
  compensation + K-value calibration) is done on the PC side by
  model/sensor_calibration.py, so the probes can be re-calibrated without
  reflashing this firmware.

  Serial Output Format (115200 baud, one line per second):
    FLOW_SENSOR_FITTED = false : tds1_voltage,tds2_voltage,temperature
                                 e.g. 1.214,0.144,24.81
    FLOW_SENSOR_FITTED = true  : tds1_voltage,tds2_voltage,temperature,flow_rate
                                 e.g. 1.214,0.144,24.81,1.85
*/

#include <DallasTemperature.h>
#include <OneWire.h>

// --- Build options ---
#define FLOW_SENSOR_FITTED false   // set true once the YF-S201 is installed

// --- Pin Definitions ---
const int PIN_TDS_IN = 1;   // ADC1 Channel for TDS Sensor 1 (feed)
const int PIN_TDS_OUT = 2;  // ADC1 Channel for TDS Sensor 2 (permeate)
const int PIN_ONE_WIRE = 4; // 1-Wire bus for DS18B20
const int PIN_FLOW = 5;     // Interrupt pin for YF-S201 flow sensor

// --- ADC averaging ---
const int ADC_SAMPLES = 30;

// --- Sensor Objects ---
OneWire oneWire(PIN_ONE_WIRE);
DallasTemperature tempSensors(&oneWire);

// --- Timing / Flow Sensor Variables ---
volatile unsigned long pulseCount = 0;
unsigned long lastSampleTime = 0;
float flowRateLPM = 0.0;
// YF-S201 calibration factor: ~7.5 pulses per second = 1 L/min
const float FLOW_CALIBRATION_FACTOR = 7.5;

void IRAM_ATTR pulseCounter() { pulseCount++; }

// Read the TDS probe output voltage (V), median-trimmed average of ADC_SAMPLES.
float readProbeVoltage(int pin) {
  long sum = 0;
  int minV = 4095, maxV = 0;
  for (int i = 0; i < ADC_SAMPLES; i++) {
    int v = analogRead(pin);
    sum += v;
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
    delay(2);
  }
  // Discard the highest and lowest outliers
  float avg = (sum - minV - maxV) / float(ADC_SAMPLES - 2);
  // ESP32 ADC: 12-bit (0-4095) with ~3.3 V full scale at 11 dB attenuation
  return (avg / 4095.0) * 3.3;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Initialize DS18B20
  tempSensors.begin();

#if FLOW_SENSOR_FITTED
  // Initialize Flow Sensor Interrupt
  pinMode(PIN_FLOW, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_FLOW), pulseCounter, FALLING);
#endif
  lastSampleTime = millis();

  // Configure ADC
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db); // Full range 0 - 3.3V
}

void loop() {
  unsigned long now = millis();

  // Sample every 1000 ms (1 Hz)
  if (now - lastSampleTime >= 1000) {
    float dt = (now - lastSampleTime) / 1000.0;
    lastSampleTime = now;

#if FLOW_SENSOR_FITTED
    // 1. Flow Rate: Q (L/min) = pulse frequency (Hz) / 7.5
    detachInterrupt(digitalPinToInterrupt(PIN_FLOW));
    flowRateLPM = (pulseCount / dt) / FLOW_CALIBRATION_FACTOR;
    pulseCount = 0;
    attachInterrupt(digitalPinToInterrupt(PIN_FLOW), pulseCounter, FALLING);
#endif

    // 2. Read Water Temperature
    tempSensors.requestTemperatures();
    float waterTemp = tempSensors.getTempCByIndex(0);
    // Fallback if sensor disconnected
    if (waterTemp < -50 || waterTemp > 85) {
      waterTemp = 25.0; // Default reference
    }

    // 3. Read TDS probe voltages
    float tds1V = readProbeVoltage(PIN_TDS_IN);
    float tds2V = readProbeVoltage(PIN_TDS_OUT);

    // 4. Output CSV formatted stream over Serial
    Serial.print(tds1V, 3);
    Serial.print(",");
    Serial.print(tds2V, 3);
    Serial.print(",");
#if FLOW_SENSOR_FITTED
    Serial.print(waterTemp, 2);
    Serial.print(",");
    Serial.println(flowRateLPM, 2);
#else
    Serial.println(waterTemp, 2);
#endif
  }
}
