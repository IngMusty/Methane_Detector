# PFEMS · Physics-Guided Fugitive Emissions Monitoring System

A regulatory-grade web application for detecting fugitive methane emissions
in offshore oil and gas production systems using routine operational
production accounting data. Built on peer-reviewed research validated on
14 years of daily data from the Jubilee offshore field, Ghana.

Developed by Savanna Dynamics Limited.

---

## What PFEMS Does

PFEMS computes a daily gas-balance methane residual from five routine
production accounting streams and applies a three-model ensemble anomaly
detection architecture to identify sustained above-baseline deviations
consistent with unmetered gas loss. Alerts are confirmed only when at least
two independent detectors agree on a positive-direction deviation, reducing
false positives from transient accounting artefacts.

---

## Features

- Field configuration with asset name, operator, and field-specific methane
  fraction input
- CSV upload with flexible column mapping for any field's data format
- Gas-balance methane residual computation
- ARIMA(1,0,1) baseline anomaly detection with positive z-score restriction
- Isolation Forest structural anomaly detection
- PCA linear autoencoder reconstruction error detection
- Majority voting ensemble with positive direction confirmation
- Signal-to-noise ratio analysis with tiered thresholds
- Rolling 90-day Minimum Detectable Leak Fraction computation and display
- Tiered alert system: CRITICAL, WARNING, WATCH, NORMAL
- Model ensemble agreement chart
- Full export suite: PDF report, Excel alert log, CSV residual data
- Dark industrial dashboard UI
- Data processed in-session only — nothing stored server-side

---

## Running Locally

```bash
