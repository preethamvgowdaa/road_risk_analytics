# Agentic Data Science: Automated Road Risk Analytics.

A working implementation of the Phase II capstone project: an agentic pipeline that ingests road sensor data, cleans and scores it, predicts deterioration risk with a trained machine learning model, visualizes results on an interactive dashboard, and drafts and dispatches prioritized maintenance reports to civic engineers across six Indian cities (Bangalore, Mumbai, Delhi, Chennai, Hyderabad, Pune)

## What is actually implemented here

Every module described as "in progress" in the mid semester report is completed in this codebase:

1. Realistic synthetic sensor dataset across named road segments (Outer Ring Road, Western Express Highway, GST Road, and so on) with a physically motivated monsoon rainfall curve, ward level drainage quality, and domain weighted risk labels rather than random labels.
2. A data quality agent that imputes missing sensor readings at the ward level and flags anomalous readings with an isolation forest.
3. A random forest classifier trained on nine engineered features, evaluated with accuracy, precision, recall, F1, ROC AUC, and a feature importance breakdown.
4. A composite risk engine that blends the model's class probabilities with transparent engineering rules into one 0 to 100 score per road segment, plus the top three contributing factors for that score.
5. An agentic report generator that reasons over a road segment's risk record and drafts a maintenance report. It uses a local Ollama model automatically if one is running on the machine, and otherwise falls back to a rule based drafting agent with varied phrasing, so the module works with zero paid APIs, matching the original bill of materials.
6. A municipal email dispatcher over Gmail SMTP, defaulting to a safe dry run mode so nothing is ever sent accidentally, with a per city, per ward civic contact directory.
7. A full Streamlit dashboard: overview metrics, model performance, a city map colored by risk tier, a ranked segment table with CSV export, and a report generation and dispatch panel with a dispatch history log.

## Project layout

```
road_risk_analytics/
  app.py                   Streamlit dashboard
  train_model.py            Trains and saves the risk model
  smoke_test.py              End to end verification script
  civic_contacts.json       Per city, per ward civic engineer contact directory
  requirements.txt
  data/
    generate_dataset.py     Synthetic sensor data generator
    road_sensor_dataset.csv Generated dataset (created on first run)
  src/
    config.py               Paths, constants, city and road metadata
    data_processing.py      Cleaning, feature engineering, anomaly detection
    risk_model.py            Random forest model plus the composite risk engine
    report_generator.py     Agentic report drafting (local LLM plus fallback)
    email_dispatch.py        SMTP dispatch with dry run safety
  models/                    Saved model, scaler, and training metrics (created on first run)
  outputs/                   Dispatch logs and any exported files
```

## Setup

```
pip install -r requirements.txt
```

## Running it

Step one, generate the dataset and train the model:

```
python train_model.py
```

Step two, verify everything works end to end:

```
python smoke_test.py
```

Step three, launch the dashboard:

```
streamlit run app.py
```

The dashboard opens in your browser. You can upload your own CSV with the same columns as `data/road_sensor_dataset.csv`, or use the bundled sample data. Use the sidebar to filter by city and risk score, then go to the "AI Reports and Dispatch" tab to generate a report for any road segment and preview the municipal email dispatch in dry run mode.

## Sending real emails (optional)

Email dispatch defaults to a dry run so the demo is always safe to run live. To actually send through Gmail:

1. Create a Gmail App Password for the sending account (this requires two factor authentication to be enabled on that account).
2. Set two environment variables before launching the app:

```
export SMTP_SENDER_EMAIL="your_address@gmail.com"
export SMTP_APP_PASSWORD="your_16_character_app_password"
```

3. In the "AI Reports and Dispatch" tab, uncheck "Dry run" before clicking "Send report".

The recipient addresses in `civic_contacts.json` are placeholder addresses for demonstration; replace them with real contacts before sending for real, or use the override recipient field in the dashboard.

## Using a local LLM for report drafting (optional)

The report generator checks for a locally running Ollama server (`http://localhost:11434`) and uses it automatically if present, matching the "Open Source Local LLM via Ollama and Flowise" line in the technology specification. If Ollama is not running, the rule based drafting agent is used instead, so the feature works out of the box with no setup and no API cost. To enable the LLM backend, install Ollama separately and pull a small model such as `llama3.2`, then just relaunch the app.

## Model performance

Run `python train_model.py` to see the live numbers for your generated dataset (they are also saved to `models/training_metrics.json` and shown on the dashboard). Because the risk labels here are derived from a genuine domain weighted formula rather than assigned at random, the reported accuracy reflects a real, defensible classification task rather than an inflated number, and is a fair figure to quote in your final review.

## Notes for the final review

* Total cost remains zero: every library used (pandas, numpy, scikit-learn, streamlit, plotly, Ollama) is free and open source, matching the original bill of materials.
* The anomaly detection stage and the composite risk engine (ML score blended with explainable engineering rules) go beyond what was in the original scope and are worth highlighting as the "one step ahead" additions when you present.
* Update `civic_contacts.json` with your actual ward engineer contacts if you plan to demonstrate a live send during the review.
