# E-Commerce Anomaly Detection

I put this together as a small, end-to-end example of detecting anomalous e-commerce transactions with classic unsupervised models. It includes a data generator, a labeling script, and a simple Streamlit dashboard for exploration.

## What it does

- Generates synthetic transaction data with injected outliers.
- Engineers basic time features and one-hot encodes locations.
- Flags anomalies using Isolation Forest and One-Class SVM.
- Displays anomalous transactions in an interactive dashboard.

## Requirements

- Python 3.9+ (any recent 3.x should work)
- Packages: pandas, numpy, scikit-learn, streamlit

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1) Generate sample data

```bash
python data_generator.py
```

This creates a `transactions.csv` file in the project folder so you can run the rest of the flow without hunting for data.

### 2) Detect anomalies

```bash
python anomaly_detection.py
```

This adds `anomaly_if` and `anomaly_svm` columns to `transactions.csv` so the dashboard has labels to display.

### 3) Run the dashboard

```bash
streamlit run app.py
```

## Notes

- The dashboard filters and displays only rows labeled as anomalies (`-1`) by the selected model.
- You can tweak contamination/nu settings in `anomaly_detection.py` to adjust sensitivity.
- If you regenerate data, rerun the detection step before launching the app.
