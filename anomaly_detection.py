import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_and_preprocess():
    df = pd.read_csv("transactions.csv", parse_dates=["timestamp"])
    
    # Feature engineering
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    X = df[["amount", "location", "hour", "dayofweek"]]
    
    # One-hot encode categorical
    X = pd.get_dummies(X, columns=["location"], drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled

def detect_anomalies(X_scaled, method="isolation_forest"):
    if method == "isolation_forest":
        model = IsolationForest(contamination=0.02)
    else:
        model = OneClassSVM(nu=0.02, kernel="rbf", gamma="scale")
    
    preds = model.fit_predict(X_scaled)
    return preds

if __name__ == "__main__":
    df, X_scaled = load_and_preprocess()
    df["anomaly_if"] = detect_anomalies(X_scaled, method="isolation_forest")
    df["anomaly_svm"] = detect_anomalies(X_scaled, method="svm")
    df.to_csv("transactions.csv", index=False)