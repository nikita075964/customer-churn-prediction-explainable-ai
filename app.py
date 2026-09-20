from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from data_loader import load_dataset
from functions import business_insights, churn_distribution, dashboard_metrics, grouped_churn, segment_analysis, statistical_tests, tenure_analysis
from model import feature_importance, train_models


ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
data = load_dataset()
fitted_models, model_results, feature_columns = train_models(data)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/overview")
def overview():
    return jsonify({"metrics": dashboard_metrics(data), "distribution": churn_distribution(data), "contract": grouped_churn(data, "Contract"), "internet": grouped_churn(data, "InternetService"), "payment": grouped_churn(data, "PaymentMethod"), "tenure": tenure_analysis(data)})


@app.get("/api/statistics")
def statistics():
    return jsonify(statistical_tests(data))


@app.get("/api/segments")
def segments():
    return jsonify(segment_analysis(data))


@app.get("/api/models")
def models():
    return jsonify({"results": model_results, "feature_importance": feature_importance(fitted_models)})


@app.get("/api/insights")
def insights():
    return jsonify(business_insights(data))


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    missing = [column for column in feature_columns if column not in payload]
    if missing:
        return jsonify({
            "error": f"Missing required field(s): {', '.join(missing)}",
            "fields": missing,
        }), 400
    row = payload.copy()
    row["SeniorCitizen"] = int(row["SeniorCitizen"])
    row["tenure"] = float(row["tenure"])
    row["MonthlyCharges"] = float(row["MonthlyCharges"])
    row["TotalCharges"] = float(row["TotalCharges"]) if str(row["TotalCharges"]).strip() else None
    prediction_row = pd.DataFrame([{column: row[column] for column in feature_columns}], columns=feature_columns)
    model_name = payload.get("model", "Random Forest")
    if model_name not in fitted_models:
        return jsonify({"error": "Unknown model", "models": list(fitted_models)}), 400
    pipeline = fitted_models[model_name]
    prediction = int(pipeline.predict(prediction_row)[0])
    probability = float(pipeline.predict_proba(prediction_row)[0, 1])
    return jsonify({"model": model_name, "prediction": "Yes" if prediction else "No", "probability": round(probability, 4), "risk": "High" if probability >= 0.5 else "Low"})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)