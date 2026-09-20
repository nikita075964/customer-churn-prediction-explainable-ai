from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_dataset() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["SeniorCitizenLabel"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})
    data["ChurnBinary"] = data["Churn"].map({"No": 0, "Yes": 1})
    return data