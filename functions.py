import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


CATEGORICAL_ANALYSIS = ["gender", "SeniorCitizenLabel", "Partner", "Dependents", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "Contract", "PaperlessBilling", "PaymentMethod"]
NUMERIC_ANALYSIS = ["tenure", "MonthlyCharges", "TotalCharges"]
SEGMENT_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def dashboard_metrics(data):
    return {"total_customers": int(len(data)), "churned_customers": int(data["ChurnBinary"].sum()), "churn_rate": round(float(data["ChurnBinary"].mean() * 100), 2), "average_monthly_charges": round(float(data["MonthlyCharges"].mean()), 2), "average_tenure": round(float(data["tenure"].mean()), 2)}


def churn_distribution(data):
    counts = data["Churn"].value_counts()
    return [{"label": label, "count": int(count), "percentage": round(float(count / len(data) * 100), 2)} for label, count in counts.items()]


def grouped_churn(data, column):
    grouped = data.groupby(column, dropna=False)["ChurnBinary"].agg(["count", "mean"]).reset_index()
    return [{"label": str(row[column]), "customers": int(row["count"]), "churn_rate": round(float(row["mean"] * 100), 2)} for _, row in grouped.iterrows()]


def tenure_analysis(data):
    groups = pd.cut(data["tenure"], bins=[-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    grouped = data.assign(TenureGroup=groups).groupby("TenureGroup", observed=True)["ChurnBinary"].agg(["count", "mean"]).reset_index()
    return [{"label": str(row["TenureGroup"]), "customers": int(row["count"]), "churn_rate": round(float(row["mean"] * 100), 2)} for _, row in grouped.iterrows()]


def statistical_tests(data):
    chi_results = []
    for column in CATEGORICAL_ANALYSIS:
        table = pd.crosstab(data[column], data["Churn"])
        chi2, p_value, _, _ = chi2_contingency(table)
        chi_results.append({"variable": column, "chi2": round(float(chi2), 4), "p_value": float(p_value), "significant": bool(p_value < 0.05)})
    t_results = []
    for column in NUMERIC_ANALYSIS:
        no_churn = data.loc[data["Churn"] == "No", column].dropna()
        churn = data.loc[data["Churn"] == "Yes", column].dropna()
        statistic, p_value = ttest_ind(no_churn, churn, equal_var=False)
        t_results.append({"variable": column, "t_statistic": round(float(statistic), 4), "p_value": float(p_value), "significant": bool(p_value < 0.05)})
    return {"chi_square": chi_results, "t_test": t_results}


def segment_analysis(data):
    segment_data = SimpleImputer(strategy="median").fit_transform(data[SEGMENT_COLUMNS])
    scaled = StandardScaler().fit_transform(segment_data)
    scores = {}
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        scores[k] = float(silhouette_score(scaled, labels))
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(scaled)
    segmented = data.assign(Segment=labels)
    summary = segmented.groupby("Segment")[SEGMENT_COLUMNS + ["ChurnBinary"]].mean().round(2).reset_index()
    summary["ChurnRatePct"] = summary["ChurnBinary"] * 100
    return {"best_k": int(best_k), "silhouette_scores": [{"k": int(k), "score": round(score, 4)} for k, score in scores.items()], "summary": [{key: float(value) if isinstance(value, (np.floating, float)) else int(value) for key, value in row.items()} for row in summary.to_dict("records")], "points": [{"x": float(row["tenure"]), "y": float(row["MonthlyCharges"]), "segment": int(row["Segment"])} for row in segmented[["tenure", "MonthlyCharges", "Segment"]].to_dict("records")[::8]]}


def business_insights(data):
    high_risk = data[(data["Contract"] == "Month-to-month") & (data["tenure"] <= 12) & (data["MonthlyCharges"] >= data["MonthlyCharges"].median())]
    return {"high_risk_size": int(len(high_risk)), "high_risk_share": round(float(len(high_risk) / len(data) * 100), 2), "high_risk_churn_rate": round(float(high_risk["ChurnBinary"].mean() * 100), 2), "recommendations": ["Strengthen onboarding during the first 6-12 months.", "Target month-to-month customers with contract migration offers.", "Investigate price and value perception among high-charge customers.", "Promote technical support and online security retention bundles.", "Prioritize outreach using churn probabilities and test interventions experimentally."]}