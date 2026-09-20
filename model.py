import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DROP_COLUMNS = ["customerID", "Churn", "ChurnBinary", "SeniorCitizenLabel"]


def train_models(data):
    features = data.drop(columns=DROP_COLUMNS)
    target = data["ChurnBinary"]
    categorical = features.select_dtypes(include=["object"]).columns.tolist()
    numeric = features.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=42, stratify=target)
    definitions = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    fitted, results = {}, []
    for name, estimator in definitions.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        probability = pipeline.predict_proba(x_test)[:, 1]
        fitted[name] = pipeline
        results.append({"model": name, "accuracy": round(float(accuracy_score(y_test, predicted)), 4), "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4), "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4), "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4), "roc_auc": round(float(roc_auc_score(y_test, probability)), 4)})
    return fitted, sorted(results, key=lambda row: row["roc_auc"], reverse=True), list(features.columns)


def feature_importance(fitted):
    output = {}
    for name in ["Random Forest", "Gradient Boosting"]:
        pipeline = fitted[name]
        model = pipeline.named_steps["model"]
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        ranked = pd.DataFrame({"feature": names, "importance": model.feature_importances_}).sort_values("importance", ascending=False).head(20)
        output[name] = [{"feature": row["feature"], "importance": round(float(row["importance"]), 5)} for _, row in ranked.iterrows()]
    return output