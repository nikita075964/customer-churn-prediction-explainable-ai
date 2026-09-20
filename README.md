# Customer Churn Prediction and Analysis

This project turns the professor's notebook analysis into a Flask web dashboard. It reads only `dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv`; no cleaned CSV is created or modified. The notebook logic is preserved in `backend/functions.py` and `backend/model.py`: `TotalCharges` conversion, categorical churn rates, tenure groups, chi-square tests, Welch's t-tests, K-Means segmentation, and the three supplied classifiers.

## Run in VS Code

Open a terminal in `customer churn_2`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python backend\app.py
```

Open `http://127.0.0.1:5000`. The first startup trains the models from the original CSV in memory. Keep the original dataset path unchanged.

## API routes

`/api/overview`, `/api/statistics`, `/api/segments`, `/api/models`, `/api/insights`, and `POST /api/predict` power the frontend. The prediction payload contains the model feature columns shown in the Churn Prediction form; `Random Forest` is the default model.