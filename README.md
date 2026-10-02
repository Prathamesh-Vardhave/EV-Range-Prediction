# EVRange: Real-World EV Range Prediction

Predicts an electric vehicle's **real-world driving range (km)** from vehicle
specs, battery health, and driving/environmental conditions — using
`real_world_range_km` as the target, deliberately **without** the
lab-certified `ARAI_certified_range_km` or the derived
`energy_consumption_kWh_per_km` as inputs (both would leak the answer).

## Project Structure

```
EVRange/
├── streamlit_app.py       # Streamlit UI for entering features and viewing predictions
├── api/
│   ├── main.py             # FastAPI app (predict endpoint)
│   └── schemas.py          # Pydantic request/response schemas
├── model/
│   └── best_model.pkl      # Trained pipeline — NOT included yet, see below
├── notebook/
│   └── ev_range_prediction.ipynb   # Full EDA + modelling notebook
├── requirements.txt
├── README.md
└── .gitignore
```

## Status

Model training is currently being done in a **Kaggle Notebook**
(`notebook/ev_range_prediction.ipynb`). The `api/` and `streamlit_app.py`
files are already wired up to load `model/best_model.pkl`, but that file is
**not present yet** — both apps will start fine and clearly tell you the
model isn't loaded until you add it.

### Once training is finished in Kaggle:

1. Export the winning pipeline (preprocessing + model) as a single
   `best_model.pkl` (e.g. with `pickle` or `joblib`).
2. Place it at `model/best_model.pkl`.
3. Run the API or the Streamlit app (see below) — predictions will work
   automatically.

## Notebook

`notebook/ev_range_prediction.ipynb` covers, end to end:

- Data loading & inspection (shape, dtypes, missing values, duplicates)
- Data cleaning
- EDA (distributions, boxplots by category, correlation heatmap)
- Feature engineering (`battery_health_pct`, `power_to_weight`,
  `speed_temp_interaction`, `is_extreme_temp`)
- Preprocessing pipeline (scaling + one-hot encoding)
- Training & comparing 7 regression models: Linear Regression, Ridge, Lasso,
  Random Forest, Gradient Boosting, XGBoost, CatBoost
- Evaluation with MAE, RMSE, R²
- Hyperparameter tuning (RandomizedSearchCV) for the strongest tree-based
  models
- Feature importance
- Identifying the best model (saving is intentionally left for you to do in
  Kaggle)

To run it on Kaggle: upload `india_ev_range_dataset.csv` as a Kaggle Dataset,
attach it to the notebook, and update `DATA_PATH` in the "Load the EV
Dataset" cell if needed.

## Running the API

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for interactive API docs.
`POST /predict` accepts the feature set defined in `api/schemas.py` and
returns `predicted_range_km`.

## Running the Streamlit App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Notes

- No Docker or deployment configuration is included yet — this is a local
  development structure only.
- The excluded features (`ARAI_certified_range_km`,
  `energy_consumption_kWh_per_km`) and dropped identifiers (`model`,
  `variant`) are documented in `api/schemas.py` and the notebook, so the
  feature set used in the API/UI matches what the model is trained on.
