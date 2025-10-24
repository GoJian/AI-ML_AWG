# Code In Development

This directory holds code that is currently in development for AI/ML AWG projects. Review what other members are working on and contact them to collaborate if you are interested! More information followed on each topic's directory...

## Code Development Topics:

- *coming soon...*


## RR9 Streamlit App - Imputation Tab

The `RR9_DT_Streamlit_App.py` app now includes an Imputation tab that implements the pipelines from `rr9_imputation`:

- KNN Imputer (with StandardScaler)
- Random Sample Imputer (distribution-preserving, custom transformer)
- MICE with Bagging Regressor (IterativeImputer + Bagging)

You can select the input dataset from either the data assembled in the "Initial Data Loading" tab or the reference CSVs in `Working_Code/rr9_imputation` or `Manuscript_Code/rr9_imputation` (e.g., `full_df.csv`, `merged_flight_data.csv`, `merged_non_flight_data.csv`).

Outputs (CSV and joblib pipeline) are saved by default in `Working_Code/rr9_imputation/` and will be reused if they already exist to avoid re-running long jobs.

To run the app locally:

```bash
streamlit run Working_Code/RR9_DT_Streamlit_App.py
```

Note: MICE + Bagging can be long-running on large datasets; prefer starting with low `n_estimators` and iterations, then scaling up.
