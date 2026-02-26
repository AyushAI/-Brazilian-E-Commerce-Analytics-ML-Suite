import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

# regression models to predict payment_value 

warnings.filterwarnings('ignore')

# ── 1. Load & clean ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('final_outlierTreated.csv')

# Drop categorical and non-predictor columns
cat_cols = ['customer_unique_id', 'customer_city', 'customer_state', 'payment_type']
drop_cols = cat_cols + ['Churn']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Target and features
TARGET = 'payment_value'
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Features : {list(X.columns)}")
print(f"Samples  : {len(df):,}")

# ── 2. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ── 3. Scale ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Models ────────────────────────────────────────────────────────────────
alphas = np.logspace(-3, 3, 50)   # 0.001 … 1000

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (CV)'       : RidgeCV(alphas=alphas, cv=5),
    'Lasso (CV)'       : LassoCV(alphas=alphas, cv=5, max_iter=5000, random_state=42),
}

# ── 5. Train & evaluate ───────────────────────────────────────────────────────
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    extra = ""
    if hasattr(model, 'alpha_'):
        extra = f"  best alpha = {model.alpha_:.4f}"

    results.append({'Model': name, 'R²': r2, 'MAE': mae, 'RMSE': rmse})
    print(f"  R²={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}{extra}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

res_df = pd.DataFrame(results).sort_values('R²', ascending=False)
print(res_df.to_string(index=False))

best = res_df.iloc[0]['Model']
print(f"\n✅  Best model: {best}  (highest R²)")
