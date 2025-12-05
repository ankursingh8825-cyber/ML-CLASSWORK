# regression_fresh.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def main():
    # -------------------------
    # 1) Create (or load) data
    # -------------------------
    np.random.seed(0)
    X = pd.DataFrame({
        "x1": np.random.rand(200) * 10,
        "x2": np.random.rand(200) * 5
    })
    y = 3 * X["x1"] + 2 * X["x2"] + np.random.randn(200) * 1.5

    # -------------------------
    # 2) Train / Test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # 3) Pipeline (scaler + model)
    # -------------------------
    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    pipeline.fit(X_train, y_train)

    # -------------------------
    # 4) Evaluate
    # -------------------------
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"R2:  {r2:.4f}")

    # -------------------------
    # 5) Save pipeline (includes scaler)
    # -------------------------
    MODEL_PATH = "linear_regression_pipeline.pkl"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved model to {MODEL_PATH}")

    # -------------------------
    # 6) Load and predict using DataFrame (no warning)
    # -------------------------
    loaded = pickle.load(open(MODEL_PATH, "rb"))

    # Single example as DataFrame with same column names
    example_df = pd.DataFrame([[4.5, 2.0]], columns=["x1", "x2"])
    pred_single = loaded.predict(example_df)
    print("Example prediction (single):", pred_single.tolist())

    # Batch example (multiple rows) as DataFrame
    batch_df = pd.DataFrame([[4.5, 2.0], [7.0, 1.2], [0.5, 4.4]], columns=["x1", "x2"])
    pred_batch = loaded.predict(batch_df)
    print("Example prediction (batch):", pred_batch.tolist())

    # If you ever have a numpy array, convert it to DataFrame first:
    numpy_input = np.array([[4.5, 2.0]])
    numpy_as_df = pd.DataFrame(numpy_input, columns=["x1", "x2"])
    print("Prediction (from numpy converted to DataFrame):", loaded.predict(numpy_as_df).tolist())

if __name__ == "__main__":
    main()
