
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "results_s10" / "shoreline_timeseries_MS.csv"
OUTPUT_DIR = BASE_DIR / "results_s11"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load monthly time series
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Date range:", df["date"].min(), "to", df["date"].max())

# --------------------------------------------------
# 3. Select target variable
# --------------------------------------------------

# Use the monthly mean distance as the forecasting target.
# Confirm the actual column name before proceeding.
TARGET_COLUMN = "distance_m"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Column '{TARGET_COLUMN}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

series = df[TARGET_COLUMN].astype(float)

if series.isna().any():
    raise ValueError("The target series contains missing values.")

# --------------------------------------------------
# 4. Chronological train/test split
# --------------------------------------------------

TRAIN_SIZE = 72
WINDOW_SIZE = 12

if len(series) != 96:
    raise ValueError(
        f"Expected 96 monthly observations, found {len(series)}."
    )

train = series.iloc[:TRAIN_SIZE].to_numpy()
test = series.iloc[TRAIN_SIZE:].to_numpy()

print("\nTrain observations:", len(train))
print("Test observations:", len(test))
print("Window size:", WINDOW_SIZE)
print("Train period:", df["date"].iloc[0], "to",
      df["date"].iloc[TRAIN_SIZE - 1])
print("Test period:", df["date"].iloc[TRAIN_SIZE], "to",
      df["date"].iloc[-1])

# --------------------------------------------------
# 5. Normalize using training data only
# --------------------------------------------------

train_mean = train.mean()
train_std = train.std()

if train_std == 0:
    raise ValueError("Training standard deviation is zero.")

train_scaled = (train - train_mean) / train_std
test_scaled = (test - train_mean) / train_std

print("\nNormalization:")
print("Training mean:", train_mean)
print("Training standard deviation:", train_std)

print("\nPreparation completed successfully.")


# --------------------------------------------------
# 6. Create sliding windows
# --------------------------------------------------

def create_sequences(data, window_size):
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    return np.array(X), np.array(y)


# Create training sequences
X_train, y_train = create_sequences(
    train_scaled,
    WINDOW_SIZE
)

# Print shapes
print("\nSliding windows:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Reshape for LSTM input:
# (samples, timesteps, features)
X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

print("X_train LSTM shape:", X_train.shape)

