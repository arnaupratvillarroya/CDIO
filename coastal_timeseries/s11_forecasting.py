
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


# --------------------------------------------------
# 7. Create test sequences
# --------------------------------------------------

# Combine the last training window with the test data.
test_input = np.concatenate([
    train_scaled[-WINDOW_SIZE:],
    test_scaled
])

X_test, y_test = create_sequences(
    test_input,
    WINDOW_SIZE
)

# Reshape for LSTM input
X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

print("\nTest windows:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# --------------------------------------------------
# 8. Build and train LSTM model
# --------------------------------------------------

import keras
from keras import layers
from keras.callbacks import EarlyStopping

# Chronological split of training windows
X_fit = X_train[:48]
y_fit = y_train[:48]

X_val = X_train[48:]
y_val = y_train[48:]

print("\nTraining/validation split:")
print("X_fit:", X_fit.shape)
print("y_fit:", y_fit.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

# Build a small LSTM model
model = keras.Sequential([
    layers.Input(shape=(WINDOW_SIZE, 1)),
    layers.LSTM(16),
    layers.Dense(1)
])

# Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_fit,
    y_fit,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=8,
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)

print("\nTraining completed.")
print("Epochs executed:", len(history.history["loss"]))


# --------------------------------------------------
# 9. Evaluate LSTM on test data
# --------------------------------------------------

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Predict normalized test values
y_pred_scaled = model.predict(X_test, verbose=0).flatten()

# Convert predictions and targets back to meters
y_pred = y_pred_scaled * train_std + train_mean
y_test_m = y_test * train_std + train_mean

# Calculate metrics
mae = mean_absolute_error(y_test_m, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test_m, y_pred)
)

r2 = r2_score(y_test_m, y_pred)

print("\nLSTM TEST METRICS")
print("-----------------")
print(f"MAE:  {mae:.3f} m")
print(f"RMSE: {rmse:.3f} m")
print(f"R2:   {r2:.3f}")


