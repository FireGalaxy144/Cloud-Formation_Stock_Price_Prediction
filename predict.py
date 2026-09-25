import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


# ============================================================
# Configuration
# ============================================================

# Stocks supported by the model
stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

# Features used during training
features = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
    "price_range",
    "ma_10",
    "ma_30"
]

# Number of previous trading days
sequence_length = 60


# ============================================================
# Ask user for stock
# ============================================================

stock = input(
    "Enter stock ticker (AAPL, MSFT, NVDA, AMZN, GOOGL): "
).upper()


# Check if stock is supported
if stock not in stocks:

    print("Invalid stock ticker.")

    print(
        "Available stocks:",
        ", ".join(stocks)
    )

    exit()


# ============================================================
# Load stock data
# ============================================================

file_path = f"data/{stock}.csv"

data = pd.read_csv(
    file_path,
    sep=";"
)


# ============================================================
# Prepare datetime
# ============================================================

data["datetime"] = pd.to_datetime(
    data["datetime"]
)

# Make sure data is chronological
data = data.sort_values(
    "datetime"
).reset_index(drop=True)


# ============================================================
# Create additional features
# ============================================================

# Daily percentage return
data["daily_return"] = (
    data["close"].pct_change()
)

# Difference between high and low
data["price_range"] = (
    data["high"] - data["low"]
)

# 10-day moving average
data["ma_10"] = (
    data["close"]
    .rolling(window=10)
    .mean()
)

# 30-day moving average
data["ma_30"] = (
    data["close"]
    .rolling(window=30)
    .mean()
)


# Remove rows with missing feature values
data = data.dropna().reset_index(
    drop=True
)


# ============================================================
# Check that enough data exists
# ============================================================

if len(data) < sequence_length:

    print(
        f"Not enough data for {stock}."
    )

    print(
        f"Need at least {sequence_length} usable rows."
    )

    exit()


# ============================================================
# Load scaler
# ============================================================

scaler = joblib.load(
    f"scalers/{stock}.pkl"
)


# ============================================================
# Scale data
# ============================================================

scaled_data = scaler.transform(
    data[features]
)


# ============================================================
# Get the latest 60 trading days
# ============================================================

last_60_days = scaled_data[
    -sequence_length:
]


# ============================================================
# Prepare LSTM input
# ============================================================

X_test = np.array(
    [last_60_days]
)


# ============================================================
# Load model
# ============================================================

model = load_model(
    "stock_model.keras"
)


# ============================================================
# Make prediction
# ============================================================

prediction_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# Convert prediction back to dollar price
# ============================================================

# The model predicts Close.
#
# Close is feature index 3:
#
# 0 = open
# 1 = high
# 2 = low
# 3 = close
# 4 = volume
# 5 = daily_return
# 6 = price_range
# 7 = ma_10
# 8 = ma_30

dummy = np.zeros(
    (1, len(features))
)

dummy[0, 3] = (
    prediction_scaled[0, 0]
)


prediction = scaler.inverse_transform(
    dummy
)[0, 3]


# ============================================================
# Get latest actual closing price
# ============================================================

current_close = data[
    "close"
].iloc[-1]


latest_date = data[
    "datetime"
].iloc[-1].date()


# ============================================================
# Calculate predicted percentage change
# ============================================================

percentage_change = (
    (prediction - current_close)
    / current_close
) * 100


# ============================================================
# Determine predicted direction
# ============================================================

if prediction > current_close:

    direction = "UP"

elif prediction < current_close:

    direction = "DOWN"

else:

    direction = "UNCHANGED"


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 40)
print("STOCK PRICE PREDICTION")
print("=" * 40)

print(
    f"Stock: {stock}"
)

print(
    f"Latest date: {latest_date}"
)

print(
    f"Current close: ${current_close:.2f}"
)

print(
    f"Predicted next close: ${prediction:.2f}"
)

print(
    f"Predicted change: {percentage_change:.2f}%"
)

print(
    f"Predicted direction: {direction}"
)

print("=" * 40)