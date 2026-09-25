import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# Configuration
# ============================================================

# Stocks to train on
stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

# Features used by the model
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

# Number of previous trading days used to make a prediction
sequence_length = 60

# Percentage of data used for training
train_split = 0.80

# Number of training epochs
epochs = 100

# Training batch size
batch_size = 32


# ============================================================
# Create directories
# ============================================================

os.makedirs("scalers", exist_ok=True)


# ============================================================
# Training data
# ============================================================

X_train = []
y_train = []


# ============================================================
# Process each stock
# ============================================================

for stock in stocks:

    print("\n" + "=" * 50)
    print(f"Processing {stock}...")
    print("=" * 50)

    # --------------------------------------------------------
    # Load stock data
    # --------------------------------------------------------

    file_path = f"data/{stock}.csv"

    data = pd.read_csv(
        file_path,
        sep=";"
    )

    # --------------------------------------------------------
    # Convert datetime
    # --------------------------------------------------------

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    # Make sure data is chronological
    data = data.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Create additional features
    # --------------------------------------------------------

    # Daily percentage return
    data["daily_return"] = (
        data["close"].pct_change()
    )

    # Difference between daily high and low
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

    # Remove rows created by rolling calculations
    data = data.dropna().reset_index(drop=True)

    # --------------------------------------------------------
    # Get feature data
    # --------------------------------------------------------

    stock_data = data[features].values

    print(
        f"Total usable rows: {len(stock_data)}"
    )

    # --------------------------------------------------------
    # Determine training size
    # --------------------------------------------------------

    train_size = int(
        len(stock_data) * train_split
    )

    print(
        f"Training rows: {train_size}"
    )

    print(
        f"Testing rows: {len(stock_data) - train_size}"
    )

    # --------------------------------------------------------
    # Separate training data
    # --------------------------------------------------------

    train_data = stock_data[:train_size]

    # --------------------------------------------------------
    # Create scaler
    # --------------------------------------------------------

    scaler = MinMaxScaler()

    # IMPORTANT:
    # Only fit the scaler on training data.
    scaler.fit(train_data)

    # Save scaler
    scaler_path = f"scalers/{stock}.pkl"

    joblib.dump(
        scaler,
        scaler_path
    )

    print(
        f"Saved scaler: {scaler_path}"
    )

    # --------------------------------------------------------
    # Scale entire dataset using training scaler
    # --------------------------------------------------------

    scaled_data = scaler.transform(
        stock_data
    )

    # --------------------------------------------------------
    # Create training sequences
    # --------------------------------------------------------

    for i in range(
        sequence_length,
        train_size
    ):

        # Previous 60 trading days
        X_train.append(
            scaled_data[
                i - sequence_length:i
            ]
        )

        # Target = next day's closing price
        #
        # Close is index 3:
        # open = 0
        # high = 1
        # low = 2
        # close = 3
        y_train.append(
            scaled_data[i, 3]
        )


# ============================================================
# Convert training data to NumPy arrays
# ============================================================

X_train = np.array(
    X_train
)

y_train = np.array(
    y_train
)


print("\n" + "=" * 50)
print("FINAL TRAINING DATA")
print("=" * 50)

print(
    f"X shape: {X_train.shape}"
)

print(
    f"y shape: {y_train.shape}"
)

print(
    f"Number of features: {X_train.shape[2]}"
)


# ============================================================
# Create LSTM model
# ============================================================

model = Sequential()

model.add(
    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )
)

model.add(
    LSTM(
        64,
        return_sequences=True
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    LSTM(
        64,
        return_sequences=True
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    LSTM(
        64
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(32)
)

model.add(
    Dense(1)
)


# ============================================================
# Compile model
# ============================================================

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# ============================================================
# Callbacks
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_learning_rate = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)


# ============================================================
# Train model
# ============================================================

print("\n" + "=" * 50)
print("TRAINING MODEL")
print("=" * 50)

history = model.fit(
    X_train,
    y_train,

    # Use the last 10% of the training data
    # as validation data.
    validation_split=0.10,

    epochs=epochs,
    batch_size=batch_size,

    # Do NOT shuffle time-series data.
    shuffle=False,

    callbacks=[
        early_stopping,
        reduce_learning_rate
    ]
)


# ============================================================
# Save model
# ============================================================

model.save(
    "stock_model.keras"
)


# ============================================================
# Save training history
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    "training_history.csv",
    index=False
)


# ============================================================
# Training summary
# ============================================================

print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)

print(
    "Model saved as: stock_model.keras"
)

print(
    "Training history saved as: training_history.csv"
)

print(
    f"Epochs completed: {len(history.history['loss'])}"
)

print(
    f"Final training loss: "
    f"{history.history['loss'][-1]:.6f}"
)

print(
    f"Final validation loss: "
    f"{history.history['val_loss'][-1]:.6f}"
)