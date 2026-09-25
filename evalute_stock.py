import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from tensorflow.keras.models import load_model


# ============================================================
# Configuration
# ============================================================

# Stocks to evaluate
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

# Number of previous trading days
sequence_length = 60

# Percentage of data used for training
train_split = 0.80


# ============================================================
# Create evaluation directory
# ============================================================

os.makedirs(
    "evaluation_results",
    exist_ok=True
)


# ============================================================
# Load trained model
# ============================================================

print(
    "Loading trained model..."
)

model = load_model(
    "stock_model.keras"
)


# ============================================================
# Store results
# ============================================================

results = []


# ============================================================
# Evaluate each stock
# ============================================================

for stock in stocks:

    print("\n" + "=" * 50)
    print(f"Evaluating {stock}...")
    print("=" * 50)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    file_path = f"data/{stock}.csv"

    data = pd.read_csv(
        file_path,
        sep=";"
    )

    # --------------------------------------------------------
    # Prepare datetime
    # --------------------------------------------------------

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    # Sort oldest → newest
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

    # Daily price range
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

    # Remove rows containing NaN values
    data = data.dropna().reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Get feature data
    # --------------------------------------------------------

    stock_data = data[
        features
    ].values

    # --------------------------------------------------------
    # Determine training/test split
    # --------------------------------------------------------

    train_size = int(
        len(stock_data) * train_split
    )

    test_size = (
        len(stock_data) - train_size
    )

    print(
        f"Total usable rows: {len(stock_data)}"
    )

    print(
        f"Training rows: {train_size}"
    )

    print(
        f"Testing rows: {test_size}"
    )

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    scaler = joblib.load(
        f"scalers/{stock}.pkl"
    )

    # --------------------------------------------------------
    # Scale data
    #
    # The scaler was fitted ONLY on the
    # training data inside train_model.py.
    # --------------------------------------------------------

    scaled_data = scaler.transform(
        stock_data
    )

    # --------------------------------------------------------
    # Create test sequences
    # --------------------------------------------------------

    X_test = []
    y_test = []

    for i in range(
        train_size,
        len(scaled_data)
    ):

        # Previous 60 days
        X_test.append(
            scaled_data[
                i - sequence_length:i
            ]
        )

        # Actual next-day close
        y_test.append(
            scaled_data[i, 3]
        )

    # Convert to NumPy
    X_test = np.array(
        X_test
    )

    y_test = np.array(
        y_test
    )

    print(
        f"Testing sequences: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Make predictions
    # --------------------------------------------------------

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    )

    # --------------------------------------------------------
    # Convert predictions back to prices
    # --------------------------------------------------------

    predictions = []

    for prediction_scaled_value in predictions_scaled:

        dummy = np.zeros(
            (1, len(features))
        )

        # Close is index 3
        dummy[0, 3] = (
            prediction_scaled_value[0]
        )

        predicted_price = (
            scaler.inverse_transform(
                dummy
            )[0, 3]
        )

        predictions.append(
            predicted_price
        )

    # --------------------------------------------------------
    # Convert actual values back to prices
    # --------------------------------------------------------

    actual_prices = []

    for actual_scaled_value in y_test:

        dummy = np.zeros(
            (1, len(features))
        )

        # Close is index 3
        dummy[0, 3] = (
            actual_scaled_value
        )

        actual_price = (
            scaler.inverse_transform(
                dummy
            )[0, 3]
        )

        actual_prices.append(
            actual_price
        )

    # Convert to NumPy arrays
    predictions = np.array(
        predictions
    )

    actual_prices = np.array(
        actual_prices
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual_prices,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_prices,
            predictions
        )
    )

    # Mean Absolute Percentage Error
    mape = np.mean(
        np.abs(
            (
                actual_prices
                - predictions
            )
            / actual_prices
        )
    ) * 100

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.append({
        "Stock": stock,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\n{stock} Results:"
    )

    print(
        f"MAE: ${mae:.2f}"
    )

    print(
        f"RMSE: ${rmse:.2f}"
    )

    print(
        f"MAPE: {mape:.2f}%"
    )

    # --------------------------------------------------------
    # Get test dates
    # --------------------------------------------------------

    test_dates = data[
        "datetime"
    ].iloc[
        train_size:
    ]

    # --------------------------------------------------------
    # Create comparison chart
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        test_dates,
        actual_prices,
        label="Actual Price"
    )

    plt.plot(
        test_dates,
        predictions,
        label="Predicted Price"
    )

    plt.title(
        f"{stock} Actual vs Predicted Closing Price"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Closing Price ($)"
    )

    plt.legend()

    plt.grid(
        True
    )

    # Save chart
    chart_path = (
        f"evaluation_results/"
        f"{stock}_evaluation.png"
    )

    plt.savefig(
        chart_path,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# Create results DataFrame
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# Print summary
# ============================================================

print("\n")
print("=" * 60)
print("MODEL EVALUATION SUMMARY")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# Save results
# ============================================================

results_df.to_csv(
    "evaluation_results/model_results.csv",
    index=False
)


print("\nResults saved to:")
print(
    "evaluation_results/model_results.csv"
)

print("\nEvaluation charts saved to:")
print(
    "evaluation_results/"
)