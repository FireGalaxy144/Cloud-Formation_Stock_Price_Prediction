
import os
import requests
import pandas as pd

from io import StringIO
from dotenv import load_dotenv


# Load API key from .env
load_dotenv()

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# Check that the API key exists
if not API_KEY:
    print("ERROR: TWELVE_DATA_API_KEY was not found.")
    print("Make sure your .env file contains:")
    print("TWELVE_DATA_API_KEY=your_api_key")
    exit()


# Stocks to update
STOCKS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL"
]

# Folder containing stock CSV files
DATA_FOLDER = "data"

# Create data folder if it doesn't exist
os.makedirs(DATA_FOLDER, exist_ok=True)


# Update each stock
for stock in STOCKS:

    print()
    print("=" * 50)
    print(f"Updating {stock}...")
    print("=" * 50)

    file_path = os.path.join(
        DATA_FOLDER,
        f"{stock}.csv"
    )


    # --------------------------------------------------
    # Load existing CSV
    # --------------------------------------------------

    if os.path.exists(file_path):

        try:

            existing_data = pd.read_csv(
                file_path,
                sep=";"
            )

            # Make sure datetime exists
            if "datetime" not in existing_data.columns:

                print("ERROR: 'datetime' column not found.")
                print("Columns found:")
                print(existing_data.columns.tolist())
                continue

            # Convert datetime
            existing_data["datetime"] = pd.to_datetime(
                existing_data["datetime"]
            )

            # Find newest date
            last_date = existing_data["datetime"].max()

            print(
                f"Last date in CSV: "
                f"{last_date.strftime('%Y-%m-%d')}"
            )

            # Request data after the newest date
            start_date = (
                last_date + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")

        except Exception as error:

            print("ERROR reading existing CSV:")
            print(error)
            continue

    else:

        print("No existing CSV found.")

        # Initial download
        start_date = "2020-01-01"


    # --------------------------------------------------
    # Request data from Twelve Data
    # --------------------------------------------------

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": stock,
        "interval": "1day",
        "start_date": start_date,
        "format": "CSV",
        "apikey": API_KEY
    }


    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

    except requests.RequestException as error:

        print("ERROR connecting to Twelve Data:")
        print(error)
        continue


    # --------------------------------------------------
    # Check HTTP response
    # --------------------------------------------------

    if response.status_code != 200:

        print(
            f"HTTP error: "
            f"{response.status_code}"
        )

        print(response.text)

        continue


    # --------------------------------------------------
    # Show API response if needed
    # --------------------------------------------------

    if not response.text.strip():

        print("Twelve Data returned an empty response.")

        continue


    # --------------------------------------------------
    # Convert API response to DataFrame
    # --------------------------------------------------

    try:

        # Twelve Data may return CSV with commas
       new_data = pd.read_csv(
        StringIO(response.text),
            sep=";"
        )

    except Exception as error:

        print("ERROR reading API response:")
        print(error)

        print()
        print("API response:")
        print(response.text[:1000])

        continue


    # --------------------------------------------------
    # Check API response columns
    # --------------------------------------------------

    print("API columns:")
    print(new_data.columns.tolist())


    # Check whether Twelve Data returned an error
    if "code" in new_data.columns:

        print()
        print("Twelve Data returned an error:")

        print(new_data.to_string(index=False))

        continue


    # Make sure datetime exists
    if "datetime" not in new_data.columns:

        print()
        print("ERROR: 'datetime' column was not found.")
        print("Columns returned by API:")
        print(new_data.columns.tolist())

        print()
        print("API response:")
        print(response.text[:1000])

        continue


    # --------------------------------------------------
    # Check for new data
    # --------------------------------------------------

    if new_data.empty:

        print("No new data available.")

        continue


    # --------------------------------------------------
    # Format new data
    # --------------------------------------------------

    new_data["datetime"] = pd.to_datetime(
        new_data["datetime"]
    )


    # --------------------------------------------------
    # Combine existing and new data
    # --------------------------------------------------

    if os.path.exists(file_path):

        combined_data = pd.concat(
            [
                existing_data,
                new_data
            ],
            ignore_index=True
        )

    else:

        combined_data = new_data


    # --------------------------------------------------
    # Remove duplicate dates
    # --------------------------------------------------

    combined_data = combined_data.drop_duplicates(
        subset=["datetime"],
        keep="last"
    )


    # --------------------------------------------------
    # Sort oldest → newest
    # --------------------------------------------------

    combined_data = combined_data.sort_values(
        "datetime"
    )


    # --------------------------------------------------
    # Save updated CSV
    # --------------------------------------------------

    combined_data.to_csv(
        file_path,
        sep=";",
        index=False
    )


    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    print()
    print(
        f"Successfully updated {stock}."
    )

    print(
        f"New rows received: "
        f"{len(new_data)}"
    )

    print(
        f"Total rows in CSV: "
        f"{len(combined_data)}"
    )

    print(
        f"Latest date: "
        f"{combined_data['datetime'].max().strftime('%Y-%m-%d')}"
    )

