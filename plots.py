import os
import pandas as pd
import matplotlib.pyplot as plt

FORECAST_DIR = "data/forecasts"
OUTPUT_DIR = "data/forecast_plots"

# Create output folder if doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all forecast CSVs
files = [f for f in os.listdir(FORECAST_DIR) if f.endswith(".csv")]

if not files:
    print("No forecast CSV files found in data/forecasts/")
    exit()

print(f"Found {len(files)} forecast files:\n", "\n".join(files))

for file in files:
    file_path = os.path.join(FORECAST_DIR, file)
    df = pd.read_csv(file_path, parse_dates=["date"])

    product_name = file.replace("_forecast.csv", "")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(df['date'], df['yhat'], marker='o', linewidth=2)

    plt.title(f"{product_name.capitalize()} – 14-Day Forecast", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Predicted Demand (yhat)", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    out_path = os.path.join(OUTPUT_DIR, f"{product_name}_forecast_plot.png")
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot: {out_path}")

print("\nAll plots generated successfully!")
