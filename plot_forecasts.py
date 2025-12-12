import pandas as pd
import matplotlib.pyplot as plt

# CHANGE THIS to the forecast file you want to visualize
csv_path = "data/forecasts/beauty_forecast.csv"

# Load forecast data
df = pd.read_csv(csv_path, parse_dates=["date"])

plt.figure(figsize=(10,5))
plt.plot(df['date'], df['yhat'], marker='o', linewidth=2)

plt.title("14-Day Sales Forecast", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Predicted Demand (yhat)", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
