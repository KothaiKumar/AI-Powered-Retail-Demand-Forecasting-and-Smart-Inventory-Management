import os
import cv2
import fnmatch
import numpy as np
import pandas as pd
from datetime import datetime
from prophet import Prophet
from tensorflow.keras.models import load_model

TRAIN_IMAGE_ROOT = "data/slash-dataset-480p"
TEST_IMAGE_ROOT = "data/test-images"
CLASS_MODEL_PATH = "d:/kothai/finalyearProject/product_classification/model/model_checkpoint.keras"

FORECAST_HORIZON = 14
LEAD_TIME_DAYS = 7
SAFETY_MULTIPLIER = 1.65
DEFAULT_CURRENT_STOCK = 10

FORECAST_OUTPUT_DIR = "data/forecasts"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def build_class_names(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def load_classification_model(path):
    return load_model(path)

def infer_img_size_from_dataset(root):
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            for f in os.listdir(p):
                fp = os.path.join(p, f)
                img = cv2.imread(fp)
                if img is not None:
                    h, w = img.shape[:2]
                    return min(h, w)
    return 224

def preprocess_image(path, target_size):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    s = min(h, w)
    cy, cx = h // 2, w // 2
    img = img[cy - s//2 : cy + s//2, cx - s//2 : cx + s//2]
    img = cv2.resize(img, (target_size, target_size))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def classify_image(model, class_names, image_path, target_size):
    x = preprocess_image(image_path, target_size)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    prob = float(np.max(preds))
    return class_names[idx], prob

def build_sales_history_from_images(root):
    records = []
    for cls in sorted(os.listdir(root)):
        cls_folder = os.path.join(root, cls)
        if not os.path.isdir(cls_folder):
            continue
        for fname in os.listdir(cls_folder):
            fp = os.path.join(cls_folder, fname)
            if os.path.isfile(fp):
                ts = os.path.getmtime(fp)
                d = datetime.fromtimestamp(ts).date()
                records.append((d, cls))

    if not records:
        return pd.DataFrame(columns=['date', 'label', 'count'])

    df = pd.DataFrame(records, columns=['date', 'label'])
    return df.groupby(['label', 'date']).size().reset_index(name='count')

def create_prophet_series_for_label(agg_df, label):
    sub = agg_df[agg_df['label'] == label]
    if sub.empty:
        return None

    sub = sub.rename(columns={'date': 'ds', 'count': 'y'})
    sub['ds'] = pd.to_datetime(sub['ds'])
    sub = sub.set_index('ds').asfreq('D', fill_value=0).reset_index()

    return sub[['ds', 'y']]

def fit_and_forecast_prophet(series_df, periods=FORECAST_HORIZON):
    if series_df is None or len(series_df) < 5:
        return None

    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(series_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    out = forecast[['ds', 'yhat']].set_index('ds').yhat
    return out.tail(periods)

def compute_reorder_suggestion(current_stock, forecast_series, lead_days=LEAD_TIME_DAYS):
    if forecast_series is None or len(forecast_series) < lead_days:
        return False, None, None

    lead_forecast = float(forecast_series.iloc[:lead_days].sum())
    safety_stock = float(SAFETY_MULTIPLIER * float(forecast_series.iloc[:lead_days].std()))
    reorder_point = lead_forecast + safety_stock
    need_reorder = current_stock < reorder_point

    return need_reorder, reorder_point, lead_forecast

def find_image_files_recursive(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            for fname in fnmatch.filter(files, ext):
                images.append(os.path.join(root, fname))
    return sorted(images)

def main():
    ensure_dir(FORECAST_OUTPUT_DIR)

    class_names = build_class_names(TRAIN_IMAGE_ROOT)
    model = load_classification_model(CLASS_MODEL_PATH)
    img_size = infer_img_size_from_dataset(TRAIN_IMAGE_ROOT)

    agg = build_sales_history_from_images(TRAIN_IMAGE_ROOT)

    test_files = find_image_files_recursive(TEST_IMAGE_ROOT)
    if not test_files:
        print("No test images found.")
        return

    for fp in test_files:
        relative_name = os.path.relpath(fp, TEST_IMAGE_ROOT)
        label, prob = classify_image(model, class_names, fp, img_size)

        print(f"\nImage: {relative_name}")
        print(f"Detected: {label}  (prob={prob:.2f})")

        series_df = create_prophet_series_for_label(agg, label)
        if series_df is None:
            print("No historical data for this product.")
            continue

        forecast_series = fit_and_forecast_prophet(series_df)
        if forecast_series is None:
            print("Not enough data to forecast.")
            continue

        out_df = pd.DataFrame({"date": forecast_series.index, "yhat": forecast_series.values})
        out_path = os.path.join(FORECAST_OUTPUT_DIR, f"{label}_forecast.csv")
        out_df.to_csv(out_path, index=False)

        print("Forecast saved to:", out_path)

        current_stock = DEFAULT_CURRENT_STOCK

        need_reorder, reorder_point, lead_forecast = compute_reorder_suggestion(
            current_stock, forecast_series
        )

        print("Current stock:", current_stock)
        print(f"7-day demand: {lead_forecast:.2f}")
        print("Reorder point:", reorder_point)

        if need_reorder:
            print(">>> ALERT: Reorder recommended for:", label)
        else:
            print("Stock is sufficient.")

    print("\nDone")

if __name__ == "__main__":
    main()
