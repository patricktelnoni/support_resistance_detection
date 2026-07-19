import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Try to import TensorFlow; if unavailable, fall back to a sklearn classifier
USE_TF = True
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    USE_TF = False
    from sklearn.ensemble import RandomForestClassifier


TICKER = "BBCA.JK"
START = "2025-07-01"
END = "2026-06-30"
PIVOT_WINDOW = 5
K_RANGE = range(2, 16)
LOOKBACK = 20
FUTURE_DAYS = 1
ZONE_TOLERANCE = 0.0  # level zone already has low/high from pivots


def find_pivots(df, w=PIVOT_WINDOW):
    high_series = df["High"]
    low_series = df["Low"]

    roll_max = high_series.rolling(2 * w + 1, center=True).max()
    roll_min = low_series.rolling(2 * w + 1, center=True).min()

    # Create boolean masks
    is_pivot_high = (high_series == roll_max) & roll_max.notna()
    is_pivot_low = (low_series == roll_min) & roll_min.notna()

    # Get the index (dates) where the condition is True by filtering the original DataFrame
    # Then extract the High/Low values using .loc with these indices
    pivot_high_idx = df[is_pivot_high].index
    pivot_low_idx = df[is_pivot_low].index

    all_dates = []
    all_prices = []
    all_types = []

    for date in pivot_high_idx:
        all_dates.append(date)
        # Use .item() to ensure a scalar is extracted, addressing FutureWarning
        all_prices.append(df.loc[date, "High"].item()) 
        all_types.append("high")

    for date in pivot_low_idx:
        all_dates.append(date)
        # Use .item() to ensure a scalar is extracted, addressing FutureWarning
        all_prices.append(df.loc[date, "Low"].item())  
        all_types.append("low")

    points = pd.DataFrame({
        "date": all_dates,
        "price": all_prices,
        "type": all_types
    })
    return points.sort_values("date").reset_index(drop=True)


def evaluate_k(prices, k_range=K_RANGE):
    X = prices.reshape(-1, 1)
    rows = []
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_) if k >= 2 else np.nan,
        })
    return pd.DataFrame(rows).set_index("k")


def elbow_k(inertia: pd.Series) -> int:
    k = inertia.index.values.astype(float)
    y = inertia.values.astype(float)
    kn = (k - k.min()) / (k.max() - k.min())
    yn = (y - y.min()) / (y.max() - y.min())
    x1, y1, x2, y2 = kn[0], yn[0], kn[-1], yn[-1]
    dist = np.abs((y2 - y1) * kn - (x2 - x1) * yn + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)
    return int(k[dist.argmax()])


def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def prepare_sequences(X, y, lookback=LOOKBACK):
    seq_X, seq_y = [], []
    for i in range(len(X) - lookback - FUTURE_DAYS + 1):
        seq_X.append(X[i : i + lookback])
        seq_y.append(y[i + lookback - 1])
    return np.array(seq_X), np.array(seq_y)


def main():
    df = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=False) # Align with user's notebook cell 7fb50706
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    print(f"Loaded {len(df)} rows")

    points = find_pivots(df)
    # Explicitly convert 'price' column to numeric
    points['price'] = pd.to_numeric(points['price'])
    if points.empty:
        raise SystemExit("No pivot points found; adjust PIVOT_WINDOW or date range")

    evalu = evaluate_k(points["price"].values)
    k_elbow = elbow_k(evalu["inertia"]) if not evalu.empty else 3
    k = k_elbow
    print(f"Using k={k} (elbow)")

    Xp = points["price"].values.reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xp)
    points = points.assign(cluster=km.labels_)
    levels = (
        points.groupby("cluster")["price"]
        .agg(level="mean", n_touch="size", low="min", high="max")
        .sort_values("level")
        .reset_index(drop=True)
    )

    last_price = float(df["Close"].iloc[-1])
    levels["zona"] = np.where(levels["level"] < last_price, "SUPPORT", "RESISTANCE")

    # Label each day with zone if close within cluster's low-high range
    def zone_for_price(p):
        hits = levels[(levels["low"] - 1e-9 <= p) & (p <= levels["high"] + 1e-9)]
        if len(hits):
            # if multiple, pick nearest level
            idx = (hits["level"].sub(p).abs()).idxmin()
            return hits.loc[idx, "zona"]
        return "NONE"

    df = df.assign(zone=df["Close"].apply(zone_for_price))

    # Create directional target: next day close > today close -> 1 else 0
    df["future_close"] = df["Close"].shift(-FUTURE_DAYS)
    df["target"] = (df["future_close"] > df["Close"]).astype(int)
    df = df.dropna(subset=["future_close"]).copy()

    # Features: log returns, MA5, MA10, Volume, zone one-hot
    df["ret"] = np.log(df["Close"]).diff().fillna(0)
    df["ma5"] = df["Close"].rolling(5).mean().fillna(method="bfill")
    df["ma10"] = df["Close"].rolling(10).mean().fillna(method="bfill")

    zones = pd.get_dummies(df["zone"]).reindex(columns=["NONE", "SUPPORT", "RESISTANCE"], fill_value=0)
    feats = pd.concat([
        df[["ret", "ma5", "ma10", "Volume" ]].reset_index(drop=True),
        zones.reset_index(drop=True),
    ], axis=1)

    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(feats)
    ys = df["target"].values

    X_seq, y_seq = prepare_sequences(Xs, ys)
    if len(X_seq) == 0:
        raise SystemExit("Not enough data for the chosen LOOKBACK")

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)
    print("Shapes", X_train.shape, X_test.shape)

    if USE_TF:
        model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=80, batch_size=16, callbacks=[es], verbose=2)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss={loss:.4f}, acc={acc:.4f}")
        preds = (model.predict(X_test) > 0.5).astype(int).reshape(-1)
        print("Sample predictions:", preds[:20])
        model.save("bbca_lstm_model.h5")
        print("Model saved to bbca_lstm_model.h5")
    else:
        # Flatten sequences for sklearn classifier
        nsamples, nt, nf = X_train.shape
        X_train_flat = X_train.reshape((nsamples, nt * nf))
        X_test_flat = X_test.reshape((X_test.shape[0], nt * nf))
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train_flat, y_train)
        acc = clf.score(X_test_flat, y_test)
        print(f"RandomForest test accuracy: {acc:.4f} (TensorFlow not available)")
        preds = clf.predict(X_test_flat)
        print("Sample predictions:", preds[:20])


if __name__ == "__main__":
    main()