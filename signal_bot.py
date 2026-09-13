"""
EMA Crossover + RSI Signal Bot
Replicates the Pine Script strategy logic (ema_rsi_strategy_v4) in Python.
Checks BTC (via Binance, free) and XAU/USD (via Twelve Data, free tier).
Sends BUY/SELL alerts with Entry/SL/TP to Telegram.
Designed to run every 5 minutes via GitHub Actions (or any cron scheduler).
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ===== Config (edit these to match your Pine Script inputs) =====
FAST_LEN = 9
SLOW_LEN = 21
RSI_LEN = 14
RSI_LOW = 30
RSI_HIGH = 70

USE_ATR = True
ATR_LEN = 14
ATR_MULT = 1.5
FIXED_SL = 50.0          # used only if USE_ATR = False
RR_RATIO = 2.0

TIMEFRAME_MIN = 5         # 5-minute candles

# ===== Secrets (set these as environment variables / GitHub Secrets) =====
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_DATA_API_KEY = os.environ["TWELVE_DATA_API_KEY"]

STATE_FILE = "state.json"


# ---------------------------------------------------------------------------
# Helper: drop the still-forming (not yet closed) candle
# ---------------------------------------------------------------------------
def drop_unclosed_candle(df, time_is_close_time):
    """
    Pine Script's alert(..., alert.freq_once_per_bar_close) only fires once a
    candle is fully closed. If our fetched data's last row is still forming
    (the live/open candle), we must drop it -- otherwise we'll generate
    signals the indicator never actually shows, and they can flip or vanish
    once the candle finally closes.
    """
    now = datetime.now(timezone.utc)
    last_time = df.iloc[-1]["time"]
    if last_time.tzinfo is None:
        last_time = last_time.tz_localize("UTC")

    close_time = last_time if time_is_close_time else last_time + timedelta(minutes=TIMEFRAME_MIN)

    if close_time > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------
def fetch_btc_klines(limit=150):
    """
    Kraken's free public OHLC endpoint (no API key needed).
    Note: Binance blocks requests coming from cloud/server IPs (GitHub Actions,
    AWS, etc.) with an HTTP 451 error -- this is a Binance-side restriction,
    not a bug in this script. Kraken does not apply that block, so it's a
    more reliable choice when running from GitHub Actions.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "XBTUSD", "interval": TIMEFRAME_MIN}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")

    result_key = next(k for k in data["result"] if k != "last")
    rows = data["result"][result_key][-limit:]

    df = pd.DataFrame(rows, columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"
    ])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)  # candle OPEN time
    df = df[["time", "high", "low", "close"]]
    return drop_unclosed_candle(df, time_is_close_time=False)


def fetch_gold_klines(limit=150):
    """Twelve Data free tier. Needs API key. ~8 requests/min, 800/day on free plan."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": f"{TIMEFRAME_MIN}min",
        "outputsize": limit,
        "apikey": TWELVE_DATA_API_KEY,
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data error: {data}")
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "time"})
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)  # Twelve Data returns newest first
    df = df[["time", "high", "low", "close"]]
    return drop_unclosed_candle(df, time_is_close_time=False)


# ---------------------------------------------------------------------------
# Indicator calculations (mirrors the Pine Script)
# ---------------------------------------------------------------------------
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100)


def atr(df, length):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def compute_signal(df):
    df = df.copy()
    df["fastEMA"] = ema(df["close"], FAST_LEN)
    df["slowEMA"] = ema(df["close"], SLOW_LEN)
    df["rsi"] = rsi(df["close"], RSI_LEN)
    df["atr"] = atr(df, ATR_LEN)

    prev = df.iloc[-2]
    last = df.iloc[-1]

    bullish_cross = prev["fastEMA"] <= prev["slowEMA"] and last["fastEMA"] > last["slowEMA"]
    bearish_cross = prev["fastEMA"] >= prev["slowEMA"] and last["fastEMA"] < last["slowEMA"]

    rsi_ok = RSI_LOW < last["rsi"] < RSI_HIGH
    buy_signal = bullish_cross and rsi_ok
    sell_signal = bearish_cross and rsi_ok

    sl_distance = last["atr"] * ATR_MULT if USE_ATR else FIXED_SL
    entry = last["close"]

    if buy_signal:
        tp = entry + sl_distance * RR_RATIO
        sl = entry - sl_distance
        return "BUY", entry, sl, tp, last["time"]
    if sell_signal:
        tp = entry - sl_distance * RR_RATIO
        sl = entry + sl_distance
        return "SELL", entry, sl, tp, last["time"]
    return None, None, None, None, last["time"]


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def send_telegram(symbol, action, entry, sl, tp, candle_time, subscribers):
    text = (
        f"🔔 {action} Signal\n"
        f"Symbol: {symbol}\n"
        f"Entry: {entry:.2f}\n"
        f"SL: {sl:.2f}\n"
        f"TP: {tp:.2f}\n"
        f"Time: {candle_time}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chat_id in subscribers:
        try:
            resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            # Don't let one bad/blocked subscriber stop delivery to everyone else
            print(f"  -> Failed to send to {chat_id}: {e}")


def send_welcome(chat_id):
    text = (
        "✅ You're subscribed!\n"
        "You'll now receive BUY/SELL signals (with Entry/SL/TP) for "
        "BTCUSDT and XAU/USD automatically."
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15).raise_for_status()
    except Exception as e:
        print(f"  -> Failed to welcome {chat_id}: {e}")


# ---------------------------------------------------------------------------
# Subscriber management (anyone who sends /start to the bot gets added)
# ---------------------------------------------------------------------------
def register_new_subscribers(state):
    subscribers = state.setdefault("subscribers", [])
    offset = state.get("last_update_id", 0) + 1

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    try:
        resp = requests.get(url, params={"offset": offset, "timeout": 0}, timeout=15)
        resp.raise_for_status()
        updates = resp.json().get("result", [])
    except Exception as e:
        print(f"[subscribers] ERROR fetching updates: {e}")
        return

    max_update_id = state.get("last_update_id", 0)
    for update in updates:
        max_update_id = max(max_update_id, update["update_id"])
        message = update.get("message")
        if not message:
            continue
        text = (message.get("text") or "").strip().lower()
        chat_id = str(message["chat"]["id"])
        if text.startswith("/start") and chat_id not in subscribers:
            subscribers.append(chat_id)
            send_welcome(chat_id)
            print(f"[subscribers] Added new subscriber: {chat_id}")

    state["last_update_id"] = max_update_id
    state["subscribers"] = subscribers


# ---------------------------------------------------------------------------
# State (avoid sending the same candle's signal twice)
# ---------------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Helper: check if data is stale (market closed / no fresh candles coming in)
# ---------------------------------------------------------------------------
STALE_THRESHOLD_MIN = 20  # if the latest closed candle is older than this, treat market as closed


def is_market_stale(candle_time):
    now = datetime.now(timezone.utc)
    if candle_time.tzinfo is None:
        candle_time = candle_time.tz_localize("UTC")
    age_minutes = (now - candle_time).total_seconds() / 60
    return age_minutes > STALE_THRESHOLD_MIN


def is_flat_market(df, lookback=5, rel_epsilon=1e-5):
    """
    Some data providers keep emitting candles with fresh timestamps even
    while a market is closed, by repeating the last known price (a 'flat'
    candle with open == high == low == close, or near enough). Timestamp
    freshness alone can't catch this, so instead check whether the recent
    candles actually have any real price range. A genuinely open, liquid
    market essentially never prints several near-zero-range bars in a row.
    """
    recent = df.tail(lookback)
    price_level = recent["close"].iloc[-1]
    avg_range = (recent["high"] - recent["low"]).mean()
    return avg_range <= price_level * rel_epsilon


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_symbol(name, fetch_fn, state):
    try:
        df = fetch_fn()

        if is_flat_market(df):
            print(f"[{name}] Market appears closed/inactive (flat candles) -- skipping")
            return

        action, entry, sl, tp, candle_time = compute_signal(df)

        if is_market_stale(candle_time):
            print(f"[{name}] Market appears closed (last candle: {candle_time}) -- skipping")
            return

        candle_key = str(candle_time)

        if action and state.get(name) != candle_key:
            send_telegram(name, action, entry, sl, tp, candle_time, state.get("subscribers", []))
            state[name] = candle_key
            print(f"[{name}] Sent {action} signal at {candle_time} to {len(state.get('subscribers', []))} subscriber(s)")
        else:
            print(f"[{name}] No new signal (last candle: {candle_time})")
    except Exception as e:
        print(f"[{name}] ERROR: {e}")


def main():
    state = load_state()

    # Bootstrap: make sure the original/admin chat ID is always in the list
    subscribers = state.setdefault("subscribers", [])
    if TELEGRAM_CHAT_ID not in subscribers:
        subscribers.append(TELEGRAM_CHAT_ID)

    register_new_subscribers(state)  # picks up anyone who sent /start since last run

    process_symbol("BTCUSDT", fetch_btc_klines, state)
    process_symbol("XAU/USD", fetch_gold_klines, state)
    save_state(state)


if __name__ == "__main__":
    main()
