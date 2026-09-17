"""
Multi-Strategy Signal Bot
Runs two independent signal engines side by side for each symbol:

  1. Black Shadow Trader (merged) -- Range Filter trend + Scalper Pro
     pivot breakout, confirmed by trend direction.
  2. Consolidation Breakout -- detects a tight consolidation (>=4
     consecutive low-range/small-body candles), then fires only when a
     candle CLOSES outside that range (a wick poking out and closing back
     inside does NOT count). SL = opposite side of the consolidation box,
     TP = 1:2 risk:reward.

Checks BTC (via Kraken, free) and XAU/USD (via Twelve Data, free tier).
Sends BUY/SELL alerts with Entry/SL/TP to all Telegram subscribers.
Designed to run every 5 minutes (via GitHub Actions + an external cron trigger).
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ===== Shared config =====
TIMEFRAME_MIN = 3
FETCH_LIMIT = 500  # generous warm-up window for the Range Filter's long EMA

# ---- Strategy 1: Black Shadow Trader (Range Filter + Scalper Pro, merged) ----
RF_PERIOD = 100
RF_MULT = 3.0
TARGET_MULT = 2.0
PIVOT_LENGTH = 3
USE_CONSOLIDATION = True
CONS_LENGTH = 10
CONS_ATR_MULT = 3.0
ATR_LEN = 14
USE_COOLDOWN = True
COOLDOWN_BARS = 30
USE_MERGE = True

# ---- Strategy 2: Consolidation Breakout ----
CANDLE_CONFIRM_BARS = 4       # min. consecutive consolidating candles
COMPRESSION_BODY_MAX = 0.5    # body must be < 50% of the candle's range
COMPRESSION_ATR_LEN = 4
BREAKOUT_TARGET_MULT = 2.0    # 1:2 risk:reward, same as the other strategy

# ===== Secrets (set as environment variables / GitHub Secrets) =====
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_DATA_API_KEY = os.environ["TWELVE_DATA_API_KEY"]

STATE_FILE = "state.json"


# ---------------------------------------------------------------------------
# Helper: drop the still-forming (not yet closed) candle
# ---------------------------------------------------------------------------
def drop_unclosed_candle(df, time_is_close_time):
    now = datetime.now(timezone.utc)
    last_time = df.iloc[-1]["time"]
    if last_time.tzinfo is None:
        last_time = last_time.tz_localize("UTC")
    close_time = last_time if time_is_close_time else last_time + timedelta(minutes=TIMEFRAME_MIN)
    if close_time > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def resample_ohlc(df_1m, minutes):
    """
    Neither Kraken nor Twelve Data offers a native 3-minute interval, so we
    fetch 1-minute candles and combine every N of them into one N-minute
    candle ourselves (open=first, high=max, low=min, close=last).
    """
    df_1m = df_1m.set_index("time")
    agg = df_1m.resample(f"{minutes}min", label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna().reset_index()
    return agg


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------
def fetch_btc_klines(limit=FETCH_LIMIT):
    """
    Kraken's free public OHLC endpoint (no API key needed, not blocked on
    cloud IPs). Kraken only supports fixed intervals (1, 5, 15... minutes),
    so we pull 1-minute candles and resample to TIMEFRAME_MIN ourselves.
    Note: Kraken's OHLC endpoint only returns roughly the most recent ~720
    one-minute candles (~12 hours) per call -- there is no free way to pull
    deeper 1-minute history from them, so the resampled series will be
    shorter than FETCH_LIMIT until Kraken's window naturally covers more.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "XBTUSD", "interval": 1}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")

    result_key = next(k for k in data["result"] if k != "last")
    rows = data["result"][result_key]

    df = pd.DataFrame(rows, columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"
    ])
    df["open"] = df["open"].astype(float)
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["time", "open", "high", "low", "close"]]

    df = resample_ohlc(df, TIMEFRAME_MIN)
    df = df.tail(limit).reset_index(drop=True)
    return drop_unclosed_candle(df, time_is_close_time=False)


def fetch_gold_klines(limit=FETCH_LIMIT):
    """
    Twelve Data free tier. Needs API key. Twelve Data also has no native
    3-minute interval, so we pull 1-minute candles and resample ourselves.
    """
    raw_needed = min(5000, limit * TIMEFRAME_MIN + 50)
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": "1min",
        "outputsize": raw_needed,
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
    df["open"] = df["open"].astype(float)
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df = df[["time", "open", "high", "low", "close"]]

    df = resample_ohlc(df, TIMEFRAME_MIN)
    df = df.tail(limit).reset_index(drop=True)
    return drop_unclosed_candle(df, time_is_close_time=False)


# ---------------------------------------------------------------------------
# Shared indicator helpers
# ---------------------------------------------------------------------------
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def atr(df, length):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


# ---------------------------------------------------------------------------
# Strategy 1: Black Shadow Trader (Range Filter + Scalper Pro, merged)
# ---------------------------------------------------------------------------
def compute_black_shadow_signal(df):
    n = len(df)
    min_bars = RF_PERIOD * 2 + CONS_LENGTH + PIVOT_LENGTH * 2 + 10
    if n < min_bars:
        return None, None, None, None, df.iloc[-1]["time"]

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    open_ = df["open"].to_numpy()

    # ----- Range Filter -----
    src = close
    diff = np.abs(np.diff(src, prepend=src[0]))
    avrng = pd.Series(diff).ewm(span=RF_PERIOD, adjust=False).mean().to_numpy()
    wper = RF_PERIOD * 2 - 1
    smrng = pd.Series(avrng).ewm(span=wper, adjust=False).mean().to_numpy() * RF_MULT

    filt = np.zeros(n)
    filt[0] = src[0]
    for i in range(1, n):
        x, r, prev = src[i], smrng[i], filt[i - 1]
        if x > prev:
            filt[i] = prev if (x - r) < prev else (x - r)
        else:
            filt[i] = prev if (x + r) > prev else (x + r)

    upward = np.zeros(n)
    downward = np.zeros(n)
    for i in range(1, n):
        if filt[i] > filt[i - 1]:
            upward[i] = upward[i - 1] + 1
            downward[i] = 0
        elif filt[i] < filt[i - 1]:
            downward[i] = downward[i - 1] + 1
            upward[i] = 0
        else:
            upward[i] = upward[i - 1]
            downward[i] = downward[i - 1]

    atr_vals = atr(df, ATR_LEN).to_numpy()

    # ----- Pivot High/Low (confirmed with a lag, like Pine's ta.pivothigh/low) -----
    L = PIVOT_LENGTH
    last_pivot_high = np.full(n, np.nan)
    last_pivot_low = np.full(n, np.nan)
    cur_ph, cur_pl = np.nan, np.nan
    for i in range(n):
        c = i - L
        if c - L >= 0:
            window_high = high[c - L:c + L + 1]
            window_low = low[c - L:c + L + 1]
            if high[c] == window_high.max():
                cur_ph = high[c]
            if low[c] == window_low.min():
                cur_pl = low[c]
        last_pivot_high[i] = cur_ph
        last_pivot_low[i] = cur_pl

    # ----- Consolidation filter -----
    past_high = pd.Series(high).shift(1).rolling(CONS_LENGTH).max().to_numpy()
    past_low = pd.Series(low).shift(1).rolling(CONS_LENGTH).min().to_numpy()
    zone_range = past_high - past_low
    if USE_CONSOLIDATION:
        is_consolidating = np.where(np.isnan(zone_range), True, zone_range <= atr_vals * CONS_ATR_MULT)
    else:
        is_consolidating = np.ones(n, dtype=bool)

    # ----- Cooldown + breakout detection (replayed bar by bar) -----
    last_trade_bar = None
    triggered_action, triggered_entry, triggered_sl, triggered_tp, triggered_bar = (None,) * 5

    for i in range(2 * L, n):
        if np.isnan(last_pivot_high[i]) or np.isnan(last_pivot_low[i]):
            continue
        if np.isnan(last_pivot_high[i - 1]) or np.isnan(last_pivot_low[i - 1]):
            continue

        is_bull_candle = close[i] > open_[i]
        is_bear_candle = close[i] < open_[i]

        crossover_high = close[i - 1] <= last_pivot_high[i - 1] and close[i] > last_pivot_high[i]
        crossunder_low = close[i - 1] >= last_pivot_low[i - 1] and close[i] < last_pivot_low[i]

        can_enter = True
        if USE_COOLDOWN:
            can_enter = (last_trade_bar is None) or (i - last_trade_bar >= COOLDOWN_BARS)

        bullish_breakout = (
            is_bull_candle and crossover_high and (low[i] > last_pivot_low[i])
            and is_consolidating[i] and can_enter
        )
        bearish_breakout = (
            is_bear_candle and crossunder_low and (high[i] < last_pivot_high[i])
            and is_consolidating[i] and can_enter
        )

        merged_bull = bullish_breakout and (not USE_MERGE or upward[i] > 0)
        merged_bear = bearish_breakout and (not USE_MERGE or downward[i] > 0)

        if merged_bull:
            entry, stop = close[i], last_pivot_low[i]
            risk = entry - stop
            if risk > 0:
                last_trade_bar = i
                triggered_action = "BUY"
                triggered_entry, triggered_sl = entry, stop
                triggered_tp = entry + risk * TARGET_MULT
                triggered_bar = i
        elif merged_bear:
            entry, stop = close[i], last_pivot_high[i]
            risk = stop - entry
            if risk > 0:
                last_trade_bar = i
                triggered_action = "SELL"
                triggered_entry, triggered_sl = entry, stop
                triggered_tp = entry - risk * TARGET_MULT
                triggered_bar = i

    candle_time = df.iloc[-1]["time"]
    if triggered_action and triggered_bar == n - 1:
        return triggered_action, triggered_entry, triggered_sl, triggered_tp, candle_time
    return None, None, None, None, candle_time


# ---------------------------------------------------------------------------
# Strategy 2: Consolidation Breakout (Consolidation DNA, simplified)
# ---------------------------------------------------------------------------
def compute_consolidation_break_signal(df):
    """
    - A candle is 'compressing' if its body is < 50% of its range AND its
      range is smaller than ATR(4) (matches the Pine script's compressionBar).
    - Once >= CANDLE_CONFIRM_BARS consecutive compressing candles occur, the
      box (highest high / lowest low of those candles) freezes as the
      consolidation zone.
    - We then wait for a candle to CLOSE outside that zone (a wick poking
      out and closing back inside does NOT count -- 'Close Break' only).
    - Entry = breakout candle's close. SL = the opposite side of the zone.
      TP = 1:2 risk:reward.
    """
    n = len(df)
    min_bars = COMPRESSION_ATR_LEN + CANDLE_CONFIRM_BARS + 5
    if n < min_bars:
        return None, None, None, None, df.iloc[-1]["time"]

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    open_ = df["open"].to_numpy()

    atr4 = atr(df, COMPRESSION_ATR_LEN).to_numpy()

    body = np.abs(close - open_)
    rng = np.maximum(high - low, 1e-9)
    small_body = body < (rng * COMPRESSION_BODY_MAX)
    atr_compress = rng < atr4
    compression_bar = small_body & atr_compress

    cons_count = 0
    cons_high = cons_low = None
    mature = False
    range_high = range_low = None

    triggered_action, triggered_entry, triggered_sl, triggered_tp, triggered_bar = (None,) * 5

    for i in range(n):
        if not mature:
            if compression_bar[i]:
                cons_count += 1
                if cons_high is None:
                    cons_high, cons_low = high[i], low[i]
                else:
                    cons_high = max(cons_high, high[i])
                    cons_low = min(cons_low, low[i])
                if cons_count >= CANDLE_CONFIRM_BARS:
                    mature = True
                    range_high, range_low = cons_high, cons_low
            else:
                cons_count = 0
                cons_high = cons_low = None
        else:
            # Mature: wait for a close outside the frozen range (close break only)
            if close[i] > range_high:
                entry = close[i]
                sl = range_low
                risk = entry - sl
                if risk > 0:
                    triggered_action = "BUY"
                    triggered_entry, triggered_sl = entry, sl
                    triggered_tp = entry + risk * BREAKOUT_TARGET_MULT
                    triggered_bar = i
                mature = False
                cons_count = 0
                cons_high = cons_low = None
            elif close[i] < range_low:
                entry = close[i]
                sl = range_high
                risk = sl - entry
                if risk > 0:
                    triggered_action = "SELL"
                    triggered_entry, triggered_sl = entry, sl
                    triggered_tp = entry - risk * BREAKOUT_TARGET_MULT
                    triggered_bar = i
                mature = False
                cons_count = 0
                cons_high = cons_low = None
            # if the candle wicks outside but closes back inside the range,
            # nothing happens here -- the range simply stays active (correct:
            # a wick-only poke must NOT count as a break)

    candle_time = df.iloc[-1]["time"]
    if triggered_action and triggered_bar == n - 1:
        return triggered_action, triggered_entry, triggered_sl, triggered_tp, candle_time
    return None, None, None, None, candle_time


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def send_telegram(symbol, action, entry, sl, tp, candle_time, subscribers, strategy_label):
    """Sends the signal and returns {chat_id: message_id} for successful sends,
    so the eventual SL/TP outcome can be sent as a reply to this exact message."""
    text = (
        f"🔔 {action} Signal ({strategy_label})\n"
        f"Symbol: {symbol}\n"
        f"Entry: {entry:.2f}\n"
        f"SL: {sl:.2f}\n"
        f"TP: {tp:.2f}\n"
        f"Time: {candle_time}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    message_ids = {}
    for chat_id in subscribers:
        try:
            resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15)
            resp.raise_for_status()
            message_ids[str(chat_id)] = resp.json()["result"]["message_id"]
        except Exception as e:
            print(f"  -> Failed to send to {chat_id}: {e}")
    return message_ids


def send_outcome_telegram(symbol, strategy_label, action, entry, sl, tp, hit, subscribers, message_ids=None):
    """Sends the SL/TP outcome as a reply to the original signal message
    (falls back to a normal message if we don't have that message_id, e.g.
    for a subscriber who joined after the signal was sent)."""
    message_ids = message_ids or {}
    if hit == "tp":
        header = "✅ TARGET HIT (Profit)"
        rr_text = "Result: +2R (hit TP)"
    else:
        header = "❌ STOP LOSS HIT (Loss)"
        rr_text = "Result: -1R (hit SL)"
    text = (
        f"{header} ({strategy_label})\n"
        f"Symbol: {symbol}\n"
        f"Direction: {action}\n"
        f"Entry: {entry:.2f}  SL: {sl:.2f}  TP: {tp:.2f}\n"
        f"{rr_text}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chat_id in subscribers:
        payload = {"chat_id": chat_id, "text": text}
        reply_id = message_ids.get(str(chat_id))
        if reply_id:
            payload["reply_to_message_id"] = reply_id
        try:
            resp = requests.post(url, data=payload, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"  -> Failed to send outcome to {chat_id}: {e}")


def send_welcome(chat_id):
    text = (
        "✅ You're subscribed!\n"
        "You'll now receive BUY/SELL signals (with Entry/SL/TP) for "
        "BTCUSDT and XAU/USD automatically, plus a follow-up once each "
        "trade hits its Stop Loss or Take Profit."
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
# Market activity checks
# ---------------------------------------------------------------------------
STALE_THRESHOLD_MIN = 20


def is_market_stale(candle_time):
    now = datetime.now(timezone.utc)
    if candle_time.tzinfo is None:
        candle_time = candle_time.tz_localize("UTC")
    age_minutes = (now - candle_time).total_seconds() / 60
    return age_minutes > STALE_THRESHOLD_MIN


def is_flat_market(df, lookback=5, rel_epsilon=1e-5):
    recent = df.tail(lookback)
    price_level = recent["close"].iloc[-1]
    avg_range = (recent["high"] - recent["low"]).mean()
    return avg_range <= price_level * rel_epsilon


# ---------------------------------------------------------------------------
# Pending trade outcome tracking (did the last signal hit SL or TP?)
# ---------------------------------------------------------------------------
def check_pending_trades(name, df, state):
    """
    For every open (unresolved) signal on this symbol, scan every candle
    since entry to see whether price touched SL or TP first. If both are
    touched within the SAME candle, we can't know the true order from
    OHLC alone -- as a conservative approximation we assume whichever
    level is closer to that candle's open was hit first.
    """
    pending = state.setdefault("pending_trades", [])
    if not pending:
        return

    still_pending = []
    for trade in pending:
        if trade["symbol"] != name:
            still_pending.append(trade)
            continue

        entry_time = pd.to_datetime(trade["entry_time"], utc=True)
        subset = df[df["time"] > entry_time]

        resolved = False
        for _, row in subset.iterrows():
            hi, lo, op = row["high"], row["low"], row["open"]
            action, sl, tp = trade["action"], trade["sl"], trade["tp"]

            if action == "BUY":
                hit_tp = hi >= tp
                hit_sl = lo <= sl
            else:  # SELL
                hit_tp = lo <= tp
                hit_sl = hi >= sl

            if hit_tp and hit_sl:
                # Both touched in the same candle -- approximate by
                # whichever level is closer to that candle's open.
                outcome = "tp" if abs(tp - op) < abs(sl - op) else "sl"
            elif hit_tp:
                outcome = "tp"
            elif hit_sl:
                outcome = "sl"
            else:
                continue

            send_outcome_telegram(
                name, trade["strategy_label"], action,
                trade["entry"], sl, tp, outcome, state.get("subscribers", []),
                trade.get("message_ids", {})
            )
            print(f"[{name}][{trade['strategy_label']}] Trade opened {trade['entry_time']} "
                  f"resolved: {outcome.upper()}")
            resolved = True
            break

        if not resolved:
            still_pending.append(trade)

    state["pending_trades"] = still_pending


# ---------------------------------------------------------------------------
# State
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
# Main
# ---------------------------------------------------------------------------
STRATEGIES = [
    ("rf_scalper", "Range Filter + Scalper Pro", compute_black_shadow_signal),
    ("consol", "Consolidation Breakout", compute_consolidation_break_signal),
]


def process_symbol(name, fetch_fn, state):
    try:
        df = fetch_fn()
    except Exception as e:
        print(f"[{name}] ERROR fetching data: {e}")
        return

    if is_flat_market(df):
        print(f"[{name}] Market appears closed/inactive (flat candles) -- skipping")
        return

    # First check if any previously-sent signal has now hit SL or TP
    check_pending_trades(name, df, state)

    for key_suffix, label, strategy_fn in STRATEGIES:
        try:
            action, entry, sl, tp, candle_time = strategy_fn(df)

            if is_market_stale(candle_time):
                print(f"[{name}][{label}] Market appears closed (last candle: {candle_time}) -- skipping")
                continue

            state_key = f"{name}_{key_suffix}"
            candle_key = str(candle_time)

            if action and state.get(state_key) != candle_key:
                message_ids = send_telegram(name, action, entry, sl, tp, candle_time, state.get("subscribers", []), label)
                state[state_key] = candle_key
                state.setdefault("pending_trades", []).append({
                    "symbol": name,
                    "strategy_label": label,
                    "action": action,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "entry_time": str(candle_time),
                    "message_ids": message_ids,
                })
                print(f"[{name}][{label}] Sent {action} signal at {candle_time} "
                      f"to {len(state.get('subscribers', []))} subscriber(s)")
            else:
                print(f"[{name}][{label}] No new signal (last candle: {candle_time})")
        except Exception as e:
            print(f"[{name}][{label}] ERROR: {e}")


def main():
    state = load_state()

    subscribers = state.setdefault("subscribers", [])
    if TELEGRAM_CHAT_ID not in subscribers:
        subscribers.append(TELEGRAM_CHAT_ID)

    register_new_subscribers(state)

    process_symbol("BTCUSDT", fetch_btc_klines, state)
    process_symbol("XAU/USD", fetch_gold_klines, state)
    save_state(state)


if __name__ == "__main__":
    main()
