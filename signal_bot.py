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

# ---- Exit rule & reporting ----
EMA_FAST = 9                  # exit when EMA(fast) crosses EMA(slow) against the trade
EMA_SLOW = 21
PIP_SIZE = {"XAU/USD": 0.1, "BTCUSDT": 1.0}  # price move counted as 1 pip (adjust if yours differs)
RISK_PER_TRADE_USD = 10.0     # $ risked per trade (= 1R); only used to show $ in reports

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
    so the close message can later be sent as a reply to this exact message.
    There is no fixed TP any more: the exit comes from an EMA cross (see
    check_pending_trades), so only Entry and SL are shown."""
    text = (
        f"🔔 {action} Signal ({strategy_label})\n"
        f"Symbol: {symbol}\n"
        f"Entry: {entry:.2f}\n"
        f"SL: {sl:.2f}\n"
        f"Exit: EMA {EMA_FAST}/{EMA_SLOW} cross (I'll reply here when to close)\n"
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


def send_outcome_telegram(symbol, strategy_label, action, entry, exit_price, sl,
                          outcome, reason, r, pips, subscribers, message_ids=None):
    """Replies to the original signal message.
    reason='ema'  -> 'CLOSE NOW' message with the result in R and pips.
    reason='sl'   -> stop loss hit (or breakeven if price had reached 1:1 first)."""
    message_ids = message_ids or {}
    result_line = f"Result: {r:+.2f}R | {pips:+.1f} pips"
    if reason == "ema":
        tag = {"win": "✅ WIN", "loss": "❌ LOSS"}.get(outcome, "➖ BREAKEVEN")
        text = (
            f"🔔 CLOSE NOW - EMA {EMA_FAST}/{EMA_SLOW} crossed ({strategy_label})\n"
            f"Symbol: {symbol}\n"
            f"Direction: {action}\n"
            f"Entry: {entry:.2f} -> Exit: {exit_price:.2f}\n"
            f"{result_line} ({tag})"
        )
    elif outcome == "breakeven":
        text = (
            f"➖ BREAKEVEN - stopped out after reaching 1:1 ({strategy_label})\n"
            f"Symbol: {symbol}\n"
            f"Direction: {action}\n"
            f"Entry: {entry:.2f}  SL: {sl:.2f}\n"
            f"{result_line}"
        )
    else:
        text = (
            f"❌ STOP LOSS HIT ({strategy_label})\n"
            f"Symbol: {symbol}\n"
            f"Direction: {action}\n"
            f"Entry: {entry:.2f}  SL: {sl:.2f}\n"
            f"{result_line}"
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
        "You'll receive BUY/SELL signals (Entry + SL) for BTCUSDT and XAU/USD. "
        f"When the EMA {EMA_FAST}/{EMA_SLOW} crosses against a trade, I'll reply to that "
        "signal telling you to close it, with the result in R and pips. "
        "You'll also get daily, weekly, monthly & yearly performance reports."
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
# Pending trade outcome tracking (win / breakeven / loss)
# ---------------------------------------------------------------------------
def _events_in_candle(action, half_level, sl, row):
    """
    Levels touched inside one candle (the stop, and the 1:1 level used for the
    breakeven rule), ordered by distance from the candle's open -- an
    approximation of which came first, since OHLC alone can't tell us.
    """
    o, hi, lo = row["open"], row["high"], row["low"]
    events = []
    if action == "BUY":
        if lo <= sl:
            events.append(("sl", sl))
        if hi >= half_level:
            events.append(("half", half_level))
    else:  # SELL
        if hi >= sl:
            events.append(("sl", sl))
        if lo <= half_level:
            events.append(("half", half_level))
    events.sort(key=lambda e: abs(e[1] - o))
    return events


def check_pending_trades(name, df, state):
    """
    Walk forward through every candle after each open signal:
      - Stop hit: LOSS (-1R), or BREAKEVEN (0R) if price had already reached
        the 1:1 level (entry +/- risk) before coming back to the stop.
      - Otherwise, when EMA(fast) crosses EMA(slow) AGAINST the trade at a
        candle close, tell the subscribers to close now. Result is measured
        in R (profit / risk) and pips from entry to that candle's close.
    """
    pending = state.setdefault("pending_trades", [])
    if not pending:
        return

    ema_fast = ema(df["close"], EMA_FAST).to_numpy()
    ema_slow = ema(df["close"], EMA_SLOW).to_numpy()
    times = df["time"].tolist()

    still_pending = []
    for trade in pending:
        if trade["symbol"] != name:
            still_pending.append(trade)
            continue

        action, sl, entry = trade["action"], trade["sl"], trade["entry"]
        risk = abs(entry - sl)
        if risk <= 0:
            continue  # invalid record, drop it
        half_level = entry + risk if action == "BUY" else entry - risk
        entry_time = pd.to_datetime(trade["entry_time"], utc=True)

        reached_half = trade.get("reached_half", False)
        reason = outcome = exit_price = exit_time = None

        for i in range(1, len(df)):
            if times[i] <= entry_time:
                continue
            row = df.iloc[i]

            for kind, _level in _events_in_candle(action, half_level, sl, row):
                if kind == "half":
                    reached_half = True
                elif kind == "sl":
                    reason = "sl"
                    exit_price = sl
                    outcome = "breakeven" if reached_half else "loss"
                    break
            if reason:
                exit_time = times[i]
                break

            if action == "BUY":
                crossed = ema_fast[i - 1] >= ema_slow[i - 1] and ema_fast[i] < ema_slow[i]
            else:
                crossed = ema_fast[i - 1] <= ema_slow[i - 1] and ema_fast[i] > ema_slow[i]
            if crossed:
                reason = "ema"
                exit_price = float(row["close"])
                exit_time = times[i]
                break

        if reason is None:
            trade["reached_half"] = reached_half
            still_pending.append(trade)
            continue

        if reason == "sl":
            r = 0.0 if outcome == "breakeven" else -1.0
        else:
            move = (exit_price - entry) if action == "BUY" else (entry - exit_price)
            r = move / risk
            outcome = "win" if r > 0 else "loss" if r < 0 else "breakeven"
        pips = r * risk / _pip_size(name)

        send_outcome_telegram(
            name, trade["strategy_label"], action, entry, exit_price, sl,
            outcome, reason, r, pips, state.get("subscribers", []),
            trade.get("message_ids", {})
        )
        state.setdefault("trade_history", []).append({
            "symbol": name,
            "strategy_label": trade["strategy_label"],
            "action": action,
            "outcome": outcome,
            "reason": reason,
            "entry": entry,
            "exit": exit_price,
            "r": round(r, 4),
            "pips": round(pips, 2),
            "entry_time": trade["entry_time"],
            "resolved_time": str(exit_time),
        })
        print(f"[{name}][{trade['strategy_label']}] Trade opened {trade['entry_time']} "
              f"closed via {reason.upper()}: {outcome.upper()} ({r:+.2f}R, {pips:+.1f} pips)")

    state["pending_trades"] = still_pending


# ---------------------------------------------------------------------------
# Periodic performance reports (daily / weekly / monthly / yearly)
# ---------------------------------------------------------------------------
def build_report_text(title, trades):
    if not trades:
        return f"📊 {title}\nNo trades in this period."

    sep = "━━━━━━━━━━━━━━━━"
    symbols = sorted({t["symbol"] for t in trades})
    strategies = sorted({t["strategy_label"] for t in trades})
    count_line = " | ".join(
        f"{sym}: {sum(1 for t in trades if t['symbol'] == sym)}" for sym in symbols
    )

    parts = [
        f"📊 {title}",
        sep,
        f"Trades by symbol: {count_line} | Total: {len(trades)}",
        "",
        _format_block("📌 Overall", trades),
    ]
    for sym in symbols:
        icon = "🥇" if "XAU" in sym else "🪙"
        parts += ["", sep, _format_block(f"{icon} {sym}", [t for t in trades if t["symbol"] == sym])]
    for label in strategies:
        parts += ["", sep, _format_block(f"🎯 {label}", [t for t in trades if t["strategy_label"] == label])]
    parts += ["", f"ℹ️ $ values assume ${RISK_PER_TRADE_USD:.0f} risked per trade (1R)."]
    return "\n".join(parts)


def _pip_size(symbol):
    return PIP_SIZE.get(symbol, 1.0)


def _usd(r):
    v = r * RISK_PER_TRADE_USD
    return f"{'+' if v >= 0 else '-'}${abs(v):.2f}"


def _stats(group):
    total = len(group)
    wins = sum(1 for t in group if t["outcome"] == "win")
    breakevens = sum(1 for t in group if t["outcome"] == "breakeven")
    losses = sum(1 for t in group if t["outcome"] == "loss")

    rated = [t for t in group if t.get("r") is not None]
    rs = [t["r"] for t in rated]
    pips = [(t.get("pips") or 0.0) for t in rated]
    pos = [x for x in rs if x > 0]
    neg = [x for x in rs if x < 0]

    gross_profit = sum(pos)
    gross_loss = -sum(neg)
    avg_win = (sum(pos) / len(pos)) if pos else None
    avg_loss = (sum(neg) / len(neg)) if neg else None
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None
    avg_rr = (avg_win / abs(avg_loss)) if (avg_win is not None and avg_loss is not None) else None

    return {
        "total": total, "wins": wins, "breakevens": breakevens, "losses": losses,
        "win_rate": (wins / total * 100) if total else 0.0,
        "avg_win": avg_win, "avg_loss": avg_loss, "avg_rr": avg_rr,
        "profit_factor": profit_factor,
        "net_r": sum(rs), "net_pips": sum(pips),
    }


def _format_block(title, group):
    s = _stats(group)
    pf = s["profit_factor"]
    pf_text = "N/A" if pf is None else ("∞" if pf == float("inf") else f"{pf:.2f}")
    rr_text = f"1 : {s['avg_rr']:.2f}" if s["avg_rr"] is not None else "N/A"
    win_text = f"{s['avg_win']:+.2f}R ({_usd(s['avg_win'])})" if s["avg_win"] is not None else "N/A"
    loss_text = f"{s['avg_loss']:+.2f}R ({_usd(s['avg_loss'])})" if s["avg_loss"] is not None else "N/A"
    return "\n".join([
        title,
        f"Trades: {s['total']}  (✅ {s['wins']} | ➖ {s['breakevens']} | ❌ {s['losses']})",
        f"Win Rate: {s['win_rate']:.1f}%",
        f"Avg R:R: {rr_text}",
        f"Profit Factor: {pf_text}",
        f"Avg Win: {win_text}",
        f"Avg Loss: {loss_text}",
        f"Net: {s['net_r']:+.2f}R ({_usd(s['net_r'])}) | {s['net_pips']:+.1f} pips",
    ])


def send_report_telegram(title, trades, subscribers):
    text = build_report_text(title, trades)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chat_id in subscribers:
        try:
            resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"  -> Failed to send report to {chat_id}: {e}")


def _trades_in_range(history, start, end):
    out = []
    for t in history:
        resolved = pd.to_datetime(t["resolved_time"], utc=True)
        if start <= resolved < end:
            out.append(t)
    return out


def check_and_send_periodic_reports(state):
    history = state.get("trade_history", [])
    now = datetime.now(timezone.utc)
    subscribers = state.get("subscribers", [])

    # 10 PM Bangladesh time (UTC+6) = 16:00 UTC
    REPORT_HOUR_UTC = 16

    today = now.date()
    if state.get("last_daily_report") != str(today):
        if state.get("last_daily_report") is not None:  # skip the very first run
            yesterday_start = datetime.combine(today - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
            yesterday_end = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
            trades = _trades_in_range(history, yesterday_start, yesterday_end)
            send_report_telegram(f"Daily Report -- {yesterday_start.date()}", trades, subscribers)
        state["last_daily_report"] = str(today)

    # ----- Weekly: fires Friday night (~10 PM BD time), covering Mon-Fri of this week -----
    iso_year, iso_week, _ = now.isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"
    if now.weekday() == 4 and now.hour >= REPORT_HOUR_UTC and state.get("last_weekly_report") != week_key:
        week_start = datetime.combine(now.date() - timedelta(days=now.weekday()), datetime.min.time(), tzinfo=timezone.utc)
        trades = _trades_in_range(history, week_start, now)
        send_report_telegram(f"Weekly Report -- {week_start.date()} to {now.date()}", trades, subscribers)
        state["last_weekly_report"] = week_key

    # ----- Monthly: fires on the last calendar day of the month (~10 PM BD time) -----
    is_last_day_of_month = (now.date() + timedelta(days=1)).month != now.month
    month_key = f"{now.year}-{now.month:02d}"
    if is_last_day_of_month and now.hour >= REPORT_HOUR_UTC and state.get("last_monthly_report") != month_key:
        month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        trades = _trades_in_range(history, month_start, now)
        send_report_telegram(f"Monthly Report -- {month_start.strftime('%B %Y')}", trades, subscribers)
        state["last_monthly_report"] = month_key

    year_key = str(now.year)
    if state.get("last_yearly_report") != year_key:
        if state.get("last_yearly_report") is not None:
            this_year_start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
            last_year_start = datetime(now.year - 1, 1, 1, tzinfo=timezone.utc)
            trades = _trades_in_range(history, last_year_start, this_year_start)
            send_report_telegram(f"Yearly Report -- {now.year - 1}", trades, subscribers)
        state["last_yearly_report"] = year_key


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

    check_and_send_periodic_reports(state)

    save_state(state)


if __name__ == "__main__":
    main()
