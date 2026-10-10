"""
Multi-Strategy Signal Bot
Runs five independent signal engines side by side for each symbol:

  1. Black Shadow Trader (merged) -- Range Filter trend + Scalper Pro
     pivot breakout, confirmed by trend direction.
  2. Consolidation Breakout -- detects a tight consolidation (>=4
     consecutive low-range/small-body candles), then fires only when a
     candle CLOSES outside that range (a wick poking out and closing back
     inside does NOT count). SL = opposite side of the consolidation box,
     TP = 1:2 risk:reward.
  3. AMD Po3 (Accumulation -> Manipulation -> Distribution) -- python port
     of the "AMD Po3 with Live Edge Stats" Pine indicator. Detects a
     compressed range, waits for one side to be swept (liquidity grab) and
     for price to RETURN inside the range, then enters in the opposite
     direction. SL = beyond the sweep extreme + ATR buffer.
  4. Order Block -- a strong impulse candle that breaks structure creates an
     order block (the last opposite candle before it). Entry = the candle that
     creates the block, SL = the order-block candle's low (BUY) / high (SELL),
     TP = 1:2 risk:reward.
  5. Range Filter Flip -- the Range Filter's Buy/Sell flip signals. SL = the
     previous swing low (BUY) / swing high (SELL). No target: the trade is
     closed when the OPPOSITE signal appears, and the new trade opens in the
     new direction.

Checks BTC (via Kraken, free) and XAU/USD (via Twelve Data, free tier).
Sends BUY/SELL alerts with Entry/SL/TP to all Telegram subscribers.
Designed to run every 3 minutes, matching the 3-minute candles (via GitHub Actions + an
external cron-job.org trigger).
"""

import os
import json
import math
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

# ---- Strategy 3: AMD Po3 (port of "AMD Po3 with Live Edge Stats") ----
# All values below are the ORIGINAL indicator defaults -- do not change.
PO3_MIN_RANGE_BARS = 12       # min range maturity; a breach before this age = reset, not a sweep
PO3_MAX_RANGE_BARS = 96       # range expires after this many bars without a sweep
PO3_COMPRESSION_PCT = 25      # Donchian(20) width must be in the bottom N% of its distribution
PO3_STAT_WINDOW = 200         # distribution window for the width percentile
PO3_RANGE_TOLERANCE = 0.10    # boundaries may "breathe" by this fraction of range width
PO3_MIN_RANGE_WIDTH_PCT = 0.15  # min range width, % of price
PO3_TRIM_TAIL_PCT = 15        # impulse-tail trim, % of width
PO3_SWEEP_RETURN_BARS = 6     # close must return inside the range within N bars, else BREAKOUT
PO3_SWEEP_DEPTH_PCT = 100     # soft depth cap (True Range percentile); 100 = off
PO3_STOP_BUF_ATR = 0.4        # stop buffer beyond the sweep extreme, x ATR(14) before the range
PO3_FIB_EXT = 1.5             # fib extension target of the manipulation leg (informational)
PO3_DIST_TIMEOUT_BARS = 64    # distribution closes as TIMEOUT after N bars (only matters for FSM replay)
PO3_RANGE_WIN = 20            # Donchian window for compression & range anchoring
PO3_PIVOT_LR = 3              # pivot left/right strength
PO3_PIVOT_CAP = 60            # max stored confirmed pivots per side
PO3_COOLDOWN_BARS = 10        # debounce between a cycle end and the next range
PO3_USE_TREND_FILTER = False  # the original indicator has no EMA30 filter, so none is applied

# ---- Strategy 4: Order Block ----
OB_IMPULSE_ATR = 1.0          # impulse candle body must be >= this x ATR(14) (measured before the impulse)
OB_BOS_LOOKBACK = 10          # impulse must close beyond the high/low of the N candles before the OB candle
OB_COOLDOWN_BARS = 5          # no new OB signal if one already fired in the last N candles
OB_TARGET_MULT = 2.0          # 1:2 risk:reward
OB_USE_TREND_FILTER = True    # same EMA trend filter as strategies 1 and 2

# ---- Strategy 5: Range Filter Flip (Buy/Sell signals of the Pine Range Filter) ----
RF5_PERIOD = 100              # sampling period (Pine default)
RF5_MULT = 3.0                # range multiplier (Pine default)
RF5_SWING_LEN = 5             # swing = pivot with N candles on each side; SL = most recent swing beyond entry
REVERSE_EXIT_STRATEGIES = {"rf_flip"}   # no TP: closed by the opposite signal (or SL)

# ---- Exit rule & reporting ----
TREND_EMA_LEN = 70            # entry trend filter: above -> BUY only, below -> SELL only
EMA_EXIT_LEN = 30             # for the last 25%: exit when price closes back across this EMA
PIP_SIZE = {"XAU/USD": 0.1, "BTCUSDT": 1.0}  # price move counted as 1 pip (adjust if yours differs)
INITIAL_BALANCE = 100.0       # demo account balance (accounting only, no real orders)
LOT_SIZE = 0.01               # every entry uses this lot size
CONTRACT_SIZE = {"XAU/USD": 100, "BTCUSDT": 1}  # units per 1.00 lot (gold 100 oz, BTC 1 coin)
BREAKEVEN_R = 1.0             # at 1:1 -> tell users to move SL to entry (risk free)
TARGET_R = 2.0                # at 1:2 -> partial close
PARTIAL_FRACTION = 0.75       # share of the position closed at TARGET_R
# Strategies with a FIXED 1:2 target: the whole trade closes at 1:2 (+2R). Everything else
# (Order Block) keeps the 1:1 breakeven -> 75% at 1:2 -> EMA exit management.
FIXED_TARGET_STRATEGIES = {"rf_scalper", "consol", "po3"}

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
# Strategy 3: AMD Po3 (Accumulation -> Manipulation -> Distribution)
# ---------------------------------------------------------------------------
def compute_po3_signal(df):
    """
    Python port of the "AMD Po3 with Live Edge Stats" Pine indicator (default
    settings: pivot boundaries, HTF-bias / killzone / EQH-EQL filters OFF, so the
    re-arm logic is inert and omitted). The full state machine is replayed over
    the candle history; a signal is returned only if the distribution opened on
    the LAST closed candle.

      idle  -> accum   : Donchian(20) width in the bottom N% of its distribution
                         -> a range is anchored (impulse tail trimmed, pivot bounds)
      accum -> sweep   : price breaches a boundary (beyond tolerance) after the
                         range is mature
      sweep -> manip   : a close returns INSIDE the range within N bars (the sweep
                         was a liquidity grab, not a breakout)
      manip -> dist    : signal opens in the OPPOSITE direction
                         (low swept -> BUY, high swept -> SELL)
      dist  -> idle    : target / stop / timeout (only needed to keep the replay
                         faithful, so a new cycle cannot start while one is live)

    Entry  = close of the return bar
    SL     = beyond the full sweep excursion +/- PO3_STOP_BUF_ATR x ATR(14)
    Target = fib extension of the manipulation leg (0 = sweep tip, 1 = opposite
             boundary, target = PO3_FIB_EXT). The bot's own 1:1 / 1:2 / EMA exit
             rules manage the trade, same as the other strategies.
    """
    n = len(df)
    W = PO3_RANGE_WIN
    N = PO3_STAT_WINDOW
    L = PO3_PIVOT_LR
    warm = N + W  # full Donchian formation inside the stats window
    if n < warm + 20:
        return None, None, None, None, df.iloc[-1]["time"]

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    atr14 = np.maximum(atr(df, 14).to_numpy(), 1e-9)

    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])

    hi20 = pd.Series(high).rolling(W).max().to_numpy()
    lo20 = pd.Series(low).rolling(W).min().to_numpy()
    ch_w = hi20 - lo20

    # Percentile rank of the current width among the PREVIOUS N widths (like ta.percentrank)
    pct_rank = np.full(n, np.nan)
    for i in range(warm, n):
        window = ch_w[i - N:i]
        pct_rank[i] = 100.0 * np.sum(window <= ch_w[i]) / N

    # True Range percentile (nearest rank) for the optional soft depth cap
    tr_thr = np.full(n, np.nan)
    if PO3_SWEEP_DEPTH_PCT < 100:
        k = max(1, int(math.ceil(PO3_SWEEP_DEPTH_PCT / 100.0 * N)))
        for i in range(N - 1, n):
            tr_thr[i] = np.sort(tr[i - N + 1:i + 1])[k - 1]

    # Confirmed pivots (value appears at bar i, belongs to bar i-L)
    ph_val = np.full(n, np.nan)
    pl_val = np.full(n, np.nan)
    for i in range(2 * L, n):
        c = i - L
        if high[c] == high[c - L:c + L + 1].max():
            ph_val[i] = high[c]
        if low[c] == low[c - L:c + L + 1].min():
            pl_val[i] = low[c]

    piv_hi, piv_lo = [], []          # (value, bar)

    state = "idle"
    last_end = -100
    range_high = range_low = tol = None
    range_start = expiry_anchor = 0
    atr_anchor = 0.0
    sweep_side = 0
    sweep_bar = 0
    sweep_extreme = 0.0
    depth_cap = None
    dist_dir = 0
    dist_start = 0
    entry_px = stop_px = tgt_px = 0.0

    signal = None

    for i in range(n):
        # ----- pivot registry -----
        if not np.isnan(ph_val[i]):
            piv_hi.append((ph_val[i], i - L))
            if len(piv_hi) > PO3_PIVOT_CAP:
                piv_hi.pop(0)
        if not np.isnan(pl_val[i]):
            piv_lo.append((pl_val[i], i - L))
            if len(piv_lo) > PO3_PIVOT_CAP:
                piv_lo.pop(0)

        have_range = range_high is not None
        tol_band = tol if have_range else 0.0
        inside = have_range and range_low <= close[i] <= range_high
        breach_hi = have_range and high[i] > range_high + tol_band
        breach_lo = have_range and low[i] < range_low - tol_band

        # ================= IDLE -> ACCUMULATION =================
        if state == "idle":
            if (i >= warm and close[i] > 0 and not np.isnan(pct_rank[i])
                    and pct_rank[i] <= PO3_COMPRESSION_PCT
                    and ch_w[i] >= PO3_MIN_RANGE_WIDTH_PCT / 100.0 * close[i]
                    and i - last_end >= PO3_COOLDOWN_BARS):
                # impulse-tail trim: drop the oldest window bars while each one dominates the width
                win_last = W - 1
                t_hi, t_lo = hi20[i], lo20[i]
                if PO3_TRIM_TAIL_PCT > 0:
                    keep = True
                    while keep and win_last > PO3_MIN_RANGE_BARS:
                        hi2 = high[i - win_last + 1:i + 1].max()
                        lo2 = low[i - win_last + 1:i + 1].min()
                        if (t_hi - t_lo) - (hi2 - lo2) > PO3_TRIM_TAIL_PCT / 100.0 * (t_hi - t_lo):
                            win_last -= 1
                            t_hi, t_lo = hi2, lo2
                        else:
                            keep = False
                cand_start = i - win_last
                cand_high, cand_low = t_hi, t_lo
                # pivot boundaries (fallback to absolute when no pivots inside the window)
                hi_piv = [v for v, b in piv_hi if b >= cand_start]
                lo_piv = [v for v, b in piv_lo if b >= cand_start]
                if hi_piv:
                    cand_high = max(hi_piv)
                if lo_piv:
                    cand_low = min(lo_piv)
                cand_w = cand_high - cand_low
                if (cand_w >= PO3_MIN_RANGE_WIDTH_PCT / 100.0 * close[i]
                        and cand_low <= close[i] <= cand_high):
                    range_start = cand_start
                    expiry_anchor = cand_start
                    range_high, range_low = cand_high, cand_low
                    tol = cand_w * PO3_RANGE_TOLERANCE
                    atr_anchor = atr14[max(i - win_last - 1, 0)]
                    state = "accum"

        # ================= ACCUMULATION =================
        elif state == "accum":
            age = i - range_start
            if i - expiry_anchor > PO3_MAX_RANGE_BARS:
                state, last_end = "idle", i                      # expired
            elif breach_hi and breach_lo:
                state, last_end = "idle", i                      # news bar
            elif breach_hi or breach_lo:
                side = 1 if breach_hi else -1
                if age < PO3_MIN_RANGE_BARS:
                    state, last_end = "idle", i                  # too young = early break, not a sweep
                else:
                    cap_px = None
                    if PO3_SWEEP_DEPTH_PCT < 100:
                        cap_px = tr_thr[i - 1] if i > 0 and not np.isnan(tr_thr[i - 1]) else tr_thr[i]
                        if np.isnan(cap_px):
                            cap_px = None
                    excurs = (high[i] - range_high) if side == 1 else (range_low - low[i])
                    if cap_px is None or excurs <= cap_px:
                        sweep_side = side
                        sweep_bar = i
                        sweep_extreme = high[i] if side == 1 else low[i]
                        depth_cap = cap_px
                        state = "manip" if inside else "sweep"   # wick sweep: same-bar return
                    elif not inside:
                        state, last_end = "idle", i              # filtered breach closing outside
            else:
                # boundary expansion within tolerance (pivot mode)
                if (not np.isnan(ph_val[i]) and i - L >= range_start
                        and range_high < ph_val[i] <= range_high + tol):
                    range_high = ph_val[i]
                if (not np.isnan(pl_val[i]) and i - L >= range_start
                        and range_low - tol <= pl_val[i] < range_low):
                    range_low = pl_val[i]

        # ================= SWEEP PENDING =================
        elif state == "sweep":
            sweep_extreme = max(sweep_extreme, high[i]) if sweep_side == 1 else min(sweep_extreme, low[i])
            opp_breach = (low[i] < range_low - tol) if sweep_side == 1 else (high[i] > range_high + tol)
            cur_exc = (sweep_extreme - range_high) if sweep_side == 1 else (range_low - sweep_extreme)
            depth_bust = depth_cap is not None and cur_exc > depth_cap

            if opp_breach or depth_bust:
                state, last_end = "idle", i                      # 2-side / too deep
            elif i - sweep_bar > PO3_SWEEP_RETURN_BARS:
                state, last_end = "idle", i                      # breakout: no return in time
            elif inside:
                state = "manip"                                  # manipulation confirmed

        # ================= DISTRIBUTION (outcome tracking) =================
        elif state == "dist":
            if i - dist_start >= 1:
                hit_t = high[i] >= tgt_px if dist_dir == 1 else low[i] <= tgt_px
                hit_s = low[i] <= stop_px if dist_dir == 1 else high[i] >= stop_px
                if hit_t or hit_s or (i - dist_start >= PO3_DIST_TIMEOUT_BARS):
                    state, last_end = "idle", i

        # ================= MANIPULATION -> DISTRIBUTION OPENS =================
        if state == "manip":
            d = 1 if sweep_side == -1 else -1                    # low swept -> long, high swept -> short
            e_px = close[i]
            fib_leg = (range_high - sweep_extreme) if d == 1 else (sweep_extreme - range_low)
            if d == 1:
                t_px = range_high + (PO3_FIB_EXT - 1.0) * fib_leg
                s_px = sweep_extreme - PO3_STOP_BUF_ATR * atr_anchor
            else:
                t_px = range_low - (PO3_FIB_EXT - 1.0) * fib_leg
                s_px = sweep_extreme + PO3_STOP_BUF_ATR * atr_anchor
            risk_ok = abs(e_px - s_px) > 1e-9 and ((d == 1 and s_px < e_px) or (d == -1 and s_px > e_px))
            reward_ok = (t_px - e_px > 1e-9) if d == 1 else (e_px - t_px > 1e-9)
            if not (risk_ok and reward_ok):
                state, last_end = "idle", i                      # degenerate geometry, no trade
            else:
                dist_dir, dist_start = d, i
                entry_px, stop_px, tgt_px = e_px, s_px, t_px
                state = "dist"
                if i == n - 1:
                    signal = ("BUY" if d == 1 else "SELL", entry_px, stop_px, tgt_px)

    candle_time = df.iloc[-1]["time"]
    if signal:
        action, entry, sl, tp = signal
        return action, entry, sl, tp, candle_time
    return None, None, None, None, candle_time


# ---------------------------------------------------------------------------
# Strategy 4: Order Block
# ---------------------------------------------------------------------------
def compute_order_block_signal(df):
    """
    Bullish order block: the last BEARISH candle right before a strong BULLISH
    impulse candle. The impulse must (a) have a body >= OB_IMPULSE_ATR x ATR,
    (b) close above the OB candle's high and (c) break structure, i.e. close above
    the highest high of the OB_BOS_LOOKBACK candles before the OB candle.
    Bearish order block = the mirror image.

    The block is "created" when the impulse candle closes; that candle is the
    entry candle.
      BUY : entry = impulse close, SL = OB candle LOW,  TP = entry + 2 x risk
      SELL: entry = impulse close, SL = OB candle HIGH, TP = entry - 2 x risk
    Only the last closed candle can trigger (older ones are history), and a
    cooldown stops back-to-back impulse candles from stacking signals.
    """
    n = len(df)
    lb = OB_BOS_LOOKBACK
    if n < lb + ATR_LEN + OB_COOLDOWN_BARS + 5:
        return None, None, None, None, df.iloc[-1]["time"]

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    open_ = df["open"].to_numpy()
    atr_vals = atr(df, ATR_LEN).to_numpy()

    def ob_at(i):
        """Order block created by the impulse candle i -> (action, entry, sl) or None."""
        j = i - 1                     # candidate OB candle
        if j - lb < 0:
            return None
        body = abs(close[i] - open_[i])
        if body < OB_IMPULSE_ATR * atr_vals[j]:
            return None
        if close[j] < open_[j] and close[i] > open_[i] and close[i] > high[j] \
                and close[i] > high[j - lb:j].max():
            return "BUY", close[i], low[j]
        if close[j] > open_[j] and close[i] < open_[i] and close[i] < low[j] \
                and close[i] < low[j - lb:j].min():
            return "SELL", close[i], high[j]
        return None

    candle_time = df.iloc[-1]["time"]
    last = n - 1
    # cooldown: an OB signal in the previous N candles blocks this one
    for k in range(1, OB_COOLDOWN_BARS + 1):
        if ob_at(last - k) is not None:
            return None, None, None, None, candle_time

    res = ob_at(last)
    if res is None:
        return None, None, None, None, candle_time
    action, entry, sl = res
    risk = abs(entry - sl)
    if risk <= 0:
        return None, None, None, None, candle_time
    tp = entry + risk * OB_TARGET_MULT if action == "BUY" else entry - risk * OB_TARGET_MULT
    return action, entry, sl, tp, candle_time


# ---------------------------------------------------------------------------
# Strategy 5: Range Filter Flip (Pine "Range Filter" Buy / Sell signals)
# ---------------------------------------------------------------------------
def rf_flip_signals(df):
    """
    Literal port of the Pine Range Filter: returns (long_sig, short_sig) boolean
    arrays. A signal fires when the filter direction FLIPS (longCondition =
    longCond and the previous state was short, and vice versa).
    """
    src = df["close"].to_numpy()
    n = len(src)
    diff = np.abs(np.diff(src, prepend=src[0]))
    avrng = pd.Series(diff).ewm(span=RF5_PERIOD, adjust=False).mean().to_numpy()
    smrng = pd.Series(avrng).ewm(span=RF5_PERIOD * 2 - 1, adjust=False).mean().to_numpy() * RF5_MULT

    filt = np.zeros(n)
    prev = 0.0                                   # Pine: nz(rngfilt[1]) = 0 on the first bar
    for i in range(n):
        x, r = src[i], smrng[i]
        if x > prev:
            cur = prev if (x - r) < prev else (x - r)
        else:
            cur = prev if (x + r) > prev else (x + r)
        filt[i] = cur
        prev = cur

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

    long_sig = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)
    cond_ini = 0
    for i in range(1, n):
        moved = src[i] != src[i - 1]
        long_cond = src[i] > filt[i] and upward[i] > 0 and moved
        short_cond = src[i] < filt[i] and downward[i] > 0 and moved
        long_sig[i] = long_cond and cond_ini == -1
        short_sig[i] = short_cond and cond_ini == 1
        if long_cond:
            cond_ini = 1
        elif short_cond:
            cond_ini = -1
    return long_sig, short_sig


def _previous_swing(df, last, side):
    """Most recent CONFIRMED swing low (side=BUY) / swing high (side=SELL) that lies
    beyond the entry price. Returns None if there is none."""
    L = RF5_SWING_LEN
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    entry = df["close"].iloc[last]
    for c in range(last - L, L - 1, -1):
        if side == "BUY":
            if low[c] == low[c - L:c + L + 1].min() and low[c] < entry:
                return float(low[c])
        else:
            if high[c] == high[c - L:c + L + 1].max() and high[c] > entry:
                return float(high[c])
    return None


def compute_rf_flip_signal(df):
    """BUY/SELL on a Range Filter flip. SL = previous swing low (BUY) / high (SELL).
    No take-profit -- the trade is closed by the opposite signal."""
    n = len(df)
    candle_time = df.iloc[-1]["time"]
    if n < RF5_PERIOD * 2 + RF5_SWING_LEN * 2 + 10:
        return None, None, None, None, candle_time
    long_sig, short_sig = rf_flip_signals(df)
    last = n - 1
    action = "BUY" if long_sig[last] else "SELL" if short_sig[last] else None
    if not action:
        return None, None, None, None, candle_time
    sl = _previous_swing(df, last, action)
    entry = float(df["close"].iloc[last])
    if sl is None or abs(entry - sl) <= 0:
        return None, None, None, None, candle_time
    return action, entry, sl, None, candle_time


def _manage_reverse_trade(trade, df, last_time, long_sig, short_sig):
    """Walks the new candles for a trade that has no target:
       stop hit              -> LOSS at the stop (-1R)
       opposite signal       -> closed at that candle's close (any R)
    Returns (reason, outcome, r_total, exit_price, exit_time) or None."""
    action, sl, entry = trade["action"], trade["sl"], trade["entry"]
    risk = abs(entry - sl)
    sign = 1 if action == "BUY" else -1
    times = df["time"].tolist()
    for i in range(1, len(df)):
        if times[i] <= last_time:
            continue
        row = df.iloc[i]
        stop_hit = row["low"] <= sl if action == "BUY" else row["high"] >= sl
        if stop_hit:
            return ("sl", "loss", -1.0, sl, times[i])
        opposite = short_sig[i] if action == "BUY" else long_sig[i]
        if opposite:
            exit_price = float(row["close"])
            r = sign * (exit_price - entry) / risk
            outcome = "win" if r > 0 else "loss" if r < 0 else "breakeven"
            return ("reverse", outcome, r, exit_price, times[i])
    return None


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def send_telegram(symbol, action, entry, sl, tp, candle_time, subscribers, strategy_label, show_tp=False, note=""):
    """Sends the signal and returns {chat_id: message_id} so later messages
    (risk free / partial close / close) can reply to this exact message."""
    text = (
        f"🔔 {action} Signal ({strategy_label})\n"
        f"Symbol: {symbol}\n"
        f"Entry: {entry:.2f}\n"
        f"SL: {sl:.2f}\n"
        + (f"TP (1:{TARGET_R:g}): {tp:.2f}\n" if show_tp else "")
        + f"Time: {candle_time}"
        + (f"\n{note}" if note else "")
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


def send_outcome_telegram(symbol, strategy_label, action, entry, exit_price, sl, outcome, reason,
                          r, pips, pnl, pct, balance, subscribers, message_ids=None):
    """Final message for a trade. reason: 'sl' | 'breakeven' | 'be_remainder' | 'ema' | 'target' | 'reverse'."""
    keep = int(round((1 - PARTIAL_FRACTION) * 100))
    tail = (
        f"Result: {r:+.2f}R | {pips:+.1f} pips\n"
        f"P&L: {_money(pnl)} ({pct:+.2f}%) on {LOT_SIZE} lot\n"
        f"Demo balance: ${balance:.2f}"
    )
    if reason == "ema":
        tag = {"win": "✅ WIN", "loss": "❌ LOSS"}.get(outcome, "➖ BREAKEVEN")
        text = (
            f"🔔 EXIT NOW - EMA {EMA_EXIT_LEN} crossed, close the last {keep}% ({strategy_label})\n"
            f"Symbol: {symbol} | {action}\n"
            f"Entry: {entry:.2f} -> Exit: {exit_price:.2f}\n"
            f"{tail}\n{tag}"
        )
    elif reason == "reverse":
        tag = {"win": "✅ WIN", "loss": "❌ LOSS"}.get(outcome, "➖ BREAKEVEN")
        text = (
            f"🔄 OPPOSITE SIGNAL - close this trade now ({strategy_label})\n"
            f"Symbol: {symbol} | {action}\n"
            f"Entry: {entry:.2f} -> Exit: {exit_price:.2f}\n"
            f"{tail}\n{tag}"
        )
    elif reason == "target":
        text = (
            f"🎯 TARGET HIT 1:{TARGET_R:g} - close the whole trade ({strategy_label})\n"
            f"Symbol: {symbol} | {action}\n"
            f"Entry: {entry:.2f} -> Exit: {exit_price:.2f}\n"
            f"{tail}\n✅ WIN"
        )
    elif reason == "be_remainder":
        text = (
            f"✅ WIN - the last {keep}% was stopped at entry ({strategy_label})\n"
            f"Symbol: {symbol} | {action}\n"
            f"{int(PARTIAL_FRACTION * 100)}% was already banked at 1:2.\n"
            f"{tail}"
        )
    elif reason == "breakeven":
        text = (
            f"➖ BREAKEVEN - price came back to entry after 1:1 ({strategy_label})\n"
            f"Symbol: {symbol} | {action}\n"
            f"Entry: {entry:.2f}\n"
            f"{tail}"
        )
    else:
        text = (
            f"❌ STOP LOSS HIT ({strategy_label})\n"
            f"Symbol: {symbol} | {action}\n"
            f"Entry: {entry:.2f}  SL: {sl:.2f}\n"
            f"{tail}"
        )
    _send_reply(text, subscribers, message_ids)


def send_welcome(chat_id):
    keep = int(round((1 - PARTIAL_FRACTION) * 100))
    text = (
        "✅ You're subscribed!\n"
        "You'll receive BUY/SELL signals (Entry + SL) for BTCUSDT and XAU/USD. "
        "Most strategies close fully at the 1:2 target. Range Filter Flip trades close on the opposite signal. Order Block signals: "
        "I'll reply to each one: at 1:1 -> move SL to entry (risk free), "
        f"at 1:2 -> close {int(PARTIAL_FRACTION * 100)}%, then when EMA {EMA_EXIT_LEN} "
        f"crosses -> close the last {keep}%. Results are tracked on demo accounts - every strategy has its own ${INITIAL_BALANCE:.0f} "
        f"balance ({LOT_SIZE} lot per trade). You'll also get daily, weekly, monthly & yearly reports."
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
    Trade management (fixed-target strategies skip stage 2: the whole trade closes at 1:2),
    walking forward through candles that are new since the
    last check (progress is stored on each trade so messages are never repeated):

      stage 0: SL at the original stop.
               stop hit          -> LOSS (-1R)
               price reaches 1:1 -> message "risk free", stage 1
      stage 1: stop is now the entry price.
               price back to entry -> BREAKEVEN (0R)
               price reaches 1:2   -> message "close 75%", stage 2
      stage 2: 75% banked at 1:2 (= +1.5R). The last 25% keeps running.
               price back to entry -> last 25% stopped at 0 (total +1.5R, WIN)
               EMA fast/slow crosses against the trade at a candle close
                                   -> message "close the rest" (total = 1.5R + 25% of its R)

    EMA crosses are ignored until 1:2 has been reached. Demo accounting: every
    trade uses LOT_SIZE lots on a balance that starts at INITIAL_BALANCE.
    """
    pending = state.setdefault("pending_trades", [])
    if not pending:
        return

    ema_exit = ema(df["close"], EMA_EXIT_LEN).to_numpy()
    closes = df["close"].to_numpy()
    times = df["time"].tolist()
    subscribers = state.get("subscribers", [])

    still_pending = []
    rf_sig = None   # Range Filter flip signals, computed lazily for reverse-exit trades
    for trade in pending:
        if trade["symbol"] != name:
            still_pending.append(trade)
            continue

        action, sl, entry = trade["action"], trade["sl"], trade["entry"]
        risk = abs(entry - sl)
        if risk <= 0:
            continue  # invalid record, drop it
        sign = 1 if action == "BUY" else -1
        r1_level = entry + sign * risk * BREAKEVEN_R
        r2_level = entry + sign * risk * TARGET_R
        message_ids = trade.get("message_ids", {})
        fixed = trade.get("fixed_target", False)   # True -> close everything at 1:2

        stage = trade.get("stage", 0)
        last_time = pd.to_datetime(trade.get("last_time") or trade["entry_time"], utc=True)
        result = None  # (reason, outcome, r_total, exit_price, exit_time)

        reverse = trade.get("mgmt") == "reverse"   # no target: closed by the opposite signal
        if reverse:
            if rf_sig is None:
                rf_sig = rf_flip_signals(df)
            result = _manage_reverse_trade(trade, df, last_time, rf_sig[0], rf_sig[1])

        for i in (() if reverse else range(1, len(df))):
            if times[i] <= last_time:
                continue
            row = df.iloc[i]
            hi, lo, op = row["high"], row["low"], row["open"]

            def touched(level):
                return hi >= level if action == "BUY" else lo <= level

            stop_level = sl if stage == 0 else entry
            stop_hit = lo <= stop_level if action == "BUY" else hi >= stop_level

            events = []
            if stop_hit:
                events.append(("stop", stop_level))
            if stage < 1 and touched(r1_level):
                events.append(("r1", r1_level))
            if stage < 2 and touched(r2_level):
                events.append(("r2", r2_level))
            # order by distance from the candle's open (best guess at what came first)
            events.sort(key=lambda e: abs(e[1] - op))

            has_r2 = any(k == "r2" for k, _ in events)
            for kind, _level in events:
                if fixed and kind == "r2":
                    result = ("target", "win", TARGET_R, r2_level, times[i])
                    break
                if fixed and kind == "r1" and has_r2:
                    continue   # target is hit on this very candle - skip the breakeven message
                if kind in ("r1", "r2") and stage < 1:
                    stage = 1
                    send_stage_telegram("r1", name, trade["strategy_label"], action, entry, risk,
                                        state.setdefault("balances", {}).get(trade["strategy_label"], INITIAL_BALANCE), subscribers, message_ids)
                if kind == "r2" and stage < 2:
                    stage = 2
                    send_stage_telegram("r2", name, trade["strategy_label"], action, entry, risk,
                                        state.setdefault("balances", {}).get(trade["strategy_label"], INITIAL_BALANCE), subscribers, message_ids)
                if kind == "stop":
                    if stage == 0:
                        result = ("sl", "loss", -1.0, sl, times[i])
                    elif stage == 1:
                        result = ("breakeven", "breakeven", 0.0, entry, times[i])
                    else:
                        result = ("be_remainder", "win", PARTIAL_FRACTION * TARGET_R, entry, times[i])
                    break
            if result:
                break

            if stage == 2:
                if action == "BUY":
                    crossed = closes[i - 1] >= ema_exit[i - 1] and closes[i] < ema_exit[i]
                else:
                    crossed = closes[i - 1] <= ema_exit[i - 1] and closes[i] > ema_exit[i]
                if crossed:
                    exit_price = float(row["close"])
                    r_rest = sign * (exit_price - entry) / risk
                    r_total = PARTIAL_FRACTION * TARGET_R + (1 - PARTIAL_FRACTION) * r_rest
                    outcome = "win" if r_total > 0 else "loss" if r_total < 0 else "breakeven"
                    result = ("ema", outcome, r_total, exit_price, times[i])
                    break

        if result is None:
            trade["stage"] = stage
            trade["last_time"] = str(times[-1])
            still_pending.append(trade)
            continue

        reason, outcome, r_total, exit_price, exit_time = result
        pnl = r_total * risk * CONTRACT_SIZE.get(name, 1.0) * LOT_SIZE
        # every strategy has its own demo account (starts at INITIAL_BALANCE)
        balances = state.setdefault("balances", {})
        balance_before = balances.get(trade["strategy_label"], INITIAL_BALANCE)
        balance_after = balance_before + pnl
        balances[trade["strategy_label"]] = balance_after
        pct = (pnl / balance_before * 100) if balance_before else 0.0
        pips = r_total * risk / _pip_size(name)

        send_outcome_telegram(
            name, trade["strategy_label"], action, entry, exit_price, sl, outcome, reason,
            r_total, pips, pnl, pct, balance_after, subscribers, message_ids
        )
        state.setdefault("trade_history", []).append({
            "symbol": name,
            "strategy_label": trade["strategy_label"],
            "action": action,
            "outcome": outcome,
            "reason": reason,
            "entry": entry,
            "exit": exit_price,
            "r": round(r_total, 4),
            "pips": round(pips, 2),
            "pnl_usd": round(pnl, 4),
            "balance_after": round(balance_after, 4),
            "entry_time": trade["entry_time"],
            "resolved_time": str(exit_time),
        })
        print(f"[{name}][{trade['strategy_label']}] Trade opened {trade['entry_time']} "
              f"closed ({reason}): {outcome.upper()} {r_total:+.2f}R, {_money(pnl)}, balance ${balance_after:.2f}")

    state["pending_trades"] = still_pending


# ---------------------------------------------------------------------------
# Periodic performance reports (daily / weekly / monthly / yearly)
# ---------------------------------------------------------------------------
def build_report_text(title, trades, start_balances=None):
    """start_balances: {strategy_label: balance at the start of the period}.
    Every strategy trades its own $INITIAL_BALANCE demo account."""
    if not start_balances:
        start_balances = {label: INITIAL_BALANCE for _, label, _, _ in STRATEGIES}
    total_start = sum(start_balances.values())

    sep = "━━━━━━━━━━━━━━━━"
    bal_lines = []
    total_end = 0.0
    for label, st in start_balances.items():
        pnl = sum(t["pnl_usd"] for t in trades
                  if t["strategy_label"] == label and t.get("pnl_usd") is not None)
        end = st + pnl
        total_end += end
        pct = (pnl / st * 100) if st else 0.0
        bal_lines.append(f"• {label}: ${st:.2f} -> ${end:.2f} ({pct:+.2f}%)")
    total_pct = ((total_end - total_start) / total_start * 100) if total_start else 0.0

    if not trades:
        return "\n".join([
            f"📊 {title}", "No trades in this period.", "",
            "Demo balances (separate account per strategy):", *bal_lines,
        ])

    symbols = sorted({t["symbol"] for t in trades})
    strategies = sorted({t["strategy_label"] for t in trades})
    count_line = " | ".join(
        f"{sym}: {sum(1 for t in trades if t['symbol'] == sym)}" for sym in symbols
    )

    parts = [
        f"📊 {title}",
        sep,
        "Demo balances (separate account per strategy):",
        *bal_lines,
        f"Combined: ${total_start:.2f} -> ${total_end:.2f} ({total_pct:+.2f}%)",
        f"Trades by symbol: {count_line} | Total: {len(trades)}",
        "",
        _format_block("📌 Overall (all strategies)", trades, total_start),
    ]
    for sym in symbols:
        icon = "🥇" if "XAU" in sym else "🪙"
        parts += ["", sep, _format_block(f"{icon} {sym} (all strategies)", [t for t in trades if t["symbol"] == sym], total_start)]
    for label in strategies:
        parts += ["", sep, _format_block(f"🎯 {label}", [t for t in trades if t["strategy_label"] == label],
                                         start_balances.get(label, INITIAL_BALANCE))]
    parts += [
        "",
        f"ℹ️ Demo: each strategy has its own ${INITIAL_BALANCE:.0f} account, {LOT_SIZE} lot per trade. "
        "Strategy blocks: % of that strategy's balance at the start of this period. "
        "Overall / symbol blocks: % of the combined balance.",
    ]
    return "\n".join(parts)


def _pip_size(symbol):
    return PIP_SIZE.get(symbol, 1.0)


def _max_drawdown(group, start_balance):
    """Simulates an equity curve from this group's trades alone (in the
    order they closed) and returns the worst peak-to-trough drop, in $ and %."""
    ordered = sorted(
        (t for t in group if t.get("pnl_usd") is not None),
        key=lambda t: t["resolved_time"],
    )
    equity = peak = start_balance
    max_dd_usd = max_dd_pct = 0.0
    for t in ordered:
        equity += t["pnl_usd"]
        peak = max(peak, equity)
        dd = peak - equity
        if dd > max_dd_usd:
            max_dd_usd = dd
            max_dd_pct = (dd / peak * 100) if peak else 0.0
    return max_dd_usd, max_dd_pct


def _stats(group, start_balance=None):
    if start_balance is None:
        start_balance = INITIAL_BALANCE
    total = len(group)
    wins = sum(1 for t in group if t["outcome"] == "win")
    breakevens = sum(1 for t in group if t["outcome"] == "breakeven")
    losses = sum(1 for t in group if t["outcome"] == "loss")

    rated = [t for t in group if t.get("r") is not None]
    rs = [t["r"] for t in rated]
    pips = [(t.get("pips") or 0.0) for t in rated]
    pnls = [t["pnl_usd"] for t in group if t.get("pnl_usd") is not None]

    pos_r = [x for x in rs if x > 0]
    neg_r = [x for x in rs if x < 0]
    pos_p = [x for x in pnls if x > 0]
    neg_p = [x for x in pnls if x < 0]

    avg_win_r = (sum(pos_r) / len(pos_r)) if pos_r else None
    avg_loss_r = (sum(neg_r) / len(neg_r)) if neg_r else None
    avg_win_usd = (sum(pos_p) / len(pos_p)) if pos_p else None
    avg_loss_usd = (sum(neg_p) / len(neg_p)) if neg_p else None

    if pnls:
        gross_profit, gross_loss = sum(pos_p), -sum(neg_p)
    else:
        gross_profit, gross_loss = sum(pos_r), -sum(neg_r)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None
    avg_rr = (avg_win_r / abs(avg_loss_r)) if (avg_win_r is not None and avg_loss_r is not None) else None
    max_dd_usd, max_dd_pct = _max_drawdown(group, start_balance)

    return {
        "total": total, "wins": wins, "breakevens": breakevens, "losses": losses,
        "win_rate": (wins / total * 100) if total else 0.0,
        "avg_win_r": avg_win_r, "avg_loss_r": avg_loss_r,
        "avg_win_usd": avg_win_usd, "avg_loss_usd": avg_loss_usd,
        "avg_rr": avg_rr, "profit_factor": profit_factor,
        "net_r": sum(rs), "net_usd": sum(pnls), "net_pips": sum(pips),
        "max_dd_usd": max_dd_usd, "max_dd_pct": max_dd_pct,
    }


def _format_block(title, group, start_balance):
    s = _stats(group, start_balance)
    pf = s["profit_factor"]
    pf_text = "N/A" if pf is None else ("∞" if pf == float("inf") else f"{pf:.2f}")
    rr_text = f"1 : {s['avg_rr']:.2f}" if s["avg_rr"] is not None else "N/A"
    net_pct = (s["net_usd"] / start_balance * 100) if start_balance else 0.0
    return "\n".join([
        title,
        f"Trades: {s['total']}  (✅ Win {s['wins']} | ➖ BE {s['breakevens']} | ❌ Loss {s['losses']})",
        f"Win Rate: {s['win_rate']:.1f}%",
        f"Avg R:R: {rr_text}",
        f"Profit Factor: {pf_text}",
        f"Avg Win: {_pair_text(s['avg_win_r'], s['avg_win_usd'])}",
        f"Avg Loss: {_pair_text(s['avg_loss_r'], s['avg_loss_usd'])}",
        f"Max Drawdown: {_money(-s['max_dd_usd'])} ({-s['max_dd_pct']:.2f}%)",
        f"Net: {s['net_r']:+.2f}R | {_money(s['net_usd'])} ({net_pct:+.2f}%) | {s['net_pips']:+.1f} pips",
    ])


def _money(v):
    return f"{'+' if v >= 0 else '-'}${abs(v):.2f}"


def _send_reply(text, subscribers, message_ids=None):
    """Sends text to every subscriber, as a reply to the original signal
    message when we have its message_id for that chat."""
    message_ids = message_ids or {}
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
            print(f"  -> Failed to send to {chat_id}: {e}")


def send_stage_telegram(kind, symbol, strategy_label, action, entry, risk, balance,
                        subscribers, message_ids=None):
    """kind='r1' -> 1:1 reached. kind='r2' -> 1:2 reached."""
    if kind == "r1":
        text = (
            f"🛡 BREAKEVEN NOW ({strategy_label})\n"
            f"Symbol: {symbol} | {action}\n"
            f"Move SL to entry: {entry:.2f}"
        )
    else:
        text = (
            f"🎯 PARTIAL CLOSE NOW ({strategy_label})\n"
            f"Symbol: {symbol} | {action}\n"
            f"Close {int(PARTIAL_FRACTION * 100)}%, keep SL at entry: {entry:.2f}"
        )
    _send_reply(text, subscribers, message_ids)


def _pair_text(r, usd):
    if r is None:
        return "N/A"
    if usd is None:
        return f"{r:+.2f}R"
    return f"{r:+.2f}R ({_money(usd)})"


def _balance_before(history, moment, label=None):
    """Balance of one strategy's account (label given) at `moment`."""
    total = INITIAL_BALANCE
    for t in history:
        if label is not None and t.get("strategy_label") != label:
            continue
        if t.get("pnl_usd") is not None and pd.to_datetime(t["resolved_time"], utc=True) < moment:
            total += t["pnl_usd"]
    return total


def _send_period_report(title, history, start, end, subscribers):
    trades = _trades_in_range(history, start, end)
    starts = {label: _balance_before(history, start, label) for _, label, _, _ in STRATEGIES}
    send_report_telegram(title, trades, subscribers, starts)


def send_report_telegram(title, trades, subscribers, start_balances=None):
    text = build_report_text(title, trades, start_balances)
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
            _send_period_report(f"Daily Report -- {yesterday_start.date()}", history,
                                yesterday_start, yesterday_end, subscribers)
        state["last_daily_report"] = str(today)

    # Weekly: Friday night (~10 PM BD time), covering Mon-Fri of this week
    iso_year, iso_week, _ = now.isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"
    if now.weekday() == 4 and now.hour >= REPORT_HOUR_UTC and state.get("last_weekly_report") != week_key:
        week_start = datetime.combine(now.date() - timedelta(days=now.weekday()), datetime.min.time(), tzinfo=timezone.utc)
        _send_period_report(f"Weekly Report -- {week_start.date()} to {now.date()}", history,
                            week_start, now, subscribers)
        state["last_weekly_report"] = week_key

    # Monthly: last calendar day of the month (~10 PM BD time)
    is_last_day_of_month = (now.date() + timedelta(days=1)).month != now.month
    month_key = f"{now.year}-{now.month:02d}"
    if is_last_day_of_month and now.hour >= REPORT_HOUR_UTC and state.get("last_monthly_report") != month_key:
        month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        _send_period_report(f"Monthly Report -- {month_start.strftime('%B %Y')}", history,
                            month_start, now, subscribers)
        state["last_monthly_report"] = month_key

    # Yearly: on the first run of a new year, for the year just finished
    year_key = str(now.year)
    if state.get("last_yearly_report") != year_key:
        if state.get("last_yearly_report") is not None:
            this_year_start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
            last_year_start = datetime(now.year - 1, 1, 1, tzinfo=timezone.utc)
            _send_period_report(f"Yearly Report -- {now.year - 1}", history,
                                last_year_start, this_year_start, subscribers)
        state["last_yearly_report"] = year_key


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def _ensure_balances(state):
    """One-time migration: split the old single balance into one account per strategy,
    rebuilt from the trade history (each strategy starts at INITIAL_BALANCE)."""
    if "balances" in state:
        return
    bal = {label: INITIAL_BALANCE for _, label, _, _ in STRATEGIES}
    for t in state.get("trade_history", []):
        if t.get("pnl_usd") is not None:
            lab = t.get("strategy_label")
            bal[lab] = bal.get(lab, INITIAL_BALANCE) + t["pnl_usd"]
    state["balances"] = bal
    state.pop("balance", None)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# (state key suffix, label, signal function, apply the EMA trend filter?)
STRATEGIES = [
    ("rf_scalper", "Range Filter + Scalper Pro", compute_black_shadow_signal, True),
    ("consol", "Consolidation Breakout", compute_consolidation_break_signal, True),
    ("po3", "AMD Po3", compute_po3_signal, PO3_USE_TREND_FILTER),
    ("order_block", "Order Block", compute_order_block_signal, OB_USE_TREND_FILTER),
    ("rf_flip", "Range Filter Flip", compute_rf_flip_signal, False),
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
    try:
        check_pending_trades(name, df, state)
    except Exception as e:
        print(f"[{name}] ERROR while tracking open trades: {e}")

    # Trend filter: above EMA(70) -> BUY entries only, below it -> SELL entries only
    trend_ema = ema(df["close"], TREND_EMA_LEN)
    current_close = df["close"].iloc[-1]
    current_trend_ema = trend_ema.iloc[-1]

    for key_suffix, label, strategy_fn, use_trend_filter in STRATEGIES:
        try:
            action, entry, sl, tp, candle_time = strategy_fn(df)

            if use_trend_filter:
                if action == "BUY" and current_close < current_trend_ema:
                    print(f"[{name}][{label}] BUY signal rejected -- price below EMA({TREND_EMA_LEN}) trend filter")
                    action = None
                elif action == "SELL" and current_close > current_trend_ema:
                    print(f"[{name}][{label}] SELL signal rejected -- price above EMA({TREND_EMA_LEN}) trend filter")
                    action = None

            if is_market_stale(candle_time):
                print(f"[{name}][{label}] Market appears closed (last candle: {candle_time}) -- skipping")
                continue

            state_key = f"{name}_{key_suffix}"
            candle_key = str(candle_time)

            fixed_target = key_suffix in FIXED_TARGET_STRATEGIES
            if action and fixed_target:
                # fixed 1:2 target for this strategy (overrides any strategy-specific target)
                risk_px = abs(entry - sl)
                tp = entry + risk_px * TARGET_R if action == "BUY" else entry - risk_px * TARGET_R

            reverse_exit = key_suffix in REVERSE_EXIT_STRATEGIES
            if action and state.get(state_key) != candle_key:
                message_ids = send_telegram(name, action, entry, sl, tp, candle_time, state.get("subscribers", []), label,
                                            show_tp=fixed_target,
                                            note="Exit: when the opposite signal appears (or SL)" if reverse_exit else "")
                state[state_key] = candle_key
                state.setdefault("pending_trades", []).append({
                    "symbol": name,
                    "strategy_label": label,
                    "action": action,
                    "entry": entry,
                    "sl": sl,
                    "entry_time": str(candle_time),
                    "message_ids": message_ids,
                    "fixed_target": fixed_target,
                    "mgmt": "reverse" if reverse_exit else None,
                })
                print(f"[{name}][{label}] Sent {action} signal at {candle_time} "
                      f"to {len(state.get('subscribers', []))} subscriber(s)")
            else:
                print(f"[{name}][{label}] No new signal (last candle: {candle_time})")
        except Exception as e:
            print(f"[{name}][{label}] ERROR: {e}")


def main():
    state = load_state()
    _ensure_balances(state)

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
