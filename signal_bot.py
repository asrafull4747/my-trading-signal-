"""
Order Block Signal Bot
Based on the user's supplied LuxAlgo "Order Block Detector" Pine Script.
Strategy:
  - Volume pivot length = 5
  - Bullish/Bearish OB detected from the confirmed volume pivot candle
  - Entry = 50% (midpoint) of the OB
  - SL = OB boundary (bullish low / bearish high), optional buffer
  - TP = 1:3 fixed risk/reward
  - Old Range Filter / Scalper and Consolidation Breakout strategies removed.

Note: XAU/USD free FX feeds may not provide exchange volume. When volume is
missing, OB_USE_VOLUME_PROXY=True uses candle range as a proxy.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ===== Shared config =====
TIMEFRAME_MIN = 3
FETCH_LIMIT = 500

# ---- Strategy: LuxAlgo-style Volume Pivot Order Block ----
OB_VOLUME_PIVOT_LENGTH = 5
OB_TARGET_R = 3.0
OB_ENTRY_PERCENT = 0.50       # 50% of the OB
OB_SL_BUFFER = 0.0            # 0 = exactly OB boundary
OB_USE_VOLUME_PROXY = True    # XAU/USD often has no exchange volume from free FX feeds
OB_MAX_ACTIVE = 20

# ---- Trade accounting ----
PIP_SIZE = {"XAU/USD": 0.1, "BTCUSDT": 1.0}
INITIAL_BALANCE = 100.0
LOT_SIZE = 0.01
CONTRACT_SIZE = {"XAU/USD": 100, "BTCUSDT": 1}

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


# ---------------------------------------------------------------------------

def fetch_btc_klines(limit=FETCH_LIMIT):
    """
    Kraken 1-minute OHLCV -> resampled 3-minute OHLCV.
    The Order Block detector uses real BTC volume, matching Pine's volume pivot.
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
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["time", "open", "high", "low", "close", "volume"]]

    df = resample_ohlcv(df, TIMEFRAME_MIN)
    df = df.tail(limit).reset_index(drop=True)
    return drop_unclosed_candle(df, time_is_close_time=False)


def fetch_gold_klines(limit=FETCH_LIMIT):
    """
    Twelve Data 1-minute -> resampled 3-minute OHLCV.
    If the provider does not return volume for XAU/USD, a candle-range proxy
    is used when OB_USE_VOLUME_PROXY=True.
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

    df = pd.DataFrame(data["values"]).rename(columns={"datetime": "time"})
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = np.nan

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df = df[["time", "open", "high", "low", "close", "volume"]]

    if df["volume"].isna().all():
        if not OB_USE_VOLUME_PROXY:
            raise RuntimeError(
                "Twelve Data returned no XAU/USD volume. "
                "Set OB_USE_VOLUME_PROXY=True or use a feed with volume."
            )
        # Proxy only for XAU/USD feeds without volume.
        df["volume"] = (df["high"] - df["low"]).abs()

    df = resample_ohlcv(df, TIMEFRAME_MIN)
    df = df.tail(limit).reset_index(drop=True)
    return drop_unclosed_candle(df, time_is_close_time=False)


def resample_ohlcv(df_1m, minutes):
    """Resample OHLCV into N-minute candles."""
    df_1m = df_1m.set_index("time")
    agg = df_1m.resample(f"{minutes}min", label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"]).reset_index()
    return agg

# ---------------------------------------------------------------------------
# Strategy: LuxAlgo-style Volume Pivot Order Block
# ---------------------------------------------------------------------------
def _volume_pivot_high(volume, length):
    """
    Pine equivalent of ta.pivothigh(volume, length, length).
    A pivot is confirmed only after `length` bars have closed to its right.
    """
    n = len(volume)
    phv = np.full(n, np.nan)
    for i in range(2 * length, n):
        c = i - length
        window = volume[c - length:c + length + 1]
        if np.isfinite(volume[c]) and volume[c] == np.nanmax(window):
            phv[i] = volume[c]
    return phv


def _calculate_ob_state(df, length=OB_VOLUME_PIVOT_LENGTH):
    """
    Replays the relevant LuxAlgo logic bar-by-bar.

    Pine:
      upper = ta.highest(length)
      lower = ta.lowest(length)
      os := high[length] > upper ? 0 : low[length] < lower ? 1 : os[1]

    Bullish OB:
      phv and os == 1
      top = hl2[length], bottom = low[length]

    Bearish OB:
      phv and os == 0
      top = high[length], bottom = hl2[length]
    """
    n = len(df)
    if n < max(2 * length + 5, 30):
        return []

    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    volume = df["volume"].to_numpy(float)

    # If volume is unavailable, make the fallback explicit.
    if not np.isfinite(volume).any():
        if OB_USE_VOLUME_PROXY:
            volume = np.maximum(high - low, 1e-12)
        else:
            return []

    # Pine ta.highest/lowest(length): current bar + previous length-1 bars.
    upper = pd.Series(high).rolling(length, min_periods=length).max().to_numpy()
    lower = pd.Series(low).rolling(length, min_periods=length).min().to_numpy()
    phv = _volume_pivot_high(volume, length)

    # os[1] starts as na in Pine; we model it as 0 until the first state change.
    os = np.zeros(n, dtype=int)
    for i in range(n):
        if i < length or not np.isfinite(upper[i]) or not np.isfinite(lower[i]):
            os[i] = os[i - 1] if i > 0 else 0
            continue
        if high[i - length] > upper[i]:
            os[i] = 0
        elif low[i - length] < lower[i]:
            os[i] = 1
        else:
            os[i] = os[i - 1] if i > 0 else 0

    obs = []

    # Replay confirmed pivots and apply the same mitigation test.
    for i in range(n):
        if not np.isfinite(phv[i]):
            continue

        c = i - length
        if c < 0:
            continue

        if os[i] == 1:
            top = (high[c] + low[c]) / 2.0
            btm = low[c]
            side = "bull"
        else:
            top = high[c]
            btm = (high[c] + low[c]) / 2.0
            side = "bear"

        # The Pine script stores time[length] as the OB's left edge.
        obs.insert(0, {
            "side": side,
            "top": float(top),
            "bottom": float(btm),
            "avg": float((top + btm) / 2.0),
            "left_time": str(df.iloc[c]["time"]),
            "confirm_time": str(df.iloc[i]["time"]),
            "pivot_index": c,
            "confirm_index": i,
        })

        # Keep a manageable list, equivalent to displaying the newest OBs.
        if len(obs) > OB_MAX_ACTIVE:
            obs.pop()

    # Apply mitigation using the same current target concept:
    # bullish removed when target_bull < ob_btm
    # bearish removed when target_bear > ob_top
    active = []
    for ob in obs:
        start = ob["confirm_index"] + 1
        if start >= n:
            active.append(ob)
            continue

        post = df.iloc[start:]
        if ob["side"] == "bull":
            # Wick mitigation: lowest low after confirmation.
            target = float(post["low"].min())
            mitigated = target < ob["bottom"]
        else:
            target = float(post["high"].max())
            mitigated = target > ob["top"]

        if not mitigated:
            active.append(ob)

    return active


def compute_order_block_signal(df):
    """
    Returns a signal only when the latest closed candle reaches the 50% level
    of the newest active bullish/bearish Order Block.

    BUY:
      entry = OB midpoint
      SL    = OB bottom - optional buffer
      TP    = entry + 3R

    SELL:
      entry = OB midpoint
      SL    = OB top + optional buffer
      TP    = entry - 3R

    A touch of the midpoint is enough; the signal is emitted once for the
    candle that first touches the level.
    """
    candle_time = df.iloc[-1]["time"]
    obs = _calculate_ob_state(df, OB_VOLUME_PIVOT_LENGTH)
    if not obs:
        return None, None, None, None, candle_time

    row = df.iloc[-1]
    # Newest active OB first because _calculate_ob_state inserts at front.
    for ob in obs:
        entry = ob["avg"]

        if ob["side"] == "bull":
            sl = ob["bottom"] - OB_SL_BUFFER
            touched = row["low"] <= entry <= row["high"]
            risk = entry - sl
            if touched and risk > 0:
                tp = entry + risk * OB_TARGET_R
                return "BUY", float(entry), float(sl), float(tp), candle_time

        else:
            sl = ob["top"] + OB_SL_BUFFER
            touched = row["low"] <= entry <= row["high"]
            risk = sl - entry
            if touched and risk > 0:
                tp = entry - risk * OB_TARGET_R
                return "SELL", float(entry), float(sl), float(tp), candle_time

    return None, None, None, None, candle_time

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def send_telegram(symbol, action, entry, sl, tp, candle_time, subscribers, strategy_label):
    """Send a new 50% Order Block entry signal."""
    text = (
        f"🔔 {action} Signal ({strategy_label})\n"
        f"Symbol: {symbol}\n"
        f"Entry (50% OB): {entry:.2f}\n"
        f"SL (OB boundary): {sl:.2f}\n"
        f"TP (1:{OB_TARGET_R:.0f}): {tp:.2f}\n"
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


def send_outcome_telegram(symbol, strategy_label, action, entry, exit_price, sl, outcome, reason,
                          r, pips, pnl, pct, balance, subscribers, message_ids=None):
    """Final message for a fixed 1:3 Order Block trade."""
    if outcome == "win":
        tag = "✅ TAKE PROFIT HIT"
    elif outcome == "loss":
        tag = "❌ STOP LOSS HIT"
    else:
        tag = "➖ CLOSED"

    text = (
        f"{tag} ({strategy_label})\n"
        f"Symbol: {symbol} | {action}\n"
        f"Entry: {entry:.2f} -> Exit: {exit_price:.2f}\n"
        f"SL: {sl:.2f} | Target: 1:{OB_TARGET_R:.0f}\n"
        f"Result: {r:+.2f}R | {pips:+.1f} pips\n"
        f"P&L: {_money(pnl)} ({pct:+.2f}%) on {LOT_SIZE} lot\n"
        f"Demo balance: ${balance:.2f}"
    )
    _send_reply(text, subscribers, message_ids)


def send_welcome(chat_id):
    text = (
        "✅ You're subscribed!\n"
        "You'll receive BTCUSDT and XAU/USD Order Block signals.\n"
        f"Entry = 50% of the Order Block | SL = OB boundary | TP = 1:{OB_TARGET_R:.0f}.\n"
        f"Signals use the confirmed volume-pivot Order Block logic. "
        f"Results are tracked on a ${INITIAL_BALANCE:.0f} demo balance ({LOT_SIZE} lot per trade). "
        "You'll also get daily, weekly, monthly & yearly reports."
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
# Pending trade outcome tracking: fixed SL / fixed 1:3 TP
# ---------------------------------------------------------------------------
def check_pending_trades(name, df, state):
    """
    Walk forward through new candles for existing Order Block trades.

    No old 1:1 breakeven, 1:2 partial-close, or EMA exit logic is used.
    Each trade stays open until:
      BUY  -> SL or 1:3 TP
      SELL -> SL or 1:3 TP

    If both SL and TP are inside the same OHLC candle, OHLC data cannot reveal
    the exact intrabar order. We use distance from the candle open as a
    deterministic approximation.
    """
    pending = state.setdefault("pending_trades", [])
    if not pending:
        return

    times = df["time"].tolist()
    subscribers = state.get("subscribers", [])
    still_pending = []

    for trade in pending:
        if trade["symbol"] != name:
            still_pending.append(trade)
            continue

        action = trade["action"]
        sl = float(trade["sl"])
        entry = float(trade["entry"])
        tp = float(trade["tp"])
        risk = abs(entry - sl)

        if risk <= 0:
            continue

        last_time = pd.to_datetime(
            trade.get("last_time") or trade["entry_time"], utc=True
        )
        message_ids = trade.get("message_ids", {})
        result = None

        for i in range(1, len(df)):
            if times[i] <= last_time:
                continue

            row = df.iloc[i]
            hi, lo, op = float(row["high"]), float(row["low"]), float(row["open"])

            if action == "BUY":
                sl_hit = lo <= sl
                tp_hit = hi >= tp
            else:
                sl_hit = hi >= sl
                tp_hit = lo <= tp

            events = []
            if sl_hit:
                events.append(("sl", sl))
            if tp_hit:
                events.append(("tp", tp))

            events.sort(key=lambda e: abs(e[1] - op))

            if events:
                kind, exit_price = events[0]
                if kind == "tp":
                    r_total = OB_TARGET_R
                    outcome = "win"
                    reason = "tp"
                else:
                    r_total = -1.0
                    outcome = "loss"
                    reason = "sl"

                result = (reason, outcome, r_total, float(exit_price), times[i])
                break

        if result is None:
            trade["last_time"] = str(times[-1])
            still_pending.append(trade)
            continue

        reason, outcome, r_total, exit_price, exit_time = result
        pnl = r_total * risk * CONTRACT_SIZE.get(name, 1.0) * LOT_SIZE
        balance_before = state.get("balance", INITIAL_BALANCE)
        balance_after = balance_before + pnl
        state["balance"] = balance_after
        pct = (pnl / balance_before * 100) if balance_before else 0.0
        pips = r_total * risk / _pip_size(name)

        send_outcome_telegram(
            name, trade["strategy_label"], action, entry, exit_price, sl, outcome,
            reason, r_total, pips, pnl, pct, balance_after, subscribers, message_ids
        )

        state.setdefault("trade_history", []).append({
            "symbol": name,
            "strategy_label": trade["strategy_label"],
            "action": action,
            "outcome": outcome,
            "reason": reason,
            "entry": entry,
            "exit": exit_price,
            "sl": sl,
            "tp": tp,
            "r": round(r_total, 4),
            "pips": round(pips, 2),
            "pnl_usd": round(pnl, 4),
            "balance_after": round(balance_after, 4),
            "entry_time": trade["entry_time"],
            "resolved_time": str(exit_time),
        })

        print(
            f"[{name}][{trade['strategy_label']}] Trade opened "
            f"{trade['entry_time']} closed ({reason}): "
            f"{outcome.upper()} {r_total:+.2f}R, {_money(pnl)}, "
            f"balance ${balance_after:.2f}"
        )

    state["pending_trades"] = still_pending

# Periodic performance reports (daily / weekly / monthly / yearly)
# ---------------------------------------------------------------------------
def build_report_text(title, trades, start_balance=None):
    if start_balance is None:
        start_balance = INITIAL_BALANCE
    if not trades:
        return f"📊 {title}\nNo trades in this period.\nDemo balance: ${start_balance:.2f}"

    sep = "━━━━━━━━━━━━━━━━"
    symbols = sorted({t["symbol"] for t in trades})
    strategies = sorted({t["strategy_label"] for t in trades})
    count_line = " | ".join(
        f"{sym}: {sum(1 for t in trades if t['symbol'] == sym)}" for sym in symbols
    )
    period_pnl = sum(t["pnl_usd"] for t in trades if t.get("pnl_usd") is not None)
    end_balance = start_balance + period_pnl
    balance_pct = (period_pnl / start_balance * 100) if start_balance else 0.0

    parts = [
        f"📊 {title}",
        sep,
        f"Demo balance: ${start_balance:.2f} -> ${end_balance:.2f} ({balance_pct:+.2f}%)",
        f"Trades by symbol: {count_line} | Total: {len(trades)}",
        "",
        _format_block("📌 Overall", trades, start_balance),
    ]
    for sym in symbols:
        icon = "🥇" if "XAU" in sym else "🪙"
        parts += ["", sep, _format_block(f"{icon} {sym}", [t for t in trades if t["symbol"] == sym], start_balance)]
    for label in strategies:
        parts += ["", sep, _format_block(f"🎯 {label}", [t for t in trades if t["strategy_label"] == label], start_balance)]
    parts += [
        "",
        f"ℹ️ Demo account: ${INITIAL_BALANCE:.0f} start, {LOT_SIZE} lot per trade. "
        "% = of the balance at the start of this period.",
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


def _pair_text(r, usd):
    if r is None:
        return "N/A"
    if usd is None:
        return f"{r:+.2f}R"
    return f"{r:+.2f}R ({_money(usd)})"


def _balance_before(history, moment):
    total = INITIAL_BALANCE
    for t in history:
        if t.get("pnl_usd") is not None and pd.to_datetime(t["resolved_time"], utc=True) < moment:
            total += t["pnl_usd"]
    return total


def _send_period_report(title, history, start, end, subscribers):
    trades = _trades_in_range(history, start, end)
    send_report_telegram(title, trades, subscribers, _balance_before(history, start))


def send_report_telegram(title, trades, subscribers, start_balance=None):
    text = build_report_text(title, trades, start_balance)
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


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
STRATEGIES = [
    ("order_block", "LuxAlgo Order Block 50% Entry", compute_order_block_signal),
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

    # First resolve previously-sent Order Block signals.
    try:
        check_pending_trades(name, df, state)
    except Exception as e:
        print(f"[{name}] ERROR while tracking open trades: {e}")

    for key_suffix, label, strategy_fn in STRATEGIES:
        try:
            action, entry, sl, tp, candle_time = strategy_fn(df)

            if is_market_stale(candle_time):
                print(
                    f"[{name}][{label}] Market appears closed "
                    f"(last candle: {candle_time}) -- skipping"
                )
                continue

            state_key = f"{name}_{key_suffix}"
            candle_key = str(candle_time)

            if action and state.get(state_key) != candle_key:
                message_ids = send_telegram(
                    name, action, entry, sl, tp, candle_time,
                    state.get("subscribers", []), label
                )
                state[state_key] = candle_key
                state.setdefault("pending_trades", []).append({
                    "symbol": name,
                    "strategy_label": label,
                    "action": action,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "entry_time": str(candle_time),
                    "last_time": str(candle_time),
                    "message_ids": message_ids,
                })
                print(
                    f"[{name}][{label}] Sent {action} signal at "
                    f"{candle_time}: entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f} "
                    f"to {len(state.get('subscribers', []))} subscriber(s)"
                )
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
