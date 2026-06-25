import streamlit as st
import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime
import ccxt

# ---------------------------- CONFIGURATION ----------------------------
EXCHANGE_SYMBOL = 'BTC/USDT'
MAX_RISK_PERCENT = 1.0
DEFAULT_BALANCE = 1000

# ---------------------------- DATA LOADING (local CSV) ----------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_data():
    try:
        df = pd.read_csv('btc_data.csv')
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.dropna(inplace=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].apply(lambda x: float(x) if pd.notnull(x) else None)
        return df
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return pd.DataFrame()

# ---------------------------- STRATEGY LIBRARY ----------------------------
def ema_cross(close):
    fast = vbt.MA.run(close, 9).ma
    slow = vbt.MA.run(close, 21).ma
    entries = fast > slow
    exits = fast < slow
    return entries, exits

def rsi_mean_rev(close, period=14, low=30, high=70):
    rsi = vbt.RSI.run(close, period).rsi
    entries = (rsi < low)
    exits = (rsi > high)
    return entries, exits

def breakout(high, low, close):
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    entries = close > prev_high
    exits = close < prev_low
    return entries, exits

def squeeze(high, low, close):
    bb = vbt.BBANDS.run(close)
    kc_mid = vbt.MA.run(close, 20).ma
    atr = vbt.ATR.run(high, low, close, 10).atr
    kc_upper = kc_mid + 1.5 * atr
    kc_lower = kc_mid - 1.5 * atr
    squeeze_on = (bb.upper < kc_upper) & (bb.lower > kc_lower)
    adx = vbt.ADX.run(high, low, close, 14).adx
    entries = squeeze_on & (adx > 20) & (close > bb.middle)
    exits = squeeze_on & (adx < 20) | (close < bb.lower)
    return entries, exits

strategies = [
    ("EMA Crossover", ema_cross),
    ("RSI Mean Reversion", rsi_mean_rev),
    ("Breakout", breakout),
    ("Squeeze", squeeze)
]

# ---------------------------- BACKTEST ENGINE ----------------------------
def backtest_strategy(name, func, df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    if name in ['EMA Crossover', 'RSI Mean Reversion', 'Squeeze']:
        entries, exits = func(close)
    elif name == 'Breakout':
        entries, exits = func(high, low, close)
    else:
        return None
    pf = vbt.Portfolio.from_signals(
        close, entries, exits,
        sl_stop=0.02, tp_stop=0.03,
        freq='1h', init_cash=1000
    )
    if len(pf.trades) == 0:
        return None
    sharpe = float(pf.sharpe_ratio() or 0)
    win_rate = float(pf.trades.win_rate() or 0)
    max_dd = float(pf.max_drawdown() or 1)
    score = sharpe * win_rate * (1 - max_dd)
    return {"pf": pf, "score": score, "trades": len(pf.trades), "win_rate": win_rate}

def select_best_strategy(df):
    recent = df.iloc[-48:]
    best_score = -np.inf
    best_name = None
    best_pf = None
    for name, func in strategies:
        res = backtest_strategy(name, func, recent)
        if res is None:
            continue
        if res['score'] > best_score:
            best_score = res['score']
            best_name = name
            best_pf = res['pf']
    # fallback if nothing works
    if best_name is None:
        best_name = "EMA Crossover (Fallback)"
        # generate a fresh entry/exit on full data for signal only
        close = recent['Close']
        entries, exits = ema_cross(close)
        best_pf = None
    return best_name, best_pf

# ---------------------------- TRADE SIGNAL DIRECTION ----------------------------
def get_trade_signal(df, best_name):
    close, high, low = df['Close'], df['High'], df['Low']
    if 'EMA Crossover' in best_name:
        fast = vbt.MA.run(close, 9).ma
        slow = vbt.MA.run(close, 21).ma
        return "LONG" if fast.iloc[-1] > slow.iloc[-1] else "SHORT"
    elif best_name == 'RSI Mean Reversion':
        rsi = vbt.RSI.run(close, 14).rsi.iloc[-1]
        return "LONG" if rsi < 30 else ("SHORT" if rsi > 70 else "NEUTRAL")
    elif best_name == 'Breakout':
        if close.iloc[-1] > high.iloc[-2]:
            return "LONG"
        elif close.iloc[-1] < low.iloc[-2]:
            return "SHORT"
        else:
            return "NEUTRAL"
    elif best_name == 'Squeeze':
        bb = vbt.BBANDS.run(close)
        adx = vbt.ADX.run(high, low, close, 14).adx.iloc[-1]
        if adx > 20 and close.iloc[-1] > bb.middle.iloc[-1]:
            return "LONG"
        elif adx > 20 and close.iloc[-1] < bb.middle.iloc[-1]:
            return "SHORT"
        else:
            return "NEUTRAL"
    else:
        return "NEUTRAL"

# ---------------------------- EXCHANGE CONNECTION ----------------------------
def get_exchange(mode):
    if mode == 'Paper':
        return None
    elif mode == 'Demo':
        if 'DEMO_API_KEY' not in st.secrets or 'DEMO_SECRET_KEY' not in st.secrets:
            st.error("Demo API keys missing.")
            return None
        exchange = ccxt.binance({
            'apiKey': st.secrets['DEMO_API_KEY'],
            'secret': st.secrets['DEMO_SECRET_KEY'],
            'enableRateLimit': True,
        })
        exchange.urls['api'] = 'https://testnet.binance.vision/api'
        exchange.urls['wss'] = 'wss://testnet.binance.vision/ws'
        return exchange
    elif mode == 'Live':
        if 'LIVE_API_KEY' not in st.secrets or 'LIVE_SECRET_KEY' not in st.secrets:
            st.error("Live API keys missing.")
            return None
        return ccxt.binance({
            'apiKey': st.secrets['LIVE_API_KEY'],
            'secret': st.secrets['LIVE_SECRET_KEY'],
            'enableRateLimit': True,
        })
    else:
        return None

def place_bracket_order(mode, direction, entry, stop, target, size):
    exchange = get_exchange(mode)
    if exchange is None:
        return False
    entry_order = exchange.create_order(
        symbol=EXCHANGE_SYMBOL,
        type='limit',
        side=direction.lower(),
        amount=size,
        price=entry,
    )
    st.write(f"✅ Entry order placed: {entry_order['id']}")
    exit_side = 'sell' if direction == 'LONG' else 'buy'
    oco_order = exchange.create_order(
        symbol=EXCHANGE_SYMBOL,
        type='OCO',
        side=exit_side,
        amount=size,
        price=target,
        stopPrice=stop,
        stopLimitPrice=stop * 0.99 if direction == 'LONG' else stop * 1.01,
    )
    st.write(f"🛡️ OCO exit placed: {oco_order['id']} (TP @{target}, SL @{stop})")
    return True

# ---------------------------- UI ----------------------------
st.set_page_config(page_title="AI Trade Co‑Pilot", layout="centered")
st.title("🤖 AI Trade Co‑Pilot")
st.markdown("**The AI chooses the strategy. You just review & confirm.**")

# Sidebar
st.sidebar.header("⚙️ Account")
mode = st.sidebar.selectbox("Trading Mode", ["Paper", "Demo", "Live"], index=0)
if mode == "Demo":
    st.sidebar.info("🧪 Demo Mode: Binance Testnet – fake money.")
elif mode == "Live":
    st.sidebar.warning("🚀 Live Mode: real Binance orders. Be careful.")
else:
    st.sidebar.info("📄 Paper Mode: simulation only.")

balance = st.sidebar.number_input("Account Balance (USDT)", value=float(DEFAULT_BALANCE), step=100.0)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.5, 3.0, float(MAX_RISK_PERCENT), step=0.1)

# Load data
df = load_data()
if df.empty:
    st.error("Data file missing. Add 'btc_data.csv' to the repository.")
    st.stop()

# AI selects strategy
best_strat, best_pf = select_best_strategy(df)

if best_strat is None:
    st.warning("No suitable strategy found. Try again later.")
else:
    direction = get_trade_signal(df, best_strat)
    if direction == "NEUTRAL":
        st.info("AI sees no clear direction. No trade suggested.")
    else:
        latest = df.iloc[-1]
        entry = float(latest['Close'])
        atr = float(vbt.ATR.run(df['High'], df['Low'], df['Close'], 14).atr.iloc[-1])
        if direction == "LONG":
            stop = entry - 2 * atr
            target = entry + 2 * atr
        else:
            stop = entry + 2 * atr
            target = entry - 2 * atr

        risk_amount = float(balance * (risk_pct / 100))
        distance = abs(entry - stop)
        size = float(risk_amount / distance) if distance > 0 else 0.0
        size = round(size, 5)

        if "Fallback" in best_strat:
            st.warning("⚠️ No optimal strategy found. Showing default EMA Crossover for demonstration.")

        st.subheader(f"📊 AI Selected Strategy: **{best_strat}**")
        col1, col2 = st.columns(2)
        col1.metric("Direction", direction)
        col2.metric("Entry Price", f"{entry:.2f} USDT")

        st.markdown(f"""
        | Stop Loss | Take Profit | Position Size (BTC) | Risk (USDT) |
        |-----------|-------------|---------------------|-------------|
        | {stop:.2f} | {target:.2f} | {size:.5f} | {risk_amount:.2f} |
        """)

        if st.button("📋 Apply Strategy"):
            st.session_state.confirm = True

        if st.session_state.get("confirm"):
            st.subheader("✅ Confirm Trade")
            st.write(f"**Strategy:** {best_strat}")
            st.write(f"**Market:** {EXCHANGE_SYMBOL}")
            st.write(f"**Type:** {direction}")
            st.write(f"**Entry:** {entry:.2f} USDT")
            st.write(f"**Stop Loss:** {stop:.2f}")
            st.write(f"**Take Profit:** {target:.2f}")
            st.write(f"**Size:** {size:.5f} BTC")
            st.write(f"**Risk:** {risk_amount:.2f} USDT ({risk_pct}% of balance)")

            if mode == "Paper":
                if st.button("🚀 Execute Trade (Paper)"):
                    st.success(f"Paper trade placed! {direction} {size:.5f} {EXCHANGE_SYMBOL} at {entry:.2f}. Stop: {stop:.2f}, Target: {target:.2f}")
                    st.balloons()
            else:
                if st.button("🚀 Execute Trade"):
                    with st.spinner("Placing orders..."):
                        success = place_bracket_order(mode, direction, entry, stop, target, size)
                    if success:
                        st.success(f"{mode} trade placed! Check your Binance app.")
                        st.balloons()
