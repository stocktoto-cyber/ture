import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

# 設定頁面資訊
st.set_page_config(page_title="台股戰情室 v1.7 AI旗艦版", layout="wide", page_icon="🎯")

# 嘗試匯入 pandas_ta (選用)
use_pandas_ta = True
try:
    import pandas_ta as ta
except Exception:
    use_pandas_ta = False

# ==========================================
# 1. 工具函式 (Utils)
# ==========================================

def normalize_ticker(ticker_raw: str) -> str:
    if not ticker_raw:
        return ticker_raw
    t = ticker_raw.strip().upper()
    if '.' in t:
        return t
    if t.isdigit():
        return t + '.TW'
    if len(t) >= 2 and t[:-1].isdigit() and t[-1].isalpha():
        return t + '.TW'
    return t

def format_price(val):
    if val is None or pd.isna(val):
        return "-"
    return f"{int(val)}" if val >= 1000 else f"{val:.2f}"

def get_market_status():
    tw = pytz.timezone('Asia/Taipei')
    now = datetime.datetime.now(tw)
    is_trading_time = (
        (now.hour == 9) or
        (9 < now.hour < 13) or
        (now.hour == 13 and now.minute <= 35)
    ) and now.weekday() < 5
    return is_trading_time, now

def _get_cache_ttl():
    is_trading, _ = get_market_status()
    return 30 if is_trading else 3600

@st.cache_data(ttl=_get_cache_ttl())
def get_stock_data(ticker, start_date, end_date):
    try:
        df = yf.Ticker(ticker).history(start=start_date, end=end_date + timedelta(days=1))
        if not df.empty and df.index[-1].date() == datetime.date.today():
            intraday = yf.Ticker(ticker).history(period="1d", interval="1m")
            if not intraday.empty:
                idx = df.index[-1]
                df.at[idx, 'Close'] = intraday.iloc[-1]['Close']
                df.at[idx, 'High'] = max(df.at[idx, 'High'], intraday['High'].max())
                df.at[idx, 'Low'] = min(df.at[idx, 'Low'], intraday['Low'].min())
    except Exception as e:
        st.warning(f"資料抓取失敗 ({ticker}): {e}")
        return pd.DataFrame()
    return df

def compute_indicators(df):
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['MA20_slope'] = df['MA20'].diff()

    std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + 2 * std
    df['BB_Lower'] = df['MA20'] - 2 * std
    df['Vol_MA5'] = df['Volume'].rolling(5).mean()

    if use_pandas_ta:
        try:
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                cols = list(macd.columns)
                line_col = next((c for c in cols if c.startswith('MACD') and not c.startswith('MACDh') and not c.startswith('MACDs')), None)
                hist_col = next((c for c in cols if c.startswith('MACDh')), None)
                sig_col  = next((c for c in cols if c.startswith('MACDs')), None)
                if line_col and hist_col and sig_col:
                    df['MACD'] = macd[line_col]
                    df['MACD_signal'] = macd[sig_col]
                    df['MACD_hist'] = macd[hist_col]
        except Exception as e:
            st.warning(f"pandas_ta MACD 計算失敗，改用內建: {e}")

    if 'MACD' not in df.columns:
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    try:
        if use_pandas_ta:
            df['RSI'] = ta.rsi(df['Close'], length=14)
        else:
            raise ValueError("pandas_ta 未安裝")
    except Exception:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    return df

def recommend_strategy_mode(df):
    if df.empty or len(df) < 20:
        return None, None
    last = df.iloc[-1]
    if len(df) < 6:
        return None, None
    prev_5d = df.iloc[-5]
    if pd.isna(prev_5d['MA20']) or prev_5d['MA20'] == 0:
        return None, None

    ma20_slope_5d = (last['MA20'] - prev_5d['MA20']) / prev_5d['MA20']
    rsi_val = last.get('RSI', 50)
    macd_positive = last.get('MACD_hist', 0) > 0

    if ma20_slope_5d > 0.005 and macd_positive:
        return "Trend", "🚀 **強勢多頭格局**：月線上揚且 MACD 動能正向，適合「趨勢突破」策略，順勢操作。"
    elif ma20_slope_5d > 0.005:
        return "Trend", "📈 **多頭格局**：月線上揚，適合「趨勢突破」策略，但 MACD 動能稍弱，留意假突破。"
    elif ma20_slope_5d < -0.005 and rsi_val < 35:
        return "Dip", "💎 **超賣反彈機會**：月線下彎但 RSI 進入超賣，適合「乖離抄底」策略，搶短反彈。"
    elif ma20_slope_5d < -0.005:
        return "Dip", "🐻 **空頭或修正格局**：月線下彎，切勿追高。適合「乖離抄底」策略，等待落底訊號。"
    else:
        return "Dip", "🦀 **盤整震盪格局**：月線走平，缺乏波段動能。適合「乖離抄底」策略，在箱型區間低買高賣。"

# ==========================================
# 2. 策略引擎 (Strategy)
# ==========================================

def _calc_addon_signals(df):
    addon_triggers = np.zeros(len(df), dtype=bool)
    closes = df['Close'].values
    ma20s = df['MA20'].values
    ma20_ups = df['MA20_slope'].values > 0
    buy_trigs = df['buy_trigger'].values
    sell_trigs = df['sell_trigger'].values

    in_position = False
    last_action_idx = -999

    for i in range(len(df)):
        if buy_trigs[i]:
            in_position = True
            last_action_idx = i
        elif sell_trigs[i]:
            in_position = False
        elif in_position and not np.isnan(ma20s[i]) and ma20s[i] > 0:
            dist_to_ma = (closes[i] - ma20s[i]) / ma20s[i]
            if 0 <= dist_to_ma <= 0.01 and ma20_ups[i] and (i - last_action_idx) > 5:
                addon_triggers[i] = True
                last_action_idx = i

    return addon_triggers

def generate_signals(df, strategy_type='Trend', vol_multiplier=1.3, confirm_days=2,
                     min_room_pct=0.03, dip_threshold=0.06, rsi_threshold=30,
                     sell_confirm_days=1, stop_loss_pct=0.0):
    df = df.copy()

    is_trading, now = get_market_status()
    last_date = df.index[-1].date()
    today_date = now.date()

    df['Proj_Volume'] = df['Volume'].astype(float)
    if last_date == today_date and is_trading:
        start_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
        minutes_passed = (now - start_time).total_seconds() / 60
        if minutes_passed > 0:
            proj_vol = (df['Volume'].iloc[-1] / minutes_passed) * 270 * 0.9
            df.iloc[-1, df.columns.get_loc('Proj_Volume')] = proj_vol

    df['MA20_slope'] = df['MA20'].diff()
    df['above_MA20'] = df['Close'] > df['MA20']
    df['below_MA20'] = df['Close'] < df['MA20']

    mask_below = df['below_MA20'].fillna(False)
    df['grp_below'] = (mask_below != mask_below.shift()).cumsum()
    df['consec_below'] = df.groupby('grp_below')['below_MA20'].cumsum()
    df.loc[~mask_below, 'consec_below'] = 0
    df['sell_condition'] = df['consec_below'] >= sell_confirm_days
    df['sell_trigger'] = df['sell_condition'] & (~df['sell_condition'].shift(1).fillna(False))

    df['buy_trigger'] = False
    df['addon_trigger'] = False
    df['condition_met'] = False

    if strategy_type == 'Trend':
        df['MA20_up'] = df['MA20_slope'] > 0
        df['vol_breakout'] = df['Proj_Volume'] > (df['Vol_MA5'] * vol_multiplier)

        room = (df['BB_Upper'] - df['Close']) / df['Close']
        df['room_ok'] = (room >= min_room_pct) | (df['Close'] >= df['BB_Upper'])

        raw_buy = (
            df['above_MA20'] &
            df['MA20_up'] &
            (df['MACD_hist'] > 0) &
            df['vol_breakout']
        )
        df['probe_buy'] = raw_buy & df['room_ok']

        mask_above = df['above_MA20'].fillna(False)
        df['grp_above'] = (mask_above != mask_above.shift()).cumsum()
        df['consec_above'] = df.groupby('grp_above')['above_MA20'].cumsum()
        df.loc[~mask_above, 'consec_above'] = 0

        df['condition_met'] = df['probe_buy'] & (df['consec_above'] >= confirm_days)
        df['buy_trigger'] = df['condition_met'] & (~df['condition_met'].shift(1).fillna(False))
        df['addon_trigger'] = _calc_addon_signals(df)

    elif strategy_type == 'Dip':
        df['Bias_MA20'] = (df['Close'] - df['MA20']) / df['MA20']
        dip_condition = df['Bias_MA20'] < (-1 * dip_threshold)
        rsi_condition = df['RSI'] < rsi_threshold
        df['condition_met'] = dip_condition & rsi_condition
        df['buy_trigger'] = df['condition_met'] & (~df['condition_met'].shift(1).fillna(False))
        df['addon_trigger'] = False

    if stop_loss_pct > 0:
        entry_price = None
        stop_trigger = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            if df['buy_trigger'].iloc[i] or df['addon_trigger'].iloc[i]:
                entry_price = df['Close'].iloc[i]
            elif df['sell_trigger'].iloc[i]:
                entry_price = None
            elif entry_price is not None:
                drawdown = (df['Close'].iloc[i] - entry_price) / entry_price
                if drawdown <= -stop_loss_pct:
                    stop_trigger[i] = True
                    entry_price = None
        df['sell_trigger'] = df['sell_trigger'] | stop_trigger

    return df

# ==========================================
# 3. 回測與繪圖 (Backtest & Plot)
# ==========================================

def run_backtest(df):
    if df.empty:
        return None, pd.DataFrame()

    trades = []
    active_batches = []
    type_map = {'Initial': '初始建倉', 'Add-on': '加碼進場'}
    equity_curve = [1.0]

    for i in range(len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        if len(active_batches) == 0 and row.get('buy_trigger', False):
            active_batches.append({'entry_date': nxt.name, 'entry_price': nxt['Open'], 'type': 'Initial'})
        elif len(active_batches) > 0 and row.get('addon_trigger', False):
            active_batches.append({'entry_date': nxt.name, 'entry_price': nxt['Open'], 'type': 'Add-on'})
        elif len(active_batches) > 0 and row.get('sell_trigger', False):
            exit_price = nxt['Open']
            batch_rets = []
            for batch in active_batches:
                ret = (exit_price - batch['entry_price']) / batch['entry_price']
                batch_rets.append(ret)
                trades.append({
                    '交易類型': type_map.get(batch['type'], batch['type']),
                    '進場日期': batch['entry_date'].strftime('%Y-%m-%d'),
                    '出場日期': nxt.name.strftime('%Y-%m-%d'),
                    '進場價格': batch['entry_price'],
                    '出場價格': exit_price,
                    '報酬率': ret
                })
            avg_ret = np.mean(batch_rets) if batch_rets else 0
            equity_curve.append(equity_curve[-1] * (1 + avg_ret))
            active_batches = []

    if active_batches:
        last_price = df.iloc[-1]['Close']
        batch_rets = []
        for batch in active_batches:
            ret = (last_price - batch['entry_price']) / batch['entry_price']
            batch_rets.append(ret)
            trades.append({
                '交易類型': type_map.get(batch['type'], batch['type']),
                '進場日期': batch['entry_date'].strftime('%Y-%m-%d'),
                '出場日期': f"{df.index[-1].strftime('%Y-%m-%d')} (未平倉)",
                '進場價格': batch['entry_price'],
                '出場價格': last_price,
                '報酬率': ret
            })
        equity_curve.append(equity_curve[-1] * (1 + np.mean(batch_rets)))

    if not trades:
        return None, pd.DataFrame()

    tdf = pd.DataFrame(trades)
    equity_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    mdd = drawdown.min()

    stats = {
        'Win Rate': len(tdf[tdf['報酬率'] > 0]) / len(tdf),
        'Total Return (Compound)': equity_arr[-1] - 1,
        'Total Return (Sum)': tdf['報酬率'].sum(),
        'Trade Count': len(tdf),
        'MDD': mdd
    }
    return stats, tdf

def plot_chart(df, ticker, stats, show_bb, sub_ind):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.15, 0.25], vertical_spacing=0.03,
        subplot_titles=(f'{ticker} K線', 'Volume (含預估)', sub_ind)
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='K線', increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange'), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='MA60'), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', dash='dot'), name='BB上緣'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', dash='dot'),
                                  fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name='BB下緣'), row=1, col=1)

    buys   = df[df['buy_trigger']]   if 'buy_trigger'   in df.columns else pd.DataFrame()
    addons = df[df['addon_trigger']] if 'addon_trigger' in df.columns else pd.DataFrame()
    sells  = df[df['sell_trigger']]  if 'sell_trigger'  in df.columns else pd.DataFrame()
    last_idx = df.index[-1]
    is_dip_mode = 'Bias_MA20' in df.columns

    if not buys.empty:
        real_buys = buys[buys.index != last_idx]
        marker_symbol = 'circle' if is_dip_mode else 'triangle-up'
        marker_color = '#00BFFF' if is_dip_mode else 'yellow'
        marker_name = '抄底買進' if is_dip_mode else '初始買進'
        fig.add_trace(go.Scatter(
            x=real_buys.index, y=real_buys['Low'] * 0.99, customdata=real_buys['Close'],
            mode='markers', marker=dict(symbol=marker_symbol, color=marker_color, size=12),
            name=marker_name,
            hovertemplate=f'日期: %{{x}}<br>{marker_name}: %{{customdata:.2f}}<extra></extra>'
        ), row=1, col=1)
        if last_idx in buys.index:
            fig.add_trace(go.Scatter(
                x=[last_idx], y=[buys.loc[last_idx]['Low'] * 0.99],
                customdata=[buys.loc[last_idx]['Close']], mode='markers',
                marker=dict(symbol=marker_symbol + '-open', color=marker_color, size=15, line=dict(width=2)),
                name=f'⚡ 盤中{marker_name}',
                hovertemplate=f'日期: %{{x}}<br>⚡ 盤中價格: %{{customdata:.2f}}<extra></extra>'
            ), row=1, col=1)

    if not addons.empty:
        real_addons = addons[addons.index != last_idx]
        fig.add_trace(go.Scatter(
            x=real_addons.index, y=real_addons['Low'] * 0.99, customdata=real_addons['Close'],
            mode='markers', marker=dict(symbol='triangle-up', color='purple', size=12),
            name='回測加碼', hovertemplate='日期: %{x}<br>加碼點: %{customdata:.2f}<extra></extra>'
        ), row=1, col=1)

    if not sells.empty:
        real_sells = sells[sells.index != last_idx]
        fig.add_trace(go.Scatter(
            x=real_sells.index, y=real_sells['High'] * 1.01, customdata=real_sells['Close'],
            mode='markers', marker=dict(symbol='triangle-down', color='lime', size=10),
            name='賣出訊號', hovertemplate='日期: %{x}<br>賣出點: %{customdata:.2f}<extra></extra>'
        ), row=1, col=1)
        if last_idx in sells.index:
            fig.add_trace(go.Scatter(
                x=[last_idx], y=[sells.loc[last_idx]['High'] * 1.01],
                customdata=[sells.loc[last_idx]['Close']], mode='markers',
                marker=dict(symbol='triangle-down-open', color='lime', size=15, line=dict(width=2)),
                name='⚡ 盤中預警賣出',
                hovertemplate='日期: %{x}<br>⚡ 盤中價格: %{customdata:.2f}<extra></extra>'
            ), row=1, col=1)

    colors = ['red' if r['Close'] >= r['Open'] else 'green' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='實際成交量'), row=2, col=1)

    if 'Proj_Volume' in df.columns and get_market_status()[0]:
        last_vol = df.iloc[-1]['Volume']
        last_proj = df.iloc[-1]['Proj_Volume']
        if last_proj > last_vol:
            fig.add_trace(go.Bar(
                x=[df.index[-1]], y=[last_proj - last_vol], base=[last_vol],
                marker_color='rgba(255, 255, 0, 0.3)', name='盤中預估增量'
            ), row=2, col=1)

    if sub_ind == 'MACD':
        hist_colors = ['red' if v >= 0 else 'green' for v in df['MACD_hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], marker_color=hist_colors, name='MACD柱'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='cyan'), name='DIF'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], line=dict(color='orange'), name='MACD'), row=3, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    title_str = f"<b>{ticker} 盤中即時分析 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d %H:%M')})</b>"
    if stats:
        title_str += (
            f" | 複利報酬: {stats['Total Return (Compound)']:.1%}"
            f" | MDD: <span style='color:red'>{stats['MDD']:.1%}</span>"
            f" | 勝率: {stats['Win Rate']:.1%}"
            f" | 交易次數: {stats['Trade Count']}"
        )

    fig.update_layout(
        template='plotly_dark', title=title_str, height=800,
        xaxis_rangeslider_visible=False, margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

# ==========================================
# 4. 快速掃描工具函式 (並發版)
# ==========================================

def _scan_single(raw_ticker, sell_confirm_days):
    try:
        df = yf.Ticker(raw_ticker).history(period="6mo")
        if df.empty:
            return None
        curr = df.iloc[-1]['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = compute_indicators(df)
        df = generate_signals(df, strategy_type='Trend', sell_confirm_days=sell_confirm_days)
        last = df.iloc[-1]
        is_trading, _ = get_market_status()

        s = "觀察"
        prefix = "⚡ " if (is_trading and last.name.date() == datetime.date.today()) else ""
        if last.get('buy_trigger'):
            s = f"{prefix}🔥 買點"
        elif last.get('addon_trigger'):
            s = f"{prefix}💜 加碼"
        elif last.get('sell_trigger'):
            s = f"{prefix}🟢 賣出"

        return {
            '代號': raw_ticker.replace('.TW', ''),
            '現價': format_price(curr),
            '月線': "🔴↗️" if last['MA20_slope'] > 0 else "🟢↘️",
            'RSI': f"{last['RSI']:.1f}" if not pd.isna(last['RSI']) else "-",
            '狀態': s
        }
    except Exception as e:
        return {'代號': raw_ticker.replace('.TW', ''), '現價': '-', '月線': '-', 'RSI': '-', '狀態': f'⚠️ 錯誤: {e}'}

# ==========================================
# 5. Streamlit UI
# ==========================================

st.title("🎯 台股戰情室 v1.7 AI旗艦版")

is_trading, now_time = get_market_status()
status_color = "lime" if is_trading else "gray"
status_text = f"盤中交易時段 ({now_time.strftime('%H:%M:%S')})" if is_trading else f"已收盤 ({now_time.strftime('%H:%M:%S')})"
st.markdown(f"**目前狀態：** <span style='color: {status_color};'>● {status_text}</span>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 個股策略分析", "📡 快速掃描", "🏛️ 大盤戰情"])

# --- Tab 1: 個股分析 ---
with tab1:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ticker_input = st.text_input("股票代號:", value='2330', help="輸入台股代號，例如 2330、0050")
    with c2:
        chk_backtest = st.checkbox("顯示回測數據", value=True)
        tog_bb = st.checkbox("顯示布林通道", value=False)
    with c3:
        btn_analyze = st.button("即時分析", type="primary", use_container_width=True)

    st.markdown("---")

    time_ranges = {
        "近半年 (預設)": "180d", "近 1 年": "1y", "近 3 年": "3y", "近 5 年": "5y",
        "AI 爆發 (2023~)": "AI", "FED 升息 (2022)": "FED", "疫情 (2020-21)": "COVID"
    }
    today = datetime.date.today()
    if 'range_select' not in st.session_state:
        st.session_state['range_select'] = "近半年 (預設)"
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = today - timedelta(days=180)
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = today

    def update_dates():
        code = time_ranges[st.session_state['range_select']]
        today = datetime.date.today()
        if code == "180d":    s = today - timedelta(days=180)
        elif code == "1y":    s = today - timedelta(days=365)
        elif code == "3y":    s = today - timedelta(days=365 * 3)
        elif code == "5y":    s = today - timedelta(days=365 * 5)
        elif code == "AI":    s = datetime.date(2023, 1, 1)
        elif code == "FED":   s, today = datetime.date(2022, 1, 1), datetime.date(2022, 12, 31)
        elif code == "COVID": s, today = datetime.date(2020, 1, 1), datetime.date(2021, 12, 31)
        else: s = today - timedelta(days=180)
        st.session_state['start_date'] = s
        st.session_state['end_date'] = today

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.selectbox("⏳ 回測區間:", list(time_ranges.keys()), key='range_select', on_change=update_dates)
    with rc2:
        col_d1, col_d2 = st.columns(2)
        with col_d1: start_date = st.date_input("開始日期", key='start_date')
        with col_d2: end_date   = st.date_input("結束日期", key='end_date')

    with st.expander("⚙️ 策略參數設定", expanded=True):
        st_strat = st.radio("⚔️ 交易策略模式", ["趨勢突破 (Trend)", "乖離抄底 (Dip)"], horizontal=True)
        st.divider()

        slider_sell_days = st.slider("賣出確認天數 (天)", 1, 5, 1,
                                      help="連續跌破月線幾天後才觸發賣出。1 代表跌破當下即觸發。")

        if "趨勢" in st_strat:
            strategy_mode = 'Trend'
            ec1, ec2, ec3 = st.columns(3)
            with ec1: slider_vol  = st.slider("量能倍數 (倍)", 1.0, 3.0, 1.3, 0.1)
            with ec2: slider_days = st.slider("買進確認天數 (天)", 1, 5, 2, 1)
            with ec3: slider_room = st.slider("布林最小空間", 0.0, 0.1, 0.03, 0.01)
            slider_dip, slider_rsi = 0.06, 30
        else:
            strategy_mode = 'Dip'
            st.info("📉 抄底邏輯：負乖離過大 + RSI 低檔時進場；依據上述「賣出確認天數」決定何時出場。")
            ec1, ec2 = st.columns(2)
            with ec1: slider_dip = st.slider("負乖離門檻 (%)", 0.0, 10.0, 2.0, 0.1) / 100
            with ec2: slider_rsi = st.slider("RSI 濾網 (<)", 10, 60, 45, 5)
            slider_vol, slider_days, slider_room = 1.3, 2, 0.03

        slider_stoploss = st.slider("固定止損 (%) — 0 表示不啟用", 0, 20, 0, 1,
                                     help="進場後跌超過此百分比強制出場。設為 0 則停用。") / 100

        tog_ind = st.radio("副圖指標", ['MACD', 'RSI'], horizontal=True,
                            index=1 if strategy_mode == 'Dip' else 0)

    if btn_analyze:
        t = normalize_ticker(ticker_input)
        with st.spinner(f"正在連線分析 {t} ..."):
            try:
                s_date, e_date = st.session_state['start_date'], st.session_state['end_date']
                df = get_stock_data(t, s_date - timedelta(days=150), e_date)

                if df.empty:
                    st.error("❌ 無資料，請確認股票代號是否正確。")
                else:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = compute_indicators(df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna())

                    rec_mode, rec_reason = recommend_strategy_mode(df)
                    if rec_mode:
                        if rec_mode == "Trend":
                            st.success(f"💡 AI 建議：**【趨勢突破】**\n\n{rec_reason}")
                        else:
                            st.warning(f"💡 AI 建議：**【乖離抄底】**\n\n{rec_reason}")

                    df = generate_signals(
                        df, strategy_type=strategy_mode,
                        vol_multiplier=slider_vol, confirm_days=slider_days,
                        min_room_pct=slider_room, dip_threshold=slider_dip,
                        rsi_threshold=slider_rsi, sell_confirm_days=slider_sell_days,
                        stop_loss_pct=slider_stoploss
                    )

                    mask = (df.index.date >= s_date) & (df.index.date <= e_date)
                    df_view = df.loc[mask].copy()

                    if df_view.empty:
                        st.warning("❌ 此區間無資料")
                    else:
                        last_bar = df_view.iloc[-1]
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("目前價格", format_price(last_bar['Close']))
                        m2.metric("MA20", format_price(last_bar['MA20']),
                                  delta="↗️ 多方" if last_bar['MA20_slope'] > 0 else "↘️ 空方")

                        if is_trading and last_bar.name.date() == now_time.date():
                            m3.metric("預估總量", f"{int(last_bar.get('Proj_Volume', 0) / 1000)} 張",
                                      delta="盤中推估", delta_color="off")
                        else:
                            m3.metric("成交量", f"{int(last_bar['Volume'] / 1000)} 張")

                        sig_status = "無訊號"
                        if last_bar.get('buy_trigger'):
                            sig_status = f"🔥 {'抄底' if strategy_mode == 'Dip' else '買進'}訊號"
                        elif last_bar.get('addon_trigger'):
                            sig_status = "💜 加碼訊號"
                        elif last_bar.get('sell_trigger'):
                            sig_status = "🟢 賣出訊號"
                        elif last_bar.get('condition_met') and strategy_mode == 'Trend':
                            sig_status = "💎 持倉續抱"
                        m4.metric("策略狀態", sig_status)

                        stats, tdf = run_backtest(df_view) if chk_backtest else (None, None)
                        st.plotly_chart(plot_chart(df_view, t, stats, tog_bb, tog_ind), use_container_width=True)

                        if stats is not None:
                            st.write("📝 **交易明細:**")
                            st.dataframe(
                                tdf.style
                                   .format({'進場價格': '{:.2f}', '出場價格': '{:.2f}', '報酬率': '{:.2%}'})
                                   .map(lambda x: 'color: #ff4b4b' if x > 0 else 'color: #00c853', subset=['報酬率']),
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"錯誤: {e}")

# --- Tab 2: 快速掃描 (並發版) ---
with tab2:
    st.info("盤中掃描：針對清單進行「趨勢策略」掃描（賣出判斷同設定天數）。已啟用並發請求，速度大幅提升。")
    txt_scan = st.text_area(
        "股票清單:",
        value="2330, 0050, 00881, 0056, 00878, 00919, 00713, 00929, 00675L, 00757",
        height=80
    )

    if st.button("執行掃描", type="primary"):
        raw_tickers = [normalize_ticker(t.strip()) for t in txt_scan.split(',') if t.strip()]
        results = []
        pb = st.progress(0)
        status_placeholder = st.empty()

        completed = 0
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_scan_single, t, slider_sell_days): t for t in raw_tickers}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                completed += 1
                pb.progress(completed / len(raw_tickers))
                status_placeholder.text(f"已完成 {completed}/{len(raw_tickers)} 支...")

        status_placeholder.empty()

        if results:
            result_df = pd.DataFrame(results)
            signal_order = {'🔥 買點': 0, '⚡ 🔥 買點': 0, '💜 加碼': 1, '⚡ 💜 加碼': 1,
                            '🟢 賣出': 2, '⚡ 🟢 賣出': 2, '觀察': 3}
            result_df['_order'] = result_df['狀態'].map(lambda x: signal_order.get(x, 9))
            result_df = result_df.sort_values('_order').drop(columns=['_order'])
            st.dataframe(result_df, use_container_width=True)

# --- Tab 3: 大盤 ---
with tab3:
    st.header("🏛️ 台灣加權指數 (TAIEX)")
    with st.spinner("Analyzing..."):
        twii_df = get_stock_data("^TWII", datetime.date.today() - timedelta(days=365), datetime.date.today())
        if not twii_df.empty:
            if isinstance(twii_df.columns, pd.MultiIndex):
                twii_df.columns = twii_df.columns.get_level_values(0)
            twii_df = compute_indicators(twii_df)
            last = twii_df.iloc[-1]
            prev = twii_df.iloc[-2]

            change = last['Close'] - prev['Close']
            change_pct = (change / prev['Close']) * 100
            bias_20 = (last['Close'] - last['MA20']) / last['MA20'] * 100
            rsi_val = last['RSI']

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("加權指數", f"{int(last['Close'])}", f"{change:.0f} ({change_pct:.2f}%)", delta_color="inverse")

            bias_label = "⚠️ 過熱" if bias_20 > 5 else "💎 超賣" if bias_20 < -5 else "正常"
            bias_color = "normal" if bias_20 > 5 else "inverse" if bias_20 < -5 else "off"
            m2.metric("月線乖離率", f"{bias_20:.2f}%", bias_label, delta_color=bias_color)

            rsi_state = "🔥 過熱" if rsi_val > 70 else "🥶 超賣" if rsi_val < 30 else "中性"
            m3.metric("RSI", f"{rsi_val:.1f}", rsi_state)

            macd_state = "📈 多方動能" if last['MACD_hist'] > 0 else "📉 空方動能"
            m4.metric("MACD 動能", macd_state)

            st.subheader("📋 戰情診斷")
            analysis_text = []
            if last['Close'] > last['MA20'] and last['MA20_slope'] > 0:
                analysis_text.append("✅ **趨勢偏多**：站上月線且月線翻揚。")
            elif last['Close'] < last['MA20']:
                analysis_text.append("⚠️ **趨勢轉弱**：跌破月線，需留意整理。")

            if rsi_val > 75:
                analysis_text.append("🔥 **高檔警戒**：RSI 過熱，不宜追高。")
            elif bias_20 < -6:
                analysis_text.append("💎 **乖離過大**：負乖離擴大，醞釀反彈。")

            if last['MACD_hist'] > 0 and last['MACD_hist'] > twii_df.iloc[-2]['MACD_hist']:
                analysis_text.append("📊 **MACD 擴張**：多方動能持續增強。")
            elif last['MACD_hist'] < 0:
                analysis_text.append("📊 **MACD 空方**：動能偏弱，觀望為宜。")

            for txt in analysis_text:
                st.write(txt)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(
                x=twii_df.index, open=twii_df['Open'], high=twii_df['High'],
                low=twii_df['Low'], close=twii_df['Close'],
                increasing_line_color='red', decreasing_line_color='green', name='K線'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=twii_df.index, y=twii_df['MA20'], line=dict(color='orange'), name='MA20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=twii_df.index, y=twii_df['MA60'], line=dict(color='blue', width=1), name='MA60'), row=1, col=1)
            fig.add_trace(go.Scatter(x=twii_df.index, y=twii_df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.update_layout(template='plotly_dark', height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
