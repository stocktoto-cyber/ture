import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="楚狂人策略-批量掃描神器", layout="wide")

# --- 側邊欄：設定股票清單 ---
st.sidebar.title("🔍 掃描設定")
st.sidebar.write("輸入股票代號 (用逗號分隔):")
st.sidebar.caption("💡 提示：台股直接輸入數字即可 (如 2330)，美股輸入代號 (如 AAPL)。")

# 預設清單 (改為純數字，不需 .TW)
default_tickers = "2330, 2454, 2317, 0050, 0056, 00878, 00919, 00713, 2303, 2603"
user_input = st.sidebar.text_area("股票清單", value=default_tickers, height=150)

scan_button = st.sidebar.button("🚀 開始掃描", type="primary")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **分類邏輯說明：**
    
    1. **🚨 潛在買點 (Day 1)**
       - 昨日收盤 < 月線
       - 目前價格 > 月線
       - *建議：等待隔日中午確認*
       
    2. **⚠️ 潛在賣點 (Day 1)**
       - 昨日收盤 > 月線
       - 目前價格 < 月線
       - *建議：等待隔日中午確認*
       
    3. **✅ 多頭趨勢**
       - 連續兩日都在月線上
       
    4. **🔻 空頭趨勢**
       - 連續兩日都在月線下
    """
)

# --- 核心函數 ---
def process_ticker(ticker_input):
    """
    自動處理代號後綴
    1. 去除空白
    2. 如果是純數字 (如 2330) -> 自動加上 .TW
    3. 如果已有後綴或為美股 -> 維持原樣
    """
    clean_ticker = ticker_input.strip().upper()
    if clean_ticker.isdigit():
        return f"{clean_ticker}.TW"
    return clean_ticker

def get_strategy_status(ticker):
    try:
        # 下載資料
        df = yf.download(ticker, period="3mo", progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if len(df) < 20:
            return None # 資料不足

        # 計算 MA20
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 取最後兩筆資料
        last_row = df.iloc[-1]      # 最新 (Today)
        prev_row = df.iloc[-2]      # 昨日 (Yesterday)
        
        # 數據提取
        price_now = last_row['Close']
        ma20_now = last_row['MA20']
        price_prev = prev_row['Close']
        ma20_prev = prev_row['MA20']

        # 狀態判斷
        is_now_above = price_now > ma20_now
        is_prev_above = price_prev > ma20_prev
        
        status = ""
        category = "" 
        
        if is_now_above and not is_prev_above:
            status = "🚨 突破月線 (Day 1) - 待確認"
            category = "1_Buy_Watch"
        elif not is_now_above and is_prev_above:
            status = "⚠️ 跌破月線 (Day 1) - 待確認"
            category = "2_Sell_Watch"
        elif is_now_above and is_prev_above:
            status = "✅ 多頭趨勢 - 續抱"
            category = "3_Bullish"
        else:
            status = "🔻 空頭趨勢 - 觀望"
            category = "4_Bearish"

        # 計算乖離率
        bias = ((price_now - ma20_now) / ma20_now) * 100

        # 顯示用的代號 (拿掉 .TW 讓版面比較乾淨，除非是美股)
        display_ticker = ticker.replace(".TW", "")

        return {
            "代號": display_ticker,
            "現價": round(price_now, 2),
            "月線(20MA)": round(ma20_now, 2),
            "乖離率(%)": round(bias, 2),
            "狀態": status,
            "Category": category
        }

    except Exception as e:
        return None

# --- 主程式 ---
st.title("📊 楚狂人月線策略 - 批量掃描儀表板")
st.write("此工具依照「月線 (20MA) 與延遲確認策略」自動掃描您的關注清單。")

if scan_button:
    # 處理輸入清單 (加入自動後綴邏輯)
    raw_list = user_input.split(",")
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_stocks = len(raw_list)
    
    for i, raw_t in enumerate(raw_list):
        if not raw_t.strip(): continue # 跳過空白
        
        # 自動轉換代號
        ticker = process_ticker(raw_t)
        
        status_text.text(f"正在分析: {ticker} ...")
        res = get_strategy_status(ticker)
        if res:
            results.append(res)
        
        progress_bar.progress((i + 1) / total_stocks)
    
    status_text.text("掃描完成！")
    progress_bar.empty()
    
    # 顯示結果
    if results:
        df_res = pd.DataFrame(results)
        
        # --- 分組顯示 ---
        
        # 1. 潛在機會
        buy_watch = df_res[df_res['Category'] == "1_Buy_Watch"]
        if not buy_watch.empty:
            st.subheader("🚨 觀察名單：剛突破月線 (Day 1)")
            st.markdown("💡 **策略**：今日不動作，若 **明日中午 12:00** 價格仍 > 月線，則進場。")
            st.dataframe(buy_watch.drop(columns=['Category']), use_container_width=True)
        else:
            st.info("目前沒有「剛突破月線」的股票。")
            
        st.markdown("---")

        # 2. 風險警示
        sell_watch = df_res[df_res['Category'] == "2_Sell_Watch"]
        if not sell_watch.empty:
            st.subheader("⚠️ 警戒名單：剛跌破月線 (Day 1)")
            st.markdown("💡 **策略**：今日不動作，若 **明日中午 12:00** 價格仍 < 月線，則出場。")
            st.dataframe(sell_watch.drop(columns=['Category']), use_container_width=True)
        
        st.markdown("---")
        
        # 3. 多空一覽
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ 多頭排列 (安心區)")
            bullish = df_res[df_res['Category'] == "3_Bullish"]
            st.dataframe(bullish.drop(columns=['Category']), use_container_width=True)
            
        with col2:
            st.subheader("🔻 空頭排列 (觀望區)")
            bearish = df_res[df_res['Category'] == "4_Bearish"]
            st.dataframe(bearish.drop(columns=['Category']), use_container_width=True)

    else:
        st.error("查無數據，請確認股票代號是否正確。")

else:
    st.write("👈 請在左側輸入股票代號（例如：2330, 0050），並點擊「開始掃描」。")
