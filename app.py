"""
📚 彼得林區台股選股系統 — lynch.py
啟動：streamlit run lynch.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime

st.set_page_config(
    page_title="📚 彼得林區台股系統",
    layout="wide",
    page_icon="📚"
)

# ══════════════════════════════════════════════════
# 1. 常數與設定
# ══════════════════════════════════════════════════

ETF_CODES = {
    '0050','0056','00878','00919','00881','006208',
    '00757','00675L','00631L','00713','00929','00981A',
    '00692','00850','00896','00900','00905','00912',
    '00940','00941','00946','00953B','00960',
    '00679B','00687B','00720B','00764B','00772B',
}

LYNCH_CAT = {
    'fast':      ('🚀 快速成長股', '林區最愛！年成長 > 15%，PEG < 1 是絕佳機會。'),
    'stalwart':  ('📈 穩定成長股', '大型穩健，適合長期持有。留意不要付太高 P/E。'),
    'slow':      ('🐢 緩慢成長股', '成熟產業、高股息。重點看股息率，而非成長。'),
    'cyclical':  ('🔄 景氣循環股', '隨景氣起伏。低谷買進，高峰前出場，時機最重要。'),
    'turnaround':('🔁 轉機股',     '曾陷困境正在復甦。高風險高報酬，需深入了解原因。'),
    'asset':     ('🏦 資產股',     '帳面資產被低估。需自行評估土地、持股等隱藏價值。'),
    'etf':       ('📦 ETF／指數基金','分散工具，適合定期定額長期持有，林區建議不挑個股者優先選擇。'),
}

CYCLICAL_SECTORS = {'半導體', '航運', '鋼鐵', '塑化', '紙類', '建材營造', 'Semiconductors', 'Steel', 'Shipping'}

# ══════════════════════════════════════════════════
# 2. 工具函式
# ══════════════════════════════════════════════════

def normalize(code: str) -> str:
    code = code.strip().upper()
    if '.' not in code:
        code += '.TW'
    return code

def fmt_price(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f"{v:,.0f}" if v >= 100 else f"{v:.2f}"

def normalize_yield(v):
    """yfinance 對台灣 ETF 有時回傳百分比值（如 7.17）而非小數（0.0717），統一轉成小數"""
    if v is None:
        return None
    return v / 100 if v > 1 else v

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(code_full: str) -> dict:
    try:
        t = yf.Ticker(code_full)
        info = dict(t.info or {})
        if not info.get('earningsGrowth'):
            try:
                fin = t.financials
                if fin is not None and not fin.empty:
                    ni_row = None
                    for key in ['Net Income', 'Net Income Common Stockholders']:
                        if key in fin.index:
                            ni_row = fin.loc[key].dropna()
                            break
                    if ni_row is not None and len(ni_row) >= 2:
                        base = ni_row.iloc[1]
                        if base and base != 0:
                            info['earningsGrowth'] = (ni_row.iloc[0] - base) / abs(base)
            except Exception:
                pass
        return info
    except Exception:
        return {}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_price(code_full: str):
    try:
        hist = yf.Ticker(code_full).history(period='2d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return None

# ══════════════════════════════════════════════════
# 3. 林區分類 & 評分引擎
# ══════════════════════════════════════════════════

def is_etf(info: dict, code_base: str) -> bool:
    qt = (info.get('quoteType') or '').upper()
    return qt == 'ETF' or code_base in ETF_CODES

def calc_etf_score(info: dict):
    """ETF 專屬評分（0–100）：費用率(30) + 殖利率(30) + 3年報酬(25) + 規模(15)"""
    detail = {}
    total  = 0

    exp_r   = info.get('annualReportExpenseRatio') or info.get('expenseRatio')
    div_y   = normalize_yield(info.get('dividendYield') or info.get('yield'))
    ret_3yr = info.get('threeYearAverageReturn')
    ret_1yr = info.get('oneYearTotalReturn') or info.get('ytdReturn')
    assets  = info.get('totalAssets')

    exp_score = 0
    if exp_r is not None:
        if   exp_r <= 0.001: exp_score = 30
        elif exp_r <= 0.003: exp_score = 25
        elif exp_r <= 0.005: exp_score = 18
        elif exp_r <= 0.010: exp_score = 10
        else:                exp_score = 4
    detail['費用率'] = {
        'val': f"{exp_r*100:.2f}%" if exp_r is not None else 'N/A',
        'score': exp_score, 'max': 30, 'note': '越低越好，林區建議 < 0.3%'
    }
    total += exp_score

    div_score = 0
    if div_y is not None:
        y = div_y * 100
        if   y >= 6.0: div_score = 30
        elif y >= 5.0: div_score = 25
        elif y >= 4.0: div_score = 20
        elif y >= 3.0: div_score = 13
        elif y >= 1.0: div_score = 6
    detail['股息殖利率'] = {
        'val': f"{div_y*100:.2f}%" if div_y is not None else 'N/A',
        'score': div_score, 'max': 30, 'note': '高殖利率 ETF 適合存股族'
    }
    total += div_score

    ret_score = 0
    ret_val   = ret_3yr if ret_3yr is not None else ret_1yr
    ret_label = '3年平均報酬' if ret_3yr is not None else '近1年報酬'
    if ret_val is not None:
        r = ret_val * 100
        if   r >= 20: ret_score = 25
        elif r >= 15: ret_score = 20
        elif r >= 10: ret_score = 14
        elif r >=  5: ret_score = 8
        elif r >=  0: ret_score = 3
    detail[ret_label] = {
        'val': f"{ret_val*100:.1f}%" if ret_val is not None else 'N/A',
        'score': ret_score, 'max': 25, 'note': '歷史績效參考（過去≠未來）'
    }
    total += ret_score

    def fmt_aum(v):
        if not v: return 'N/A'
        if v >= 1e8: return f"{v/1e8:.0f} 億"
        return f"{v/1e4:.0f} 萬"
    aum_score = 0
    if assets:
        if   assets >= 1e11: aum_score = 15
        elif assets >= 1e10: aum_score = 12
        elif assets >= 1e9:  aum_score = 7
        elif assets >= 1e8:  aum_score = 3
    detail['資產規模 AUM'] = {
        'val': fmt_aum(assets),
        'score': aum_score, 'max': 15, 'note': '規模越大流動性越好、越穩定'
    }
    total += aum_score

    return total, detail


def grade_etf(score):
    if score >= 80: return '⭐⭐⭐⭐⭐', '超優質 ETF'
    if score >= 65: return '⭐⭐⭐⭐',   '優質 ETF'
    if score >= 50: return '⭐⭐⭐',     '穩健選擇'
    if score >= 35: return '⭐⭐',       '普通，可考慮替代'
    return               '⭐',          '費用或績效偏弱'


def lynch_etf_voice(info, score, detail):
    lines = []
    exp_r   = info.get('annualReportExpenseRatio') or info.get('expenseRatio')
    div_y   = info.get('dividendYield') or info.get('yield')
    assets  = info.get('totalAssets')
    ret_3yr = info.get('threeYearAverageReturn')

    div_y_n = normalize_yield(div_y)
    if exp_r is not None:
        if exp_r <= 0.003:
            lines.append(f"✅ 費用率 **{exp_r*100:.2f}%** 非常低。林區說：長期下來，低費用是複利的朋友，每省一分費用都是淨報酬。")
        elif exp_r > 0.01:
            lines.append(f"⚠️ 費用率 **{exp_r*100:.2f}%** 偏高。每年都在侵蝕你的報酬，有低費用的替代品嗎？")
    if div_y_n is not None and div_y_n > 0.04:
        lines.append(f"💰 殖利率 **{div_y_n*100:.2f}%**，對存股族來說很吸引人。林區提醒：也要確認這殖利率能否持續。")
    if ret_3yr is not None:
        if ret_3yr >= 0.15:
            lines.append(f"📈 三年平均報酬 **{ret_3yr*100:.1f}%**，表現亮眼。但記住：過去績效不代表未來。")
        elif ret_3yr < 0:
            lines.append(f"📉 三年平均報酬 **{ret_3yr*100:.1f}%** 是負的。這段時間標的市場表現不佳，評估是否符合你的長期邏輯。")
    if assets and assets < 1e9:
        lines.append("⚠️ 資產規模較小，流動性風險較高，進出場時要注意成交量。")
    lines.append("📦 **林區對 ETF 的核心觀點**：「如果你沒有時間研究個股，買廣泛市場指數 ETF，然後定期定額、長期持有。這是大多數人最聰明的做法。」")
    return lines


def classify_lynch(info: dict, code_base: str) -> str:
    if is_etf(info, code_base):
        return 'etf'
    eps_g    = info.get('earningsGrowth')
    rev_g    = info.get('revenueGrowth')
    sector   = info.get('sector', '') or ''
    industry = info.get('industry', '') or ''
    div_y    = normalize_yield(info.get('dividendYield') or 0) or 0

    for kw in CYCLICAL_SECTORS:
        if kw in sector or kw in industry:
            return 'cyclical'
    if (eps_g and eps_g > 0.15) or (rev_g and rev_g > 0.15):
        return 'fast'
    if eps_g is not None and eps_g < 0.05 and div_y > 0.03:
        return 'slow'
    if eps_g is not None and -0.4 < eps_g < 0:
        return 'turnaround'
    return 'stalwart'


def calc_score(info: dict, code_base: str):
    if is_etf(info, code_base):
        return None, {}, None

    detail = {}
    total  = 0
    pe      = info.get('trailingPE')
    eps_g   = info.get('earningsGrowth')
    rev_g   = info.get('revenueGrowth')
    debt_eq = info.get('debtToEquity')
    cash    = info.get('totalCash')
    debt    = info.get('totalDebt')

    peg = None
    peg_score = 0
    if pe and eps_g and eps_g > 0:
        peg = pe / (eps_g * 100)
        if   peg <= 0.5:  peg_score = 35
        elif peg <= 0.75: peg_score = 28
        elif peg <= 1.0:  peg_score = 20
        elif peg <= 1.5:  peg_score = 10
    detail['PEG 比率'] = {
        'val': f"{peg:.2f}" if peg else 'N/A',
        'score': peg_score, 'max': 35, 'note': 'PEG < 1 = 低估、< 0.5 = 超值'
    }
    total += peg_score

    eps_score = 0
    if eps_g is not None:
        g = eps_g * 100
        if   g > 25: eps_score = 25
        elif g > 15: eps_score = 20
        elif g > 10: eps_score = 15
        elif g >  5: eps_score = 8
    detail['EPS 年成長率'] = {
        'val': f"{eps_g*100:.1f}%" if eps_g is not None else 'N/A',
        'score': eps_score, 'max': 25, 'note': '林區偏好 > 15% 的高成長'
    }
    total += eps_score

    debt_score = 0
    if debt_eq is not None:
        d = debt_eq / 100
        if   d < 0.25: debt_score = 20
        elif d < 0.50: debt_score = 15
        elif d < 1.00: debt_score = 8
    detail['負債／股東權益'] = {
        'val': f"{debt_eq:.0f}%" if debt_eq is not None else 'N/A',
        'score': debt_score, 'max': 20, 'note': '越低越安全，林區偏好低負債公司'
    }
    total += debt_score

    cash_score = 0
    if cash and debt:
        ratio = cash / debt
        if   ratio >= 1.0: cash_score = 10
        elif ratio >= 0.5: cash_score = 5
    elif cash and not debt:
        cash_score = 10
    cash_str = 'N/A'
    if cash and debt and debt > 0:
        cash_str = f"{cash/debt:.0%}"
    elif cash and not debt:
        cash_str = '無負債'
    detail['現金覆蓋負債'] = {
        'val': cash_str,
        'score': cash_score, 'max': 10, 'note': '現金 > 負債 = 財務安全邊際高'
    }
    total += cash_score

    rev_score = 0
    if rev_g is not None:
        g = rev_g * 100
        if   g > 15: rev_score = 10
        elif g >  5: rev_score = 7
        elif g >  0: rev_score = 3
    detail['營收年成長率'] = {
        'val': f"{rev_g*100:.1f}%" if rev_g is not None else 'N/A',
        'score': rev_score, 'max': 10, 'note': '成長動能持續性'
    }
    total += rev_score

    return total, detail, peg


def grade(score):
    if score >= 80: return '⭐⭐⭐⭐⭐', '林區強力推薦'
    if score >= 65: return '⭐⭐⭐⭐',   '符合林區標準'
    if score >= 50: return '⭐⭐⭐',     '部分符合，持續觀察'
    if score >= 35: return '⭐⭐',       '偏弱，謹慎持有'
    return               '⭐',          '不符合林區標準'


def action(score, pnl_pct=None):
    if score is None:
        return '📦 ETF：定期定額，長期持有'
    if score >= 65:
        if pnl_pct is not None and pnl_pct < -10:
            return '💎 基本面佳 × 跌深：逢低加碼好時機'
        return '✅ 基本面健康，繼續持有 / 可考慮加碼'
    if score >= 50:
        return '👀 基本面普通，持有觀察，勿輕易加碼'
    if score >= 35:
        return '⚠️ 基本面轉弱，考慮減碼或設停損'
    return '🚪 不符林區標準，建議重新審視持倉理由'


def lynch_voice(info, score, peg, cat_key):
    lines = []
    eps_g   = info.get('earningsGrowth')
    debt_eq = info.get('debtToEquity')
    cash    = info.get('totalCash')
    debt    = info.get('totalDebt')

    if peg:
        if peg < 0.5:
            lines.append(f"🔥 PEG 只有 **{peg:.2f}**！這正是我最愛找的機會——股價嚴重低估公司的成長速度。")
        elif peg < 1.0:
            lines.append(f"✅ PEG = **{peg:.2f}**，低於 1，代表股價尚未完全反映成長潛力，值得關注。")
        elif peg < 1.5:
            lines.append(f"⚠️ PEG = **{peg:.2f}**，略偏高。我會希望再等等，看能否在更低價格買到。")
        else:
            lines.append(f"❌ PEG = **{peg:.2f}**，成長速度已跟不上股價，我不會在這個位置進場。")
    else:
        lines.append("⚠️ PEG 無法計算（缺少 EPS 成長數據），這讓我很難判斷合理價值。先去找財報確認。")

    if eps_g is not None:
        if eps_g > 0.2:
            lines.append(f"📈 EPS 年成長 **{eps_g*100:.0f}%**，這種成長動能如果能維持 3～5 年，股價必然反映。")
        elif eps_g > 0.1:
            lines.append(f"📊 EPS 年成長 **{eps_g*100:.0f}%**，穩健但不算亮眼，適合「穩定成長股」邏輯。")
        elif eps_g < 0:
            lines.append(f"⚠️ EPS 負成長 **{eps_g*100:.0f}%**，先搞清楚是一次性因素還是長期惡化。")

    if debt_eq is not None:
        if debt_eq < 25:
            lines.append(f"💪 負債比只有 **{debt_eq:.0f}%**，財務非常穩健。景氣轉差也撐得住。")
        elif debt_eq > 100:
            lines.append(f"❗ 負債比高達 **{debt_eq:.0f}%**，這讓我擔心。利率上升或景氣收縮時，高負債公司最危險。")

    if cash and debt and debt > 0:
        ratio = cash / debt
        if ratio >= 1:
            lines.append(f"✅ 現金足以覆蓋所有負債（{ratio:.0%}），這是我愛看到的「現金防護網」。")

    if cat_key == 'fast':
        lines.append("🚀 這是一支快速成長股——林區最愛的類型。只要成長持續、PEG 合理，我願意長期持有。")
    elif cat_key == 'slow':
        lines.append("🐢 緩慢成長股的重點是股息，而不是資本利得。如果股息穩定、負債低，也是不錯的選擇。")
    elif cat_key == 'cyclical':
        lines.append("🔄 景氣循環股最重要的不是 PEG，而是**買在哪個景氣位置**。低谷才是進場時機。")
    elif cat_key == 'turnaround':
        lines.append("🔁 轉機股的關鍵問題：**為什麼它會好轉？** 搞清楚復甦原因，才能確認這不是陷阱。")

    if not lines:
        lines.append("⚠️ 基本面數據不足，無法做完整評估。林區說：「不了解就不要買。」先去看財報！")

    return lines

# ══════════════════════════════════════════════════
# 4. UI — 頁首
# ══════════════════════════════════════════════════

st.title("📚 彼得林區台股選股系統")
st.markdown(
    "> *「真正傷害投資人的，不是市場下跌，而是在錯誤的股票上等待。"
    "找到你了解的好公司，在合理價格買進，然後持有。」— 彼得林區*"
)
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 個股林區評分", "💼 持股體檢"])

# ══════════════════════════════════════════════════
# 5. Tab 1：個股評分
# ══════════════════════════════════════════════════

with tab1:
    col_inp, col_btn = st.columns([3, 1])
    with col_inp:
        code_raw = st.text_input(
            "股票代號（台股）:",
            value="2330",
            placeholder="例：2330、2886、00878、0050",
            label_visibility="collapsed"
        )
    with col_btn:
        btn1 = st.button("🔍 林區分析", type="primary", use_container_width=True)

    if btn1:
        code_base = code_raw.strip().upper()
        code_full = normalize(code_raw)

        with st.spinner(f"分析 {code_full} 中..."):
            info  = fetch_info(code_full)
            price = fetch_price(code_full)

        if not info and price is None:
            st.error("❌ 無法取得資料，請確認股票代號是否正確（例：2330、00878）")
            st.stop()

        name    = (info.get('longName') or info.get('shortName') or code_base)
        cat_key = classify_lynch(info, code_base)
        cat_nm, cat_desc = LYNCH_CAT[cat_key]
        is_etf_ = (cat_key == 'etf')
        score, detail, peg = calc_score(info, code_base)
        etf_score, etf_detail = calc_etf_score(info) if is_etf_ else (None, {})

        if is_etf_:
            stars, grade_text = grade_etf(etf_score)
        else:
            stars, grade_text = grade(score) if score is not None else ('📦', 'ETF')

        st.subheader(f"{name}　{code_base}")
        c_cat, c_act = st.columns([1, 2])
        with c_cat:
            st.info(f"**{cat_nm}**\n\n{cat_desc}")
        with c_act:
            if is_etf_:
                st.metric("ETF 綜合評分", f"{etf_score} / 100", grade_text)
            elif score is not None:
                st.metric("林區評分", f"{score} / 100", grade_text)

        st.markdown("---")

        pe    = info.get('trailingPE')
        eps_g = info.get('earningsGrowth')
        rev_g = info.get('revenueGrowth')
        div_y = normalize_yield(info.get('dividendYield'))

        if is_etf_:
            exp_r   = info.get('annualReportExpenseRatio') or info.get('expenseRatio')
            ret_3yr = info.get('threeYearAverageReturn')
            ret_1yr = info.get('oneYearTotalReturn') or info.get('ytdReturn')
            assets  = info.get('totalAssets')
            def fmt_aum2(v):
                if not v: return 'N/A'
                if v >= 1e8: return f"{v/1e8:.0f}億"
                return f"{v/1e4:.0f}萬"
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("目前股價",   f"${fmt_price(price)}" if price else 'N/A')
            m2.metric("股息殖利率", f"{div_y*100:.2f}%" if div_y else 'N/A')
            m3.metric("費用率",     f"{exp_r*100:.3f}%" if exp_r else 'N/A')
            m4.metric("3年平均報酬",f"{ret_3yr*100:.1f}%" if ret_3yr else 'N/A')
            m5.metric("近1年報酬",  f"{ret_1yr*100:.1f}%" if ret_1yr else 'N/A')
            m6.metric("資產規模",   fmt_aum2(assets))
        else:
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("目前股價",   f"${fmt_price(price)}" if price else 'N/A')
            m2.metric("本益比 P/E", f"{pe:.1f}" if pe else 'N/A')
            m3.metric("EPS 成長",   f"{eps_g*100:.1f}%" if eps_g else 'N/A')
            m4.metric("PEG 比率",   f"{peg:.2f}" if peg else 'N/A',
                      delta="✅ 低估" if (peg and peg < 1) else ("⚠️ 偏高" if (peg and peg > 1.5) else None),
                      delta_color="normal" if (peg and peg < 1) else "inverse")
            m5.metric("營收成長",   f"{rev_g*100:.1f}%" if rev_g else 'N/A')
            m6.metric("股息殖利率", f"{div_y*100:.2f}%" if div_y else 'N/A')

        def render_score_detail(det, radar_color, fill_color, title):
            st.markdown(f"### 📊 {title}")
            cats_r = list(det.keys())
            vals_r = [det[d]['score'] / det[d]['max'] * 100 for d in det]
            fig_r = go.Figure(go.Scatterpolar(
                r=vals_r + [vals_r[0]],
                theta=cats_r + [cats_r[0]],
                fill='toself',
                fillcolor=fill_color,
                line=dict(color=radar_color, width=2),
            ))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                    angularaxis=dict(tickfont=dict(size=12))
                ),
                template='plotly_dark',
                showlegend=False,
                height=350,
                margin=dict(t=30, b=30, l=50, r=50)
            )
            st.plotly_chart(fig_r, use_container_width=True)
            for metric, d in det.items():
                colA, colB, colC, colD = st.columns([2.5, 4, 1, 1.5])
                with colA:
                    st.write(f"**{metric}**")
                    st.caption(d['note'])
                with colB:
                    pct = d['score'] / d['max'] if d['max'] else 0
                    emoji = "🟢" if pct >= 0.7 else ("🟡" if pct >= 0.4 else "🔴")
                    st.progress(pct, text=emoji)
                with colC:
                    st.write(f"**{d['score']}/{d['max']}**")
                with colD:
                    st.write(d['val'])

        if is_etf_:
            render_score_detail(etf_detail, '#60A5FA', 'rgba(96,165,250,0.15)', 'ETF 評分明細')
            st.markdown(f"### 💡 建議：{action(None)}")
            st.markdown("---")
            st.markdown("### 💬 如果彼得林區持有這檔 ETF，他會說⋯")
            for line in lynch_etf_voice(info, etf_score, etf_detail):
                st.write(line)
        elif score is not None:
            render_score_detail(detail, '#F59E0B', 'rgba(255,215,0,0.15)', '林區評分明細')
            st.markdown(f"### 💡 建議：{action(score)}")
            st.markdown("---")
            st.markdown("### 💬 如果彼得林區持有這支股票，他會說⋯")
            for line in lynch_voice(info, score, peg, cat_key):
                st.write(line)

# ══════════════════════════════════════════════════
# 6. Tab 2：持股體檢
# ══════════════════════════════════════════════════

with tab2:
    st.subheader("💼 持股體檢")
    st.caption("輸入你的持股，系統用彼得林區標準逐一評估，給出留 / 觀察 / 減碼建議。")

    with st.expander("📝 輸入持股清單", expanded=True):
        st.caption("格式：股票代號, 持股張數, 平均成本（每行一支，成本可不填）")
        default_txt = (
            "2330, 1, 900\n"
            "0056, 5, 35\n"
            "00878, 10, 20\n"
            "2886, 2, 42\n"
        )
        port_text = st.text_area("持股清單:", value=default_txt, height=160, label_visibility="collapsed")
        btn2 = st.button("🔍 開始體檢", type="primary")

    if btn2:
        holdings = []
        for line in port_text.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if not parts or not parts[0]:
                continue
            code_r  = parts[0].strip()
            boards  = float(parts[1].replace(',', '')) if len(parts) > 1 and parts[1].strip() else 1
            cost_pp = float(parts[2].replace(',', '')) if len(parts) > 2 and parts[2].strip() else None
            holdings.append({'code': code_r, 'boards': boards, 'cost': cost_pp})

        if not holdings:
            st.error("請輸入至少一支持股")
            st.stop()

        progress_bar = st.progress(0, text="體檢中...")

        def check_one(h):
            code_b = h['code'].strip().upper()
            code_f = normalize(h['code'])
            info_  = fetch_info(code_f)
            price_ = fetch_price(code_f)

            name_ = info_.get('longName') or info_.get('shortName') or code_b
            if len(name_) > 12:
                name_ = name_[:10] + '…'

            cat_k    = classify_lynch(info_, code_b)
            cat_n, _ = LYNCH_CAT[cat_k]
            is_etf_h = (cat_k == 'etf')
            sc, det, pg = calc_score(info_, code_b)

            if is_etf_h:
                etf_sc, _ = calc_etf_score(info_)
                st_stars, _ = grade_etf(etf_sc)
                sc = etf_sc
            else:
                st_stars, _ = grade(sc) if sc is not None else ('⭐', '')

            shares_count = h['boards'] * 1000
            cost_total   = (h['cost'] * shares_count) if h['cost'] else None
            val_total    = (price_ * shares_count) if price_ else None
            pnl_         = (val_total - cost_total) if (val_total and cost_total) else None
            pnl_pct_     = ((price_ - h['cost']) / h['cost'] * 100) if (price_ and h['cost']) else None

            pe_    = info_.get('trailingPE')
            eps_g_ = info_.get('earningsGrowth')

            return {
                '代號':       code_b,
                '名稱':       name_,
                '林區分類':   cat_n,
                '現價':       f"${fmt_price(price_)}" if price_ else 'N/A',
                '成本/股':    f"${h['cost']:,.0f}" if h['cost'] else '—',
                '損益%':      f"{pnl_pct_:+.1f}%" if pnl_pct_ is not None else '—',
                '未實現損益': f"${pnl_:+,.0f}" if pnl_ is not None else '—',
                'P/E':        f"{pe_:.1f}" if pe_ else 'N/A',
                'EPS成長':    f"{eps_g_*100:.0f}%" if eps_g_ else 'N/A',
                'PEG':        f"{pg:.2f}" if pg else 'N/A',
                '林區分數':   sc if sc is not None else 0,
                '評級':       st_stars,
                '林區建議':   action(sc, pnl_pct_),
                '_score':     sc or 0,
                '_cost':      cost_total or 0,
                '_val':       val_total or 0,
                '_pnl':       pnl_ or 0,
                '_pnl_pct':   pnl_pct_,
            }

        results = []
        done_count = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(check_one, h): h for h in holdings}
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    results.append(res)
                done_count += 1
                progress_bar.progress(done_count / len(holdings), text=f"體檢中... {done_count}/{len(holdings)}")

        progress_bar.empty()

        if not results:
            st.error("無法取得任何資料")
            st.stop()

        results.sort(key=lambda x: x['_score'], reverse=True)

        st.markdown("### 📋 持股評分總表")
        disp_cols = ['代號','名稱','林區分類','現價','成本/股','損益%','未實現損益',
                     'P/E','EPS成長','PEG','林區分數','評級','林區建議']
        df_show = pd.DataFrame(results)[disp_cols]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📊 投資組合總覽")

        total_cost  = sum(r['_cost'] for r in results if r['_cost'])
        total_val   = sum(r['_val']  for r in results if r['_val'])
        total_pnl   = total_val - total_cost if (total_val and total_cost) else None
        pnl_pct_all = total_pnl / total_cost * 100 if (total_pnl and total_cost) else None
        num_scores  = [r['_score'] for r in results if isinstance(r['_score'], (int, float)) and r['_score'] > 0]
        avg_score   = np.mean(num_scores) if num_scores else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("總投入成本",   f"${total_cost:,.0f}" if total_cost else '—')
        c2.metric("目前市值",     f"${total_val:,.0f}"  if total_val  else '—')
        c3.metric("未實現損益",
                  f"${total_pnl:+,.0f}" if total_pnl else '—',
                  f"{pnl_pct_all:+.1f}%" if pnl_pct_all else None)
        c4.metric("投組評分均分", f"{avg_score:.0f}/100" if avg_score else '—')

        st.markdown("---")
        st.markdown("### 💬 彼得林區看你的投資組合")

        gems    = [r for r in results if r['_score'] >= 65 and r['林區分類'] != LYNCH_CAT['etf'][0]]
        weak    = [r for r in results if r['_score'] < 50  and r['林區分類'] != LYNCH_CAT['etf'][0]]
        etfs_r  = [r for r in results if r['林區分類'] == LYNCH_CAT['etf'][0]]
        dip_opp = [r for r in results if r['_score'] >= 65
                   and r['_pnl_pct'] is not None and r['_pnl_pct'] < -8
                   and r['林區分類'] != LYNCH_CAT['etf'][0]]

        if gems:
            st.success(f"✅ **{', '.join(r['代號'] for r in gems)}** 基本面符合林區標準，是組合核心。繼續持有，有機會加碼。")
        if dip_opp:
            st.info(f"💎 **{', '.join(r['代號'] for r in dip_opp)}** 基本面佳但跌幅超過 8%。林區說：『好公司跌價只是給你更好的買進機會。』")
        if weak:
            st.warning(f"⚠️ **{', '.join(r['代號'] for r in weak)}** 基本面評分偏低。先問自己「我為什麼要持有它？」如果答不出來，就該考慮出場。")
        if etfs_r:
            st.info(f"📦 **{', '.join(r['代號'] for r in etfs_r)}** 是 ETF，適合定期定額長期持有，不要頻繁進出。")

        if total_val:
            st.markdown("---")
            st.markdown("### 🥧 持股市值分配")
            pie_labels = [r['代號'] for r in results if r['_val'] > 0]
            pie_values = [r['_val']  for r in results if r['_val'] > 0]
            pie_colors = []
            for r in results:
                if r['_val'] <= 0:
                    continue
                is_etf_r = (r['林區分類'] == LYNCH_CAT['etf'][0])
                if is_etf_r:
                    pie_colors.append('#60A5FA')
                elif r['_score'] >= 65:
                    pie_colors.append('#4ADE80')
                elif r['_score'] >= 50:
                    pie_colors.append('#FBBF24')
                else:
                    pie_colors.append('#F87171')

            fig_pie = go.Figure(go.Pie(
                labels=pie_labels,
                values=pie_values,
                marker=dict(colors=pie_colors, line=dict(color='#0B1629', width=2)),
                textinfo='label+percent',
                hovertemplate='%{label}<br>市值：$%{value:,.0f}<br>佔比：%{percent}<extra></extra>',
                hole=0.4
            ))
            fig_pie.update_layout(
                template='plotly_dark',
                height=380,
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(font=dict(size=12)),
                annotations=[dict(text='市值分配', x=0.5, y=0.5, font_size=14, showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("🟢 個股評分 ≥ 65　🟡 50–64　🔴 < 50　🔵 ETF")

        st.markdown("---")
        st.markdown("---")
        st.markdown(
            "> 💬 *「股票市場是把錢從急躁者的口袋，"
            "轉移到有耐心者的口袋。」— 彼得林區*"
        )
        st.markdown("**林區投資核心原則：**")
        st.write("1. **了解你買的每一支股票**：說得出公司如何賣錢，才有資格持有。")
        st.write("2. **PEG < 1 是最好的起點**：成長快、股價合理，才是林區的最愛。")
        st.write("3. **不要因為股價下跌就賣**：如果基本面沒壞，跨是加碼機會。")
        st.write("4. **長期持有才能讓複利發揮**：林區的 Magellan 基金年均報酬 29%，靠的是持有，不是頻繁交易。")
