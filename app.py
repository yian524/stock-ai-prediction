import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from snownlp import SnowNLP
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import os
import streamlit as st


# === 🔐 密碼保護功能開始 ===
def check_password():
    """回傳 True 代表登入成功，False 代表失敗"""
    
    # 1. 如果已經登入成功過，就直接放行
    if st.session_state.get("password_correct", False):
        return True

    # 2. 顯示輸入框
    st.header("🔒 請輸入存取密碼")
    password_input = st.text_input("Password", type="password")
    
    if st.button("登入"):
        # 這裡檢查密碼是否等於我們設定的 "my_friend_password"
        # (稍後會在 Secrets 設定真正的密碼)
        if password_input == st.secrets["PASSWORD"]:
            st.session_state["password_correct"] = True
            st.rerun() # 重新整理頁面以顯示內容
        else:
            st.error("❌ 密碼錯誤，請重新輸入")
            
    return False

# 如果密碼檢查沒通過，就直接停止執行下面的程式
if not check_password():
    st.stop()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 設定網頁基本資訊 ---
st.set_page_config(page_title="台股 AI 預測 (多模型平均版)", layout="wide")

st.title("🇹🇼 台股 AI 預測系統 (集成學習版)")
st.markdown("""
此版本採用 **集成學習 (Ensemble Learning)** 概念：
透過多次訓練模型並取 **平均值**，消除單次訓練的隨機誤差，提供更穩定的預測結果。
""")

# --- 側邊欄設定 ---
st.sidebar.header("設定參數")

stock_map = {
    "2330 台積電": "2330.TW",
    "2317 鴻海": "2317.TW",
    "2454 聯發科": "2454.TW",
    "2603 長榮": "2603.TW",
    "3231 緯創": "3231.TW",
    "2382 廣達": "2382.TW",
    "3008 大立光": "3008.TW",
    "自訂輸入": "CUSTOM"
}

selected_label = st.sidebar.selectbox("選擇股票", list(stock_map.keys()))

if selected_label == "自訂輸入":
    stock_ticker = st.sidebar.text_input("請輸入台股代碼 (需加 .TW)", "2330.TW")
    stock_id = stock_ticker.split(".")[0] 
    stock_name_for_ptt = stock_id 
else:
    stock_ticker = stock_map[selected_label]
    stock_id = stock_ticker.split(".")[0] 
    stock_name_for_ptt = selected_label.split(" ")[1] 

look_back = st.sidebar.slider("參考過去幾天 (Time Steps)", 10, 60, 30)
epochs = st.sidebar.slider("訓練次數 (Epochs)", 1, 30, 10)

# === 新增：讓使用者決定跑幾次 ===
ensemble_runs = st.sidebar.slider("預測平均次數 (建議 3~5 次)", 1, 10, 3)
st.sidebar.caption(f"注意：設定 {ensemble_runs} 次，訓練時間就會變成 {ensemble_runs} 倍。")

# --- 爬蟲函式 ---

def get_yahoo_news_sentiment(stock_id):
    url = f"https://tw.stock.yahoo.com/quote/{stock_id}.TW/news"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find_all('h3')
        scores, titles_list, seen_titles = [], [], set()
        
        count = 0
        for h in headlines:
            if count >= 8: break
            text = h.get_text().strip()
            if len(text) < 5 or text in seen_titles: continue
            seen_titles.add(text)
            s = SnowNLP(text)
            scores.append(s.sentiments)
            titles_list.append(f"[Yahoo] ({s.sentiments:.2f}) {text}")
            count += 1
            
        return (np.mean(scores), titles_list) if scores else (0.5, ["Yahoo: 未抓取到新聞"])
    except Exception as e:
        return 0.5, [f"Yahoo 錯誤: {e}"]

def get_ptt_sentiment(keyword):
    url = f"https://www.ptt.cc/bbs/Stock/search?q={keyword}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    cookies = {'over18': '1'}
    try:
        response = requests.get(url, headers=headers, cookies=cookies, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        titles_tags = soup.find_all('div', class_='title')
        scores, titles_list, seen_titles = [], [], set()
        
        count = 0
        for t in titles_tags:
            if count >= 5: break
            if t.find('a'):
                text = t.find('a').get_text().strip()
                if "已被刪除" in text or text in seen_titles: continue
                seen_titles.add(text)
                s = SnowNLP(text)
                scores.append(s.sentiments)
                titles_list.append(f"[PTT] ({s.sentiments:.2f}) {text}")
                count += 1
        
        return (np.mean(scores), titles_list) if scores else (0.5, ["PTT: 無結果"])
    except Exception as e:
        return 0.5, [f"PTT 錯誤: {e}"]

# --- 資料處理 ---

def preprocess_data(df, look_back):
    dataset = df['Close'].values.reshape(-1, 1)
    np.random.seed(42) # 這裡固定是為了讓"過去的假特徵"一致，不影響模型訓練的隨機性
    sentiment_history = np.random.uniform(0, 1, size=(len(dataset), 1))
    combined_data = np.hstack((dataset, sentiment_history))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_data)
    return scaled_data, scaler, dataset

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- 主程式 ---

st.subheader(f"📊 分析標的：{stock_ticker}")

with st.spinner('正在下載所有歷史資料 (可能需要幾秒鐘)...'):
    # ✅ 修改 1: 改成 "max" 抓取該股票上市以來的所有資料
    df = yf.download(stock_ticker, period="max")

if df is not None and not df.empty:
    
    # 資料清洗 (防呆機制)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs('Close', axis=1, level=0, drop_level=True)
        except KeyError:
            df = df.iloc[:, 3].to_frame()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.columns = ['Close']
    df = df.dropna()

    # ✅ 修改 2: 改用 Plotly 繪製專業互動圖表
    # 這會產生一個可以縮放、有滑桿的 K 線圖效果
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='收盤價'))
    fig.update_layout(
        title=f"{stock_ticker} 歷史股價走勢",
        xaxis_title="日期",
        yaxis_title="股價",
        xaxis_rangeslider_visible=True, # 開啟下方的時間拉桿
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 準備訓練資料
    # ✅ 優化: 雖然圖表秀 30 年，但訓練 AI 如果用 30 年會跑太久
    # 我們取 "最近 5 年" (約 1250 天) 來訓練就好，這樣準確度夠，速度也快
    training_limit = 1250 
    if len(df) > training_limit:
        df_for_training = df.iloc[-training_limit:]
    else:
        df_for_training = df
        
    scaled_data, scaler, raw_data = preprocess_data(df_for_training, look_back)
    
    # ... (下方的訓練與預測程式碼保持不變) ...
    # 注意：下面的 raw_data 變數是來自 df_for_training
    
    train_size = int(len(scaled_data) * 0.9)
    train_data = scaled_data[0:train_size, :]
    
    x_train, y_train = [], []
    for i in range(look_back, len(train_data)):
        x_train.append(train_data[i-look_back:i, :])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    if st.button(f'🚀 啟動多模型分析 (共執行 {ensemble_runs} 次)'):
        
        # 1. 爬蟲 (只做一次，節省時間)
        st.write("---")
        st.info("正在進行新聞與輿情分析...")
        yahoo_score, yahoo_titles = get_yahoo_news_sentiment(stock_id)
        ptt_score, ptt_titles = get_ptt_sentiment(stock_name_for_ptt)
        if "無結果" in ptt_titles[0]: ptt_score, ptt_titles = get_ptt_sentiment(stock_id)
        
        final_sentiment = (yahoo_score + ptt_score) / 2
        
        col1, col2 = st.columns(2)
        col1.metric("綜合情緒分數", f"{final_sentiment:.2f}")
        with col2.expander("查看新聞來源"):
            for t in yahoo_titles + ptt_titles: st.write(t)
            
        # 2. 多次訓練與預測 (Ensemble)
        st.write("---")
        st.subheader(f"🧠 正在訓練 {ensemble_runs} 個 AI 模型...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        prediction_list = []
        
        # 準備最後一天的輸入資料
        last_days = scaled_data[-look_back:].copy()
        last_days[-1, 1] = final_sentiment
        X_input = last_days.reshape(1, look_back, 2)
        
        # === 迴圈開始 ===
        for i in range(ensemble_runs):
            status_text.text(f"正在訓練第 {i+1} / {ensemble_runs} 個模型...")
            
            # 每次建立新模型，權重都會隨機初始化
            model = build_model((x_train.shape[1], x_train.shape[2]))
            
            # 訓練 (verbose=0 不顯示個別進度，以免洗版)
            model.fit(x_train, y_train, batch_size=16, epochs=epochs, verbose=0)
            
            # 預測
            pred_scaled = model.predict(X_input, verbose=0)
            
            # 反正規化
            temp = np.zeros((1, 2))
            temp[0, 0] = pred_scaled[0, 0]
            pred_price = scaler.inverse_transform(temp)[0][0]
            
            prediction_list.append(pred_price)
            
            # 更新進度條
            progress_bar.progress((i + 1) / ensemble_runs)
            
        # === 迴圈結束 ===
        
        status_text.text("所有模型訓練完成！")
        
        # 計算統計數據
        avg_price = np.mean(prediction_list)
        max_price = np.max(prediction_list)
        min_price = np.min(prediction_list)
        last_close = raw_data[-1][0]
        
        st.subheader("🔮 最終集成預測結果")
        
        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("昨日收盤價", f"{last_close:.2f}")
        r_col2.metric("AI 平均預測價", f"{avg_price:.2f}", delta=f"{avg_price - last_close:.2f}")
        r_col3.metric("預測區間 (最高/最低)", f"{max_price:.1f} ~ {min_price:.1f}")
        
        st.write(f"個別模型預測值： {[round(p, 1) for p in prediction_list]}")
        
        if final_sentiment > 0.6 and (avg_price - last_close) > 0:
            st.success("結論：多模型一致看好，情緒樂觀 🚀")
        elif final_sentiment < 0.4 and (avg_price - last_close) < 0:
            st.error("結論：多模型一致看跌，情緒保守 📉")
        else:
            st.info("結論：模型意見分歧或與情緒面不一致，建議區間操作 ⚖️")

else:
    st.error("無法取得資料。")