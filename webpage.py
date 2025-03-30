import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from streamlit_autorefresh import st_autorefresh
import pytz
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

def get_binance_klines(symbol='BTCUSDT', interval='1m', start_time='2025-01-24 00:00:00',
                       end_time='2025-03-24 00:00:00', limit=1000):
    """
    获取 Binance 交易对的历史 K 线数据，支持选择开始和结束时间。
    参数:
    - symbol: 交易对（例如 BTCUSDT）
    - interval: 时间间隔（例如 '1m', '5m', '1h', '1d'）
    - start_time: 开始时间（字符串，格式为 'YYYY-MM-DD HH:MM:SS' 或 datetime 对象）
    - end_time: 结束时间（字符串，格式为 'YYYY-MM-DD HH:MM:SS' 或 datetime 对象）
    - limit: 返回数据的数量限制（最多 1000）
    """
    tz = pytz.timezone('Asia/Hong_Kong')

    # 统一处理时间输入
    def parse_time(time_input, default_delta=None):
        if time_input is None:
            return datetime.now(tz) if default_delta is None else datetime.now(tz) - default_delta
        if isinstance(time_input, (int, float)):
            return datetime.fromtimestamp(time_input / 1000, tz=pytz.utc).astimezone(tz)
        if isinstance(time_input, str):
            time_input = datetime.strptime(time_input, '%Y-%m-%d %H:%M:%S')
        if time_input.tzinfo is None:
            return tz.localize(time_input)
        return time_input.astimezone(tz)

    end_time = parse_time(end_time)
    start_time = parse_time(start_time, relativedelta(months=3))

    # 转换为UTC时间戳（毫秒）
    start_timestamp = int(start_time.astimezone(pytz.utc).timestamp() * 1000)
    end_timestamp = int(end_time.astimezone(pytz.utc).timestamp() * 1000)

    # 获取数据
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_timestamp

    while current_start < end_timestamp:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_timestamp,
            'limit': limit
        }
        data = requests.get(url, params=params).json()
        if not data:
            break
        all_data.extend(data)
        current_start = data[-1][6] + 1  # 更新起始时间

    # 创建DataFrame
    columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
               'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Volume',
               'Taker Buy Quote Volume', 'Ignore']
    df = pd.DataFrame(all_data, columns=columns)

    # 处理时间和数值类型
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Hong_Kong')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Hong_Kong')
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume',
                    'Taker Buy Base Volume', 'Taker Buy Quote Volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time']]

# ==== Timezone Settings ====
HK_TZ = pytz.timezone('Asia/Hong_Kong')
UTC_TZ = pytz.utc


def convert_date_to_timestamp(date_obj, end_of_day=False):
    dt = HK_TZ.localize(datetime(
        date_obj.year, date_obj.month, date_obj.day,
        hour=23 if end_of_day else 0,
        minute=59 if end_of_day else 0,
        second=59 if end_of_day else 0
    ))
    utc_dt = dt.astimezone(UTC_TZ)
    return int(utc_dt.timestamp() * 1000)


# ==== Page Configuration ====
st.set_page_config(page_title="Crypto Real-time Dashboard", layout="wide", page_icon="₿")

st.markdown("""
    <style>
    .stMetric { text-align: center; }
    .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)


# ==== Real-time Price ====
BINANCE_API_URL = "https://api.binance.com/api/v3"

@st.cache_data(ttl=10)
def fetch_realtime_price(symbol='BTCUSDT'):
    try:
        response = requests.get(f"{BINANCE_API_URL}/ticker/price", params={'symbol': symbol})
        response.raise_for_status()
        data = response.json()
        return {
            'price': float(data['price']),
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.error(f"Failed to fetch real-time price: {str(e)}")
        st.stop()


# ==== Main Function ====
def realtime_analytics():
    st.title("📈 Crypto Real-time Analytics Dashboard")
    st_autorefresh(interval=60 * 1000, key="auto_refresh")

    # ==== Sidebar ====
    with st.sidebar:
        st.header("Control Panel")
        selected_symbol = st.selectbox("Trading Pair", [
            'BTCUSDT', 'ETHUSDT', 'XRPUSDT',
            'BNBUSDT', 'SOLUSDT', 'ADAUSDT'
        ])

        selected_interval = st.selectbox("K-line Interval",
                                         ["1m", "5m", "15m", "1h", "4h", "1d"], index=4)

        current_hk_time = datetime.now(HK_TZ)
        today_date = current_hk_time.date()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date",
                                       value=current_hk_time - timedelta(days=60),
                                       min_value=current_hk_time - timedelta(days=8*365),
                                       max_value=today_date)
        with col2:
            end_date = st.date_input("End Date",
                                     value=today_date,
                                     min_value=current_hk_time - timedelta(days=8*365),
                                     max_value=today_date)

        start_time = convert_date_to_timestamp(start_date)
        end_time = convert_date_to_timestamp(end_date, end_of_day=True)

    # ==== Data Fetching ====
    hist_data = get_binance_klines(
        symbol=selected_symbol,
        interval=selected_interval,
        start_time=start_time,
        end_time=end_time,
        limit=1000
    )

    realtime_data = fetch_realtime_price(symbol=selected_symbol)

    # ==== Real-time Metrics ====
    with st.container():
        col1, col2, col3 = st.columns(3)
        if len(hist_data) >= 2:
            delta = hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]
            delta_pct = (delta / hist_data['Close'].iloc[-2]) * 100
        else:
            delta = 0
            delta_pct = 0
        with col1:
            st.metric("Current Price", f"${realtime_data['price']:,.2f}", f"{delta:+.2f} ({delta_pct:+.2f}%)")
        with col2:
            st.metric("Recent High", f"${hist_data['Close'].max():,.2f}")
        with col3:
            st.metric("Recent Low", f"${hist_data['Close'].min():,.2f}")

    st.markdown("---")

    # ==== Price & Volume Chart ====
    fig = go.Figure()

    # Price Line
    fig.add_trace(go.Scatter(
        x=hist_data['Close Time'],
        y=hist_data['Close'].round(2),
        name='Price',
        line=dict(color='#1f77b4', width=2),
        yaxis='y2',
        showlegend = False
    ))

    # Volume Bar
    fig.add_trace(go.Bar(
        x=hist_data['Close Time'],
        y=hist_data['Volume'],
        name='Volume',
        marker={'color' : '#19D3F3', 'line_width':0.15},
        yaxis='y',
        showlegend = False
    ))

    # Real-time Price Point
    fig.add_trace(go.Scatter(
        x=[realtime_data['timestamp']],
        y=[realtime_data['price']],
        mode='markers+text',
        text=["Live Price"],
        textposition="top center",
        marker=dict(color='red', size=10),
        name='Live Price',
        yaxis='y2'
    ))

    # Layout
    fig.update_layout(
        title={
            'text': f"{selected_symbol} Price & Volume Analysis",
            'x': 0.5, 'xanchor': 'center', 'font': dict(size=20)
        },
        xaxis=dict(autorange=True,
                   title_text="Date",
                   rangeslider=dict(visible=True),
                   showgrid=True,
                   type="date"
                   ),
        yaxis=dict(
            anchor="x",
            autorange=True,
            domain=[0, 0.3],
            # linecolor="#607d8b",
            mirror=True,
            showline=True,
            side="left",
            # tickfont={"color": "#607d8b"},
            tickmode="auto",
            ticks="",
            title="Volume",
            # titlefont={"color": "#607d8b"},
            type="linear",
            zeroline=False,
        ),
        yaxis2=dict(
            anchor="x",
            autorange=True,
            domain=[0.3, 1],
            # linecolor="#6600FF",
            mirror=True,
            showline=True,
            side="left",
            tickfont={"color": "#6600FF"},
            tickmode="auto",
            ticks="",
            title="Price",
            titlefont={"color": "#6600FF"},
            type="linear",
            zeroline=False,
            fixedrange=False
        ),
        hovermode="x unified",
        height=600,  # 增加整体图表高度
        margin=dict(t=40, b=20, l=40, r=40),  # 减少上下边距
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


    # ==== Footer ====
    st.caption("Data Source: Binance Public API.")
    st.caption("Chart refresh rate: Every 60 seconds.")


def model_predictions():
    """Model Predictions Page with LightGBM and CatBoost Comparison"""
    st.title("🤖 AI Trading Model Comparison")
    PREDICTION_MULTIPLIER = 10
    # 创建Asset_ID到名称的映射字典
    asset_mapping = {
        0: "Bitcoin Cash",
        1: "Binance Coin",
        2: "Bitcoin",
        3: "Cardano",
        4: "Dogecoin",
        5: "EOS.IO",
        6: "Ethereum",
        7: "Ethereum Classic",
        8: "IOTA",
        9: "Litecoin",
        10: "Maker",
        11: "Monero",
        12: "Stellar",
        13: "TRON"
    }

    # 侧边栏控件
    with st.sidebar:
        st.header("Model Configuration")

        # 选择加密货币（添加唯一key）
        selected_id = st.selectbox(
            "Select Cryptocurrency",
            options=list(asset_mapping.keys()),
            format_func=lambda x: f"{asset_mapping[x]} (ID: {x})",
            index=2,
            key="crypto_selectbox"  # Unique key
        )

        # 选择数据间隔（添加唯一key）
        interval = st.selectbox(
            "Data Interval",
            options=["1m", "5m", "15m", "1h"],
            index=0,
            key="interval_selectbox"  # Unique key
        )

        # 选择要显示的模型（添加唯一key）
        selected_models = st.multiselect(
            "Select Models to Compare",
            options=["LightGBM", "CatBoost"],
            default=["LightGBM", "CatBoost"],
            key="model_multiselect"  # Unique key
        )

        st.markdown("---")
        st.subheader("Time Range Selection")

    # 读取数据函数
    @st.cache_data
    def load_data(asset_id, model_type):
        # 获取当前脚本所在的目录
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DATA_DIR = os.path.join(BASE_DIR, "model_data")
        
        file_path = os.path.join(MODEL_DATA_DIR, f"{model_type}/Asset_ID_{asset_id}_{model_type}.csv.gz")
        df = pd.read_csv(file_path, compression='gzip')
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    # 加载数据
    data = {}
    if "LightGBM" in selected_models:
        lgbm_df = load_data(selected_id, "LightGBM")
        data["LightGBM"] = lgbm_df
    if "CatBoost" in selected_models:
        cat_df = load_data(selected_id, "CatBoost")
        data["CatBoost"] = cat_df

    # 获取时间范围
    all_dfs = list(data.values())
    full_df = pd.concat(all_dfs) if all_dfs else pd.DataFrame()

    if not full_df.empty:
        # 设置默认时间范围（开始时间后的1个月）
        default_start = full_df['datetime'].min()
        default_end = min(default_start + pd.DateOffset(months=1), full_df['datetime'].max())

        # 在侧边栏添加日期选择器
        with st.sidebar:
            start_date = st.date_input(
                "Start date",
                value=default_start.date(),
                min_value=full_df['datetime'].min().date(),
                max_value=full_df['datetime'].max().date(),
                key="start_date_input"  # Unique key
            )

            end_date = st.date_input(
                "End date",
                value=default_end.date(),
                min_value=full_df['datetime'].min().date(),
                max_value=full_df['datetime'].max().date(),
                key="end_date_input"  # Unique key
            )

        # 转换日期为datetime
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # 包含结束日

        # 处理每个模型的数据
        processed_data = {}
        for model_name, df in data.items():
            # 筛选时间范围内的数据
            temp_df = df[(df['datetime'] >= start_datetime) & (df['datetime'] <= end_datetime)].copy()

            # 根据间隔重新采样数据
            if interval != "1m":
                interval_map = {
                    "5m": "5T",
                    "15m": "15T",
                    "1h": "1H"
                }
                temp_df = temp_df.set_index('datetime').resample(interval_map[interval]).mean().reset_index()

            processed_data[model_name] = temp_df

        # --- 使用 Plotly Go 创建交互式图表 ---
        fig = go.Figure()

        # 添加实际目标曲线（只添加一次）
        fig = go.Figure()
        if processed_data:
            first_df = list(processed_data.values())[0]
            target_col = 'target' if 'target' in first_df.columns else 'Target'
            fig.add_trace(go.Scatter(
                x=first_df['datetime'],
                y=first_df[target_col],
                name='Actual Target',
                line=dict(color='blue', width=2),
                opacity=0.8
            ))

        colors = {'LightGBM': 'red', 'CatBoost': 'green'}
        for model_name, df in processed_data.items():
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df['final_predictions'] * PREDICTION_MULTIPLIER,  # 关键修改点
                name=f'{model_name} Predictions (x{PREDICTION_MULTIPLIER})',
                line=dict(color=colors.get(model_name, 'gray'), width=1.5),
                opacity=0.7
            ))

        # 更新图表布局
        fig.update_layout(
            title=f'{asset_mapping[selected_id]} (ID: {selected_id}) - Model Comparison | Interval: {interval}',
            xaxis_title='Datetime',
            yaxis_title='Value',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=600,
            template='plotly_white'
        )

        # --- 在 Streamlit 中显示 ---
        st.plotly_chart(fig, use_container_width=True)

        # 添加模型性能指标比较
        st.subheader("📈 Model Performance Comparison")

        if len(processed_data) > 1:
            metrics = {}
            for model_name, df in processed_data.items():
                target_col = 'target' if 'target' in df.columns else 'Target'
                pred_col = 'final_predictions'
                if target_col in df.columns and pred_col in df.columns:
                    adjusted_predictions = df[pred_col] * PREDICTION_MULTIPLIER  # 关键修改点
                    error = df[target_col] - adjusted_predictions
                    metrics[model_name] = {
                        'MAE': round(abs(error).mean(), 6),
                        'RMSE': round((error ** 2).mean() ** 0.5, 6),
                        'Correlation': round(df[target_col].corr(adjusted_predictions), 4)
                    }

            # 显示指标表格
            metrics_df = pd.DataFrame(metrics).T
            st.dataframe(metrics_df.style.highlight_min(axis=0, color='lightgreen'))

            # 添加数据统计摘要
            st.subheader("📊 Data Summary")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Cryptocurrency", asset_mapping[selected_id])
            with cols[1]:
                st.metric("Data Interval", interval)

            # # 显示原始数据（可选）
            # if st.checkbox("Show raw data", key="show_raw_data_checkbox"):  # Unique key
            #     selected_model = st.radio(
            #         "Select model to view",
            #         list(processed_data.keys()),
            #         key="model_radio"  # Unique key
            #     )
            #     st.dataframe(processed_data[selected_model])

        else:
            st.warning("No data available for the selected models.")


def crypto_news_reader():
    st.title("📰 Cryptocurrency News Reader")
    st.write("Get the latest news about your favorite cryptocurrencies from Cryptopanic")

    # Sidebar for user inputs
    with st.sidebar:
        st.subheader("Settings")
        selected_currencies = st.multiselect(
            "Select cryptocurrencies",
            options=["BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "DOT", "MATIC", "MC", "BNB"],
            default=["BTC", "MC"]
        )
        items_per_page = st.slider("News per page", 5, 20, 10)
        filter_option = st.selectbox("Filter by", ["rising 🚀", "hot 🔥", "bullish 📈"], index=0)

    if not selected_currencies:
        st.warning("Please select at least one cryptocurrency")
        return

    # 使用直接API_KEY（注意：实际开发中建议使用st.secrets）
    API_KEY = '8f8cf342f37496f2feea9e0daeccdae63c20df77'

    # 构造API URL（使用新格式）
    url = f"https://cryptopanic.com/api/v1/posts/"
    params = {
        'auth_token': API_KEY,
        'currencies': ",".join(selected_currencies),
        'kind': 'news',
        'public': 'true',
        'filter': filter_option  # 使用用户选择的过滤选项
    }

    # Fetch data with loading indicator
    with st.spinner("Fetching latest crypto news..."):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get('results'):
                st.info("No news found for the selected currencies")
                return

            # Display news items
            st.subheader(f"Latest {filter_option.capitalize()} News ")

            for i, news_item in enumerate(data['results'][:items_per_page]):
                with st.container():
                    col1, col2 = st.columns([0.8, 0.2])

                    with col1:
                        st.markdown(f"### {news_item['title']}")
                        st.caption(f"Source: {news_item['source']['title']} | Published: {news_item['published_at']}")
                        if news_item.get('description'):
                            st.write(news_item['description'])

                    with col2:
                        # 改进的情感分析显示
                        if news_item.get('votes'):
                            positive = news_item['votes']['positive']
                            negative = news_item['votes']['negative']
                            sentiment = positive - negative

                            if sentiment > 0:
                                st.success(f"👍 {positive}")
                            elif sentiment < 0:
                                st.error(f"👎 {abs(negative)}")
                            else:
                                st.info("🤝 Neutral")

                        # 获取原始文章链接（优先显示原始页面，而不是Cryptopanic中转页）
                        original_url = news_item.get("metadata", {}).get("original_url", news_item["url"])
                        st.markdown(f"[Read full article →]({original_url})")

                if i < items_per_page - 1:  # 避免最后多余的分隔线
                    st.divider()

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch news: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


# ==== Main App ====
def main():
    # Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select Feature",
            ["Real-time Market", "Model Analysis", "Market News"],
            index=0
        )

    # Page routing
    if page == "Real-time Market":
        realtime_analytics()
    elif page == "Model Analysis":
        model_predictions()
    elif page == "Market News":
        crypto_news_reader()


if __name__ == "__main__":
    main()
