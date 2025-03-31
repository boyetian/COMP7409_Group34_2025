import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import os
import pytz
import yfinance as yf
import requests
import pandas as pd

# ==== Timezone Settings ====
HK_TZ = pytz.timezone('Asia/Hong_Kong')
UTC_TZ = pytz.utc


# ==== Yahoo Finance Functions ====
def get_yfinance_data(symbol='BTC-USD', interval='1h', start_date=None, end_date=None):
    """
    获取 Yahoo Finance 的历史数据
    参数:
    - symbol: 交易对（例如 BTC-USD）
    - interval: 时间间隔 ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    - start_date: 开始日期 (datetime.date 或 datetime.datetime)
    - end_date: 结束日期 (datetime.date 或 datetime.datetime)
    """
    # 转换时间格式
    if start_date is None:
        start_date = datetime.now() - timedelta(days=60)
    if end_date is None:
        end_date = datetime.now()

    # 下载数据
    ticker = yf.Ticker(symbol)
    data = ticker.history(
        interval=interval,
        start=start_date,
        end=end_date,
        auto_adjust=False
    )

    # 处理数据格式
    if data.empty:
        return pd.DataFrame()

    data = data.reset_index()

    # Check if the datetime column is named 'Date' or 'Datetime'
    datetime_col = 'Datetime' if 'Datetime' in data.columns else 'Date'

    # Convert to datetime and handle timezone
    if pd.api.types.is_datetime64tz_dtype(data[datetime_col]):
        # Already timezone-aware, just convert to HK time
        data['Open Time'] = pd.to_datetime(data[datetime_col]).dt.tz_convert('Asia/Hong_Kong')
    else:
        # Not timezone-aware, localize to UTC then convert to HK time
        data['Open Time'] = pd.to_datetime(data[datetime_col]).dt.tz_localize('UTC').dt.tz_convert('Asia/Hong_Kong')

    data['Close Time'] = data['Open Time']

    # 重命名列以匹配原代码
    data = data.rename(columns={
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    })

    return data[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time']]


def fetch_realtime_price(symbol='BTC-USD'):
    """
    获取 Yahoo Finance 的实时价格
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='1m')
        if data.empty:
            return None

        latest = data.iloc[-1]
        return {
            'price': latest['Close'],
            'timestamp': datetime.now(HK_TZ)
        }
    except Exception as e:
        st.error(f"Failed to fetch real-time price: {str(e)}")
        return None


# ==== Helper Functions ====
def convert_date_to_timestamp(date_obj, end_of_day=False):
    """
    转换日期对象为时间戳
    """
    if isinstance(date_obj, datetime):
        dt = date_obj
    else:
        dt = datetime.combine(date_obj, datetime.min.time())

    dt = HK_TZ.localize(dt) if dt.tzinfo is None else dt.astimezone(HK_TZ)

    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    else:
        dt = dt.replace(hour=0, minute=0, second=0)

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


# ==== Main Function ====
def realtime_analytics():
    st.title("📈 Crypto Real-time Analytics Dashboard")

    # ==== Sidebar ====
    with st.sidebar:
        st.header("Control Panel")
        selected_symbol = st.selectbox("Trading Pair", [
            'BTC-USD', 'ETH-USD', 'XRP-USD',
            'BNB-USD', 'SOL-USD', 'ADA-USD'
        ])

        selected_interval = st.selectbox("K-line Interval",
                                         ["1m", "5m", "15m", "30m", "60m", "1d"],
                                         index=4)

        current_hk_time = datetime.now(HK_TZ)
        today_date = current_hk_time.date()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date",
                                       value=current_hk_time - timedelta(days=60),
                                       min_value=current_hk_time - timedelta(days=8 * 365),
                                       max_value=today_date)
        with col2:
            end_date = st.date_input("End Date",
                                     value=today_date,
                                     min_value=current_hk_time - timedelta(days=8 * 365),
                                     max_value=today_date)

    # ==== Data Fetching ====
    hist_data = get_yfinance_data(
        symbol=selected_symbol,
        interval=selected_interval,
        start_date=start_date,
        end_date=end_date
    )

    realtime_data = fetch_realtime_price(symbol=selected_symbol)

    # ==== Real-time Metrics ====
    with st.container():
        col1, col2, col3 = st.columns(3)
        if len(hist_data) >= 2 and realtime_data:
            delta = realtime_data['price'] - hist_data['Close'].iloc[-1]
            delta_pct = (delta / hist_data['Close'].iloc[-1]) * 100
        else:
            delta = 0
            delta_pct = 0

        with col1:
            st.metric("Current Price",
                      f"${realtime_data['price']:,.2f}" if realtime_data else "N/A",
                      f"{delta:+.2f} ({delta_pct:+.2f}%)")
        with col2:
            st.metric("Recent High", f"${hist_data['Close'].max():,.2f}" if not hist_data.empty else "N/A")
        with col3:
            st.metric("Recent Low", f"${hist_data['Close'].min():,.2f}" if not hist_data.empty else "N/A")

    st.markdown("---")

    # ==== Price & Volume Chart ====
    if not hist_data.empty and realtime_data:
        fig = go.Figure()

        # Price Line
        fig.add_trace(go.Scatter(
            x=hist_data['Close Time'],
            y=hist_data['Close'].round(2),
            name='Price',
            line=dict(color='#1f77b4', width=2),
            yaxis='y2',
            showlegend=False
        ))

        # Volume Bar
        fig.add_trace(go.Bar(
            x=hist_data['Close Time'],
            y=hist_data['Volume'],
            name='Volume',
            marker={'color': '#19D3F3', 'line_width': 0.15},
            yaxis='y',
            showlegend=False
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
            xaxis=dict(
                autorange=True,
                title_text="Date",
                rangeslider=dict(visible=True),
                showgrid=True,
                type="date"
            ),
            yaxis=dict(
                anchor="x",
                autorange=True,
                domain=[0, 0.3],
                mirror=True,
                showline=True,
                side="left",
                tickmode="auto",
                ticks="",
                title="Volume",
                type="linear",
                zeroline=False,
            ),
            yaxis2=dict(
                anchor="x",
                autorange=True,
                domain=[0.3, 1],
                mirror=True,
                showline=True,
                side="left",
                tickfont={"color": "#6600FF"},
                title="Price",
                titlefont={"color": "#6600FF"},
                type="linear",
                zeroline=False
            ),
            hovermode="x unified",
            height=600,
            margin=dict(t=40, b=20, l=40, r=40),
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected time range")

    # ==== Footer ====
    st.caption("Data Source: Yahoo Finance")
    st.caption("Chart refresh rate: Every 60 seconds")


def model_predictions():
    """Model Predictions Page with LightGBM"""
    st.title("🤖 AI Trading Model Comparison")
    # 初始化路径（兼容本地和云端）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DATA_DIR = os.path.join(BASE_DIR, "model_data")

    # 如果不存在，尝试临时目录
    if not os.path.exists(MODEL_DATA_DIR):
        MODEL_DATA_DIR = os.path.join("/tmp", "model_data")

    # 确保目录存在
    os.makedirs(MODEL_DATA_DIR, exist_ok=True)

    FEATURE_IMPORTANCE_DIR = os.path.join(MODEL_DATA_DIR, "Feature Importance Pictures")

    # Add calculation mode selection
    calculation_mode = st.sidebar.radio(
        "Calculation Mode",
        ["Single Calculation", "Mixed Calculation"],
        index=0,
        key="calc_mode_radio"
    )

    if calculation_mode == "Single Calculation":
        asset_groups = {
            "Good Predictors": {
                2: "Bitcoin Cash",
                5: "EOS.IO",
                8: "IOTA",
                9: "Litecoin",
                11: "Monero",
                12: "Stellar",
                13: "TRON"
            },
            "Poor Predictors": {
                0: "Binance Coin",
                1: "Bitcoin",
                3: "Cardano",
                4: "Dogecoin",
                6: "Ethereum",
                7: "Ethereum Classic",
                10: "Maker"
            }
        }
    elif calculation_mode == "Mixed Calculation":
        asset_groups = {
            "Good Predictors": {
                0: "Binance Coin",
                1: "Bitcoin",
                2: "Bitcoin Cash",
                6: "Ethereum",
                7: "Ethereum Classic",
                9: "Litecoin",
                10: "Maker",
                11: "Monero"
            },
            "Poor Predictors": {
                3: "Cardano",
                4: "Dogecoin",
                5: "EOS.IO",
                8: "IOTA",
                12: "Stellar",
                13: "TRON"
            }
        }

    asset_mapping = {**asset_groups["Good Predictors"], **asset_groups["Poor Predictors"]}

    with st.sidebar:
        st.header("Model Configuration")

        selected_group = st.selectbox(
            "Select Predictor Group",
            options=list(asset_groups.keys()),
            index=0,
            key="group_selectbox"
        )

        selected_id = st.selectbox(
            "Select Cryptocurrency",
            options=list(asset_groups[selected_group].keys()),
            format_func=lambda x: f"{asset_groups[selected_group][x]}",
            index=0,
            key="crypto_selectbox"
        )

        if calculation_mode == "Mixed Calculation":
            selected_models = st.multiselect(
                "Select Models to Compare",
                options=['LightGBM', 'CatBoost'],
                default=['LightGBM'],
                key="model_multiselect"
            )
        else:
            selected_models = ['LightGBM']

        interval = st.selectbox(
            "Data Interval",
            options=["1m", "5m", "15m", "1h"],
            index=0,
            key="interval_selectbox"
        )

    @st.cache_data
    def load_data(asset_id, calculation_mode, selected_models=['LightGBM']):
        try:
            dfs = []

            for model in selected_models:
                if calculation_mode == "Single Calculation":
                    asset_name = asset_mapping[asset_id]
                    file_path = os.path.join(MODEL_DATA_DIR, "LightGBM", f"{asset_id}_{asset_name}.parquet")
                    df = pd.read_parquet(file_path)
                    df = df.rename(columns={
                        'actual_close': 'actual',
                        'predicted_close': 'predicted'
                    })

                    # Ensure datetime column exists for single calculation
                    if 'datetime' not in df.columns:
                        # If timestamp exists, convert to datetime
                        if 'timestamp' in df.columns:
                            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                        else:
                            # If no timestamp, create datetime from index
                            df = df.reset_index()
                            if 'datetime' in df.columns:
                                if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                                    df['datetime'] = pd.to_datetime(df['datetime'])
                            else:
                                # If no datetime column at all, create a dummy one
                                df['datetime'] = pd.date_range(start='2021-06-13', periods=len(df), freq='1min')

                    # Ensure we have the required columns
                    required_cols = ['datetime', 'actual', 'predicted']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        raise ValueError(f"Missing required columns: {missing_cols}")

                    # Sort by datetime
                    df = df.sort_values('datetime')

                    # Handle potential duplicates
                    df = df.drop_duplicates(subset=['datetime'], keep='last')

                else:
                    file_path = os.path.join(MODEL_DATA_DIR, "mixed_calculation", model, f"{asset_id}.parquet")
                    df = pd.read_parquet(file_path)

                    # Handle datetime conversion for mixed calculation data
                    if 'datetime' in df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                            df['datetime'] = pd.to_datetime(df['datetime'])
                    else:
                        # If no datetime column, create one from timestamp
                        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

                    # Ensure we have the required columns
                    required_cols = ['datetime', 'actual', 'predicted']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        raise ValueError(f"Missing required columns: {missing_cols}")

                    # Sort by datetime
                    df = df.sort_values('datetime')

                    # Handle potential duplicates
                    df = df.drop_duplicates(subset=['datetime'], keep='last')

                df['model'] = model
                dfs.append(df)

            combined_df = pd.concat(dfs)
            return combined_df[['datetime', 'actual', 'predicted', 'model']]

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    df = load_data(selected_id, calculation_mode, selected_models)

    if df.empty:
        st.warning("No data available for the selected cryptocurrency.")
        return

    st.markdown("---")
    st.subheader("Time Range Selection")

    # Get actual data time range
    data_start = df['datetime'].min().to_pydatetime()
    data_end = df['datetime'].max().to_pydatetime()

    with st.sidebar:
        start_date = st.date_input(
            "Start date",
            value=data_start.date(),
            min_value=data_start.date(),
            max_value=data_end.date(),
            key="start_date_input"
        )

        end_date = st.date_input(
            "End date",
            value=min(data_start + pd.DateOffset(months=7), data_end).date(),
            min_value=data_start.date(),
            max_value=data_end.date(),
            key="end_date_input"
        )

    start_datetime = pd.to_datetime(start_date).replace(hour=0, minute=0, second=0)
    end_datetime = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)

    filtered_df = df[
        (df['datetime'] >= start_datetime) &
        (df['datetime'] <= end_datetime)
        ].copy()

    if filtered_df.empty:
        st.warning("No data available in the selected time range.")
        return

    if interval != "1m":
        resample_rule = {
            "5m": "5min",
            "15m": "15min",
            "1h": "1H"
        }[interval]

        # Group by model and resample each separately
        resampled_dfs = []
        for model in selected_models:
            model_df = filtered_df[filtered_df['model'] == model].copy()

            # Set datetime as index for resampling
            model_df = model_df.set_index('datetime')

            # Resample only numeric columns
            resampled = model_df[['actual', 'predicted']].resample(resample_rule).mean()

            # Interpolate missing values (optional)
            resampled = resampled.interpolate(method='time')

            # Add model column back
            resampled['model'] = model

            # Reset index
            resampled = resampled.reset_index()

            resampled_dfs.append(resampled)

        filtered_df = pd.concat(resampled_dfs)

    prediction_quality = "Good" if selected_id in asset_groups["Good Predictors"] else "Poor"

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = go.Figure()

        # Add actual price (only once)
        first_model_df = filtered_df[filtered_df['model'] == selected_models[0]]
        fig.add_trace(go.Scatter(
            x=first_model_df['datetime'],
            y=first_model_df['actual'],
            name='Actual Price',
            line=dict(color='blue', width=2),
            opacity=0.8
        ))

        model_colors = {'LightGBM': 'red', 'CatBoost': 'green'}
        for model in selected_models:
            model_df = filtered_df[filtered_df['model'] == model]
            fig.add_trace(go.Scatter(
                x=model_df['datetime'],
                y=model_df['predicted'],
                name=f'{model} Predicted',
                line=dict(color=model_colors.get(model, 'orange'), width=1.5),
                opacity=0.7
            ))

        fig.update_layout(
            title=f'{asset_mapping[selected_id]} Price Prediction ({prediction_quality} Predictor) | Interval: {interval}',
            xaxis_title='Datetime',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=600,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔍 Feature Importance")

        if calculation_mode == "Single Calculation":
            feature_img_path = os.path.join(
                FEATURE_IMPORTANCE_DIR,
                f"{selected_id}_feature_importance.png"
            )
            if os.path.exists(feature_img_path):
                with st.container(border=True):
                    st.image(feature_img_path,
                             caption=f'{asset_mapping[selected_id]}',
                             use_container_width=True)
                    st.caption("Higher values indicate more important features")
            else:
                st.error("Feature importance visualization not available")
        else:
            for model in selected_models:
                feature_img_path = os.path.join(
                    MODEL_DATA_DIR,
                    "mixed_calculation",
                    "feature_importance",
                    f"{model}.png"
                )
                if os.path.exists(feature_img_path):
                    with st.container(border=True):
                        st.image(feature_img_path,
                                 caption=f'{model} Feature Importance',
                                 use_container_width=True)
                        st.caption(f"{model} feature importance (applies to all assets)")
                else:
                    st.warning(f"No feature importance image found for {model}")

    st.subheader("📈 Model Performance Metrics")
    metrics_data = []
    for model in selected_models:
        model_df = filtered_df[filtered_df['model'] == model]
        error = model_df['actual'] - model_df['predicted']
        metrics_data.append({
            'Model': model,
            'MAE': round(float(abs(error).mean()), 4),
            'RMSE': round((error ** 2).mean() ** 0.5, 4),
            'Correlation': round(model_df['actual'].corr(model_df['predicted']), 4),
            'Data Points': len(model_df)
        })

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.set_index('Model'), use_container_width=True)

    with st.expander("Advanced Analysis Options"):
        tab1, tab2 = st.tabs(["Model Parameters", "Feature List"])

        with tab1:
            if calculation_mode == "Single Calculation":
                st.json({
                    "objective": "regression",
                    "metric": "rmse",
                    "boosting_type": "gbdt",
                    "num_leaves": 100,
                    "max_depth": -1,
                    "min_data_in_leaf": 500,
                    "learning_rate": 0.01,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.5,
                    "reg_lambda": 0.5,
                    "n_jobs": -1,
                    "random_state": 42,
                    "verbosity": -1,
                    "n_estimators": 1000
                })
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("LightGBM Parameters")
                    st.json({
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'num_leaves': 100,
                        'max_depth': -1,
                        'min_data_in_leaf': 500,
                        'learning_rate': 0.01,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.5,
                        'reg_lambda': 0.5,
                        'n_jobs': -1,
                        'random_state': 42,
                        'verbosity': -1,
                        'n_estimators': 1000
                    })
                with col2:
                    st.subheader("CatBoost Parameters")
                    st.json({
                        'iterations': 1500,
                        'learning_rate': 0.03,
                        'depth': 8,
                        'l2_leaf_reg': 3,
                        'random_seed': 42,
                        'loss_function': 'RMSE',
                        'eval_metric': 'RMSE',
                        'early_stopping_rounds': 100,
                        'task_type': 'CPU',
                        'verbose': 50
                    })

        with tab2:
            st.write("Used Features:")
            st.dataframe(pd.DataFrame([
                'datetime',
                'timestamp',
                'Asset_ID',
                'Count',
                'Open',
                'High',
                'Low',
                'Close',
                'Volume',
                'VWAP',
                'Target',
                'Close_now_15',
                'Volume_now_15',
                'Close_now_30',
                'Volume_now_30',
                'Close_now_60',
                'Volume_now_60',
                'Close_now_90',
                'Volume_now_90',
                'Close_now_150',
                'Volume_now_150',
                'Close_now_600',
                'Volume_now_600',
                'Close_now_1500',
                'Volume_now_1500',
                'w',
                'W_Close_now_15'
            ], columns=["Feature Names"]))

    if st.checkbox("Show raw data preview"):
        st.dataframe(filtered_df.head(10))


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
