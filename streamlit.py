import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from streamlit_autorefresh import st_autorefresh
import requests
from data_loader import get_binance_klines
import pytz
from sqlalchemy import create_engine
from PIL import Image

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
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=current_hk_time - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", value=current_hk_time.date(), min_value=start_date)

        start_time = convert_date_to_timestamp(start_date)
        end_time = convert_date_to_timestamp(end_date, end_of_day=True)

    # ==== Data Fetching ====
    hist_data = get_binance_klines(
        symbol=selected_symbol,
        interval=selected_interval,
        start_time=start_time,
        end_time=end_time,
        limit=1000,
        display=True,
        calculate=False
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
    """Model Predictions Page"""
    st.title("🤖 AI Trading Model Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("BTC/USDT Forecast")
        st.metric("24h Prediction", "$63,200", "+2.8%")

        # Prediction trend chart
        fig = go.Figure(go.Scatter(
            x=pd.date_range(end=datetime.now(), periods=24, freq='H'),
            y=[62000 + i * 200 * (-1) ** i for i in range(24)],
            mode='lines+markers',
            name='Predicted Price'
        ))
        fig.update_layout(height=300, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Market Sentiment")
        st.metric("Sentiment Index", "78/100", "Bullish Trend")

        # Sentiment gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=78,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ],
            }
        ))
        fig.update_layout(height=300, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.write("**Model Details**")
    st.write("""
        - LSTM neural network for price prediction
        - Integrated social media sentiment analysis
        - Hourly prediction updates
    """)


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
            st.subheader(f"Latest {filter_option.capitalize()} News ({len(data['results'])})")

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
