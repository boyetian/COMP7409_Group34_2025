import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from streamlit_autorefresh import st_autorefresh
import requests
from data_loader import get_binance_klines
import pytz

# 定义时区
UTC_TZ = pytz.utc
HK_TZ = pytz.timezone('Asia/Hong_Kong')


def convert_date_to_timestamp(date_obj, end_of_day=False):
    """将香港时间日期对象转换为UTC时间戳（毫秒）"""
    dt = HK_TZ.localize(
        datetime(
            date_obj.year,
            date_obj.month,
            date_obj.day,
            hour=23 if end_of_day else 0,
            minute=59 if end_of_day else 0,
            second=59 if end_of_day else 0
        )
    )
    utc_dt = dt.astimezone(UTC_TZ)
    return int(utc_dt.timestamp() * 1000)


# 页面配置
st.set_page_config(
    page_title="Crypto Real-time Analytics Dashboard",
    page_icon="₿",
    layout="wide"
)

BINANCE_API_URL = "https://api.binance.com/api/v3"


@st.cache_data(ttl=10)
def fetch_realtime_price(symbol='BTCUSDT'):
    """获取实时价格（使用公开API）"""
    try:
        response = requests.get(
            f"{BINANCE_API_URL}/ticker/price",
            params={'symbol': symbol}
        )
        response.raise_for_status()
        data = response.json()
        return {
            'price': float(data['price']),
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.error(f"Failed to fetch real-time price: {str(e)}")
        st.stop()


def main():
    st.title("📈 Crypto Real-time Analytics Dashboard")
    st.markdown("---")

    # Auto-refresh component
    st_autorefresh(interval=60 * 1000, key="data_refresh")

    # Initial data load
    hist_data = get_binance_klines(symbol='BTCUSDT', interval='1d',
                                   start_time=None, end_time=None,
                                   limit=1000, calculate=False)
    realtime_data = fetch_realtime_price()

    # Sidebar controls
    with st.sidebar:
        st.header("Control Panel")

        # Symbol selection
        selected_symbol = st.selectbox(
            "Select Trading Pair",
            [
                'BTCUSDT', 'ETHUSDT', 'XRPUSDT',
                'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
                'DOGEUSDT', 'TRXUSDT', 'LINKUSDT'
            ],
            index=0
        )

        # Interval selection
        selected_interval = st.selectbox(
            "Select Time Interval",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=1
        )

        # Date inputs
        current_hk_time = datetime.now(HK_TZ)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date (HKT)",
                value=current_hk_time - timedelta(days=30),
                max_value=current_hk_time.date()
            )
        with col2:
            end_date = st.date_input(
                "End Date (HKT)",
                value=current_hk_time.date(),
                min_value=start_date
            )

        # Convert to timestamps
        start_time = convert_date_to_timestamp(start_date)
        end_time = convert_date_to_timestamp(end_date, end_of_day=True)

    # Fetch filtered data
    hist_data = get_binance_klines(
        symbol=selected_symbol,
        interval=selected_interval,
        start_time=start_time,
        end_time=end_time,
        limit=1000,
        calculate=False
    )

    realtime_data = fetch_realtime_price(symbol=selected_symbol)

    # Real-time metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        if len(hist_data) >= 2:
            delta = hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]
            delta_pct = (delta / hist_data['Close'].iloc[-2]) * 100
        else:
            delta = 0
            delta_pct = 0

        st.metric(
            f"{selected_symbol} Current Price",
            f"${realtime_data['price']:,.2f}",
            delta=f"{delta:+.2f} ({delta_pct:+.2f}%)"
        )

    # Visualization
    fig = go.Figure()

    # Price line chart
    fig.add_trace(go.Scatter(
        x=hist_data['Close Time'],
        y=hist_data['Close'],
        name='Price Trend',
        line=dict(color='#6600FF')
    ))

    # Realtime price marker
    fig.add_trace(go.Scatter(
        x=[realtime_data['timestamp']],
        y=[realtime_data['price']],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Live Price'
    ))

    fig.update_layout(
        title=f"{selected_symbol} Price Chart (with Live Updates)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()