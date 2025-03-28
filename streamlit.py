import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import requests
from data_loader import get_binance_klines
import pytz

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
def main():
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


if __name__ == "__main__":
    main()
