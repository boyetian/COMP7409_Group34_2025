import requests
import pandas as pd
from datetime import datetime
import os
import pytz
from dateutil.relativedelta import relativedelta

def get_binance_klines(symbol='BTCUSDT', interval='1m', start_time='2025-01-24 00:00:00',
                       end_time='2025-03-24 00:00:00', limit=1000, calculate = True):
    """
    获取 Binance 交易对的历史 K 线数据，支持选择开始和结束时间。
    参数:
    - symbol: 交易对（例如 BTCUSDT）
    - interval: 时间间隔（例如 '1m', '5m', '1h', '1d'）
    - start_time: 开始时间（字符串，格式为 'YYYY-MM-DD HH:MM:SS'）
    - end_time: 结束时间（字符串，格式为 'YYYY-MM-DD HH:MM:SS'）
    - limit: 返回数据的数量限制（最多 1000）
    返回:
    dataframe 包括
    1. Open Time: 当前K线的开盘时间戳，Unix时间戳（毫秒），需要转换为日期时间格式。
    2. Open: 当前K线的开盘价，即该时间段开始时的交易价格。
    3. High: 当前K线的最高价，即该时间段内的最高交易价格。
    4. Low: 当前K线的最低价，即该时间段内的最低交易价格。
    5. Close: 当前K线的收盘价，即该时间段结束时的交易价格。
    6. Volume: 当前K线内的交易量，即该时间段内成交的总数量。
    7. Close Time: 当前K线的结束时间戳，Unix时间戳（毫秒），需要转换为日期时间格式。
    8. Quote Asset Volume: 以法定货币计量的交易量，即该时间段内的交易总价值。
    9. Number of Trades: 当前时间段内的交易次数。
    10. Taker Buy Base Volume: Taker（主动方）在该时间段买入的基础资产量。
    11. Taker Buy Quote Volume: Taker（主动方）在该时间段买入的法定货币量。
    12. Ignore: 占位符，通常不使用，可忽略。
    """
    # 设置东八区时区
    # 处理时间参数
    tz = pytz.timezone('Asia/Hong_Kong')
    UTC_TZ = pytz.utc

    # 处理end_time
    if end_time is None:
        end_time = datetime.now(tz)
    else:
        if isinstance(end_time, (int, float)):  # 处理时间戳
            end_time = datetime.fromtimestamp(end_time/1000, tz=UTC_TZ).astimezone(tz)
        elif isinstance(end_time, str):  # 处理字符串
            end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
            end_time = tz.localize(end_time)
        else:  # 处理datetime对象
            if end_time.tzinfo is None:
                end_time = tz.localize(end_time)
            else:
                end_time = end_time.astimezone(tz)

    # 处理start_time
    if start_time is None:
        start_time = end_time - relativedelta(months=3)
    else:
        if isinstance(start_time, (int, float)):  # 处理时间戳
            start_time = datetime.fromtimestamp(start_time/1000, tz=UTC_TZ).astimezone(tz)
        elif isinstance(start_time, str):  # 处理字符串
            start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            start_time = tz.localize(start_time)
        else:  # 处理datetime对象
            if start_time.tzinfo is None:
                start_time = tz.localize(start_time)
            else:
                start_time = start_time.astimezone(tz)

    # 转换为UTC时间戳（毫秒）
    start_timestamp = int(start_time.astimezone(pytz.utc).timestamp() * 1000)
    end_timestamp = int(end_time.astimezone(pytz.utc).timestamp() * 1000)
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
        response = requests.get(url, params=params)
        data = response.json()
        if not data:  # 数据已全部获取
            break
        all_data.extend(data)
        # 更新下一个请求的起始时间（最后一条的Close Time + 1ms）
        current_start = data[-1][6] + 1  # 避免重复
    # 创建DataFrame并处理时间列
    columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
               'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Volume',
               'Taker Buy Quote Volume', 'Ignore']
    df = pd.DataFrame(all_data, columns=columns)
    # 转换时间戳为东八区时间
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(
        'Asia/Hong_Kong')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(
        'Asia/Hong_Kong')
    # 数值类型转换
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Quote Asset Volume', 'Taker Buy Base Volume', 'Taker Buy Quote Volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    if calculate:
        # 计算15分钟VWAP
        df['VWAP_15m'] = (df['Close'] * df['Volume']).rolling(15).sum() / df['Volume'].rolling(15).sum()
    return df


def fetch_and_save_data(symbols, start_time, end_time, output_folder="data"):
    """
    批量获取多个币种的历史数据，并保存为CSV文件
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for symbol in symbols:
        print(f"正在获取 {symbol} 数据...")
        df_binance = get_binance_klines(symbol=symbol, start_time=start_time, end_time=end_time)

        # 输出数据的前几行，查看计算后的 VWAP 列
        print(df_binance[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP_15m']].head())

        # 保存到本地csv文件，方便后期回测使用
        output_file = os.path.join(output_folder, f'{symbol}.csv')
        df_binance.to_csv(output_file)
        print(f"{symbol} 数据已保存至 {output_file}")


if __name__ == "__main__":
    symbols = [
        'BTCUSDT',  # 比特币与Tether交易对
        # 'ETHUSDT',  # 以太坊与Tether交易对
        # 'XRPUSDT',  # 瑞波币与Tether交易对
        # 'BNBUSDT',  # Binance Coin与Tether交易对
        # 'SOLUSDT',  # Solana与Tether交易对
        # 'ADAUSDT',  # Cardano与Tether交易对
        # 'DOGEUSDT',  # 狗狗币与Tether交易对
        # 'TRXUSDT',  # TRON与Tether交易对
        # 'LINKUSDT'  # Chainlink与Tether交易对
    ]

    start_time = None  # 设置开始时间 e.g.'2025-02-24 00:00:00'
    end_time = None  # 设置结束时间

    fetch_and_save_data(symbols, start_time, end_time)