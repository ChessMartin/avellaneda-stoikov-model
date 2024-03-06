from pathlib import Path
import pandas as pd
import numpy as np


def load_data(message_data: Path, order_book_data: Path) -> pd.DataFrame:
    orders_df = pd.read_csv(message_data)
    book_stats_df = pd.read_csv(order_book_data)

    assert len(orders_df) == len(book_stats_df), "Datasets must have the same number of rows"

    merged_df = pd.concat([orders_df, book_stats_df], axis=1)

    merged_df['full_ts'] = pd.to_datetime(merged_df['ts'], unit='s', origin='2013-09-01')
    merged_df = merged_df[merged_df['order_type'] == 1]
    merged_df.reset_index(drop=True, inplace=True)

    merged_df.drop(columns=['order_type', 'order_id', 'volume', 'price', 'ask_volume', 'bid_volume'], inplace=True)

    min_timestamp = merged_df['ts'].min()
    max_timestamp = merged_df['ts'].max()

    merged_df['ts'] = (merged_df['ts'] - min_timestamp) / (max_timestamp - min_timestamp)

    merged_df['ask_price'] /= 10_000
    merged_df['bid_price'] /= 10_000

    return merged_df


def add_mid_price_column(df):
    df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2

    return df


def add_sigma_column(df):
    df['log_return'] = np.log(df['mid_price'] / df['mid_price'].shift(1))

    df['sigma'] = np.std(df['log_return'].dropna() * 54817)

    return df.dropna()


def add_time_horizon_column(df):
    df = df.sort_values(by='ts', ascending=True)

    last_timestamp = df['ts'].iloc[-1]
    df['time_horizon'] = last_timestamp - df['ts']

    return df


def add_k_column(df):
    total_orders = len(df)
    time_duration = df['ts'].iloc[-1] - df['ts'].iloc[0]

    k = total_orders / time_duration if time_duration > 0 else 0

    df['k'] = k

    return df


def add_A_column(df):
    df['spread'] = df['ask_price'] - df['bid_price']
    df['A'] = df['spread'].mean()

    return df
