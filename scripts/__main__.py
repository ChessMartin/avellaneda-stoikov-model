from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import preprocessing
import model

MESSAGE_DATA_FILE_PATH = Path('../data/raw/AAPL_2012-06-21_34200000_57600000_message_1.csv')
ORDERBOOK_DATA_FILE_PATH = Path('../data/raw/AAPL_2012-06-21_34200000_57600000_orderbook_1.csv')

if __name__ == "__main__":
    df = preprocessing.load_data(MESSAGE_DATA_FILE_PATH, ORDERBOOK_DATA_FILE_PATH)

    df = preprocessing.add_mid_price_column(df)
    df = preprocessing.add_time_horizon_column(df)
    df = preprocessing.add_sigma_column(df)
    df = preprocessing.add_k_column(df)
    df = preprocessing.add_A_column(df)

    print(df[['mid_price', 'sigma', 'time_horizon', 'k', 'A']])
    test_dic = {
        'ts': df['ts'].copy(),
        'bid': df['bid_price'].copy(),
        'ask': df['ask_price'].copy(),
        'adj_bid': [],
        'adj_ask': []
    }

    inventory_level = 1
    risk_aversion = 0.1
    for index, row in df.iterrows():
        adj_bid, adj_ask = model.optimal_quotes(
            s=row['mid_price'],
            sigma=10,
            q=inventory_level,
            T=row['time_horizon'],
            k=row['k'],
            A=row['A'],
            gamma=risk_aversion
        )
        if adj_bid >= row['ask_price']:
            inventory_level += 1
            print("there is a Buy trade")

        if adj_ask <= row['bid_price']:
            inventory_level -= 1
            print("there is a Sell trade")

        test_dic['adj_bid'].append(adj_bid)
        test_dic['adj_ask'].append(adj_ask)

    test_df = df.iloc[10_000:10_500]
    test_df = pd.DataFrame(test_dic)

    test_df.set_index('ts', inplace=True)
    test_df['mid'] = (test_df.ask + test_df.bid) / 2
    test_df['adj_mid'] = (test_df.adj_ask + test_df.adj_bid) / 2

    test_df['spread'] = (test_df.adj_ask - test_df.adj_bid)
    test_df['adj_spread'] = (test_df.adj_ask - test_df.adj_bid)

    plt.plot(test_df.bid, c='blue', ls='-')
    plt.plot(test_df.ask, c='red', ls='-')
    plt.plot(test_df.mid, c='black', ls='-')

    plt.plot(test_df.adj_bid, c='blue', ls='--')
    plt.plot(test_df.adj_ask, c='red', ls='--')
    plt.plot(test_df.adj_mid, c='black', ls='--')

    plt.show()
