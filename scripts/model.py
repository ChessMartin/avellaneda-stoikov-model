import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def fit_A_parameter(df, bins=72):
    df['binned'] = pd.cut(df['ts'], bins=bins, labels=False)
    grouped = df.groupby('binned').size()

    bin_edges = np.linspace(df['ts'].min(), df['ts'].max(), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def exp_decay(x, A, k):
        return A * np.exp(-k * x)

    popt, pcov = curve_fit(exp_decay, bin_centers, grouped, p0=(grouped.max(), 0.1))

    return popt[0]


def optimal_quotes(s, sigma, q, T, k, A, gamma):
    """
    Calculate the optimal bid and ask quotes based on the Avellaneda-Stoikov model.

    Parameters:
    - s:     The current stock price.
    - sigma: The volatility of the stock.
    - q:     The current inventory of the market maker.
    - T:     The time horizon for the market making strategy.
    - k:     The order arrival rate (intensity).
    - A:     The spread adjustment parameter.
    - gamma: The risk aversion parameter.

    Returns:
    - bid: The optimal bid price.
    - ask: The optimal ask price.
    """
    adjusted_mid = s - gamma * sigma**2 * q * T

    delta = np.sqrt(gamma * sigma**2 * T + (2/gamma) * np.log(1 + (gamma/k)))
    spread = delta + (2/gamma) * np.log(1 + (gamma * A))

    bid = adjusted_mid - spread / 2
    ask = adjusted_mid + spread / 2

    return bid, ask
