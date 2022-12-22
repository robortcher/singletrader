from pypfopt.efficient_frontier import EfficientFrontier
import numpy as np
import pandas as pd
import warnings
import math
import cvxpy as cp
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.risk_models import fix_nonpositive_semidefinite
def get_beta_cov_matrix(prices,market_returns,price_data=False,frequency=252):
    r"""
    calculate assets' beta with market return 
    """
    prices = prices.copy()
    if price_data:
        prices = prices.pct_change()
    if isinstance(market_returns, str):
        market_returns = prices.pop(market_returns)
    
    corr_with_mkt = prices.corrwith(market_returns)
    beta = corr_with_mkt * prices.std() / market_returns.std()
    
    r_expected = pd.DataFrame(market_returns).values.dot(pd.DataFrame(beta).T)
    r_expected = pd.DataFrame(r_expected,columns=beta.index,index=market_returns.index)
    alpha = prices - r_expected
    omiga = np.diag(alpha.var())
    
    cov_matrix_capm = pd.DataFrame(beta).dot(pd.DataFrame(beta).T) * market_returns.var() * frequency + omiga * frequency
    return beta,cov_matrix_capm


def get_efficient_frontier_(linspace=1000,constraint=None,**kwargs):
    r"""
    calculate general efficient frontier
    """
    expected_returns = kwargs['expected_returns']
    cov_matrix = kwargs['cov_matrix']
    
    sigma_max = np.diagonal(cov_matrix).max() ** 0.5
    sigma_min = np.diagonal(cov_matrix).min() ** 0.5
    if type(linspace) is int:
        sigma_range = np.linspace(sigma_min,sigma_max, linspace)
    else:
        sigma_range = linspace
    
    # return_max = expected_returns.max()
    # return_min = expected_returns.min()
    # return_range = np.linspace(return_min,return_max, linspace)
    
    ef = EfficientFrontier(solver=cp.ECOS,**kwargs)

    if constraint is not None:
        ef.add_constraint(constraint)
    
    weights = []
    performances = []
    for sigma in sigma_range:
        ef.efficient_risk(sigma)
        weights.append(ef.weights)
        performances.append(ef.portfolio_performance())
    
    
    performances_df = pd.DataFrame(performances,columns=['return','volatility','sharpe_ratio'])
    weights_df = pd.DataFrame(weights,columns=expected_returns.index)
    return performances_df,weights_df

def get_efficient_frontier(linspace=1000,constraint=None,sector_constraint=None,method='risk',**kwargs):
    r"""
    calculate general efficient frontier
    """
    expected_returns = kwargs['expected_returns']
    cov_matrix = kwargs['cov_matrix']
    
    sigma_max = np.diagonal(cov_matrix).max() ** 0.5
    sigma_min = np.diagonal(cov_matrix).min() ** 0.5
    if type(linspace) is int:
        sigma_range = np.linspace(sigma_min,sigma_max, linspace)
    else:
        sigma_range = linspace
    
    # return_max = expected_returns.max()
    # return_min = expected_returns.min()
    # return_range = np.linspace(return_min,return_max, linspace)
    
    ef = EfficientFrontier(**kwargs)
    # ef2 = EfficientFrontier(**kwargs)
    # cons = constraints.copy()
    # for _con in cons:
    if constraint is not None:
        if type(constraint) is list:
            for cons in constraint:
                  ef.add_constraint(cons)
        else:
            ef.add_constraint(constraint)
    
    if sector_constraint is not None:
        ef.add_sector_constraints(**sector_constraint)
        # ef.add_sector_constraint(sector_constraint['sector_mapper'],sector_constraint['sector_lower'],sector_constraint['sector_upper'])
    
    weights = []
    performances = []
    if method == 'risk':
        for sigma in sigma_range:
            ef.efficient_risk(sigma)
            weights.append(ef.weights)
            performances.append(ef.portfolio_performance())

    elif method == 'max_sharpe':
        ef.max_sharpe()
        weights.append(ef.weights)
        performances.append(ef.portfolio_performance())
    
    
    
    performances_df = pd.DataFrame(performances,columns=['return','volatility','sharpe_ratio'])
    weights_df = pd.DataFrame(weights,columns=expected_returns.index)
    return performances_df,weights_df




def get_implied_risk_aversion(market_prices,price_data=False,risk_free_rate=0.02,frequency=252):
    r"""
    Calculate the market-implied risk-aversion parameter (i.e market price of risk)
    based on market prices. For example, if the market has excess returns of 10% a year
    with 5% variance, the risk-aversion parameter is 2, i.e you have to be compensated 2x
    the variance.

    .. math::

        \delta = \frac{R - R_f}{\sigma^2}

    :param market_prices: the (daily) returns/prices of the market portfolio, e.g SPY.
    :type market_prices: pd.Series with DatetimeIndex.
    :param price_data: wheather the market_prices is prices
    :type price_data: bool
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                            The period of the risk-free rate should correspond to the
                            frequency of expected returns.
    :type risk_free_rate: float, optional
    :raises TypeError: if market_prices cannot be parsed
    :return: market-implied risk aversion
    :rtype: float
    """
    if price_data:
        market_prices = market_prices.pct_change()
    r_mkt = market_prices.mean() * frequency
    var_mkt = market_prices.var() * frequency
    return (r_mkt - risk_free_rate) / var_mkt


def get_market_implied_prior_returns(
        market_caps, risk_aversion, cov_matrix, risk_free_rate=0.02
    ):
    r"""
    Compute the prior estimate of returns implied by the market weights.
    In other words, given each asset's contribution to the risk of the market
    portfolio, how much are we expecting to be compensated?

    .. math::

        \Pi = \delta \Sigma w_{mkt}

    :param market_caps: market capitalisations of all assets
    :type market_caps: {ticker: cap} dict or pd.Series
    :param risk_aversion: risk aversion parameter
    :type risk_aversion: positive float
    :param cov_matrix: covariance matrix of asset returns
    :type cov_matrix: pd.DataFrame
    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                           You should use the appropriate time period, corresponding
                           to the covariance matrix.
    :type risk_free_rate: float, optional
    :return: prior estimate of returns as implied by the market caps
    :rtype: pd.Series
    """
    if not isinstance(cov_matrix, pd.DataFrame):
        warnings.warn(
            "If cov_matrix is not a dataframe, market cap index must be aligned to cov_matrix",
            RuntimeWarning,
        )
    mcaps = pd.Series(market_caps)
    mkt_weights = mcaps / mcaps.sum()
    # Pi is excess returns so must add risk_free_rate to get return.
    return risk_aversion * cov_matrix.dot(mkt_weights) + risk_free_rate



# class BlackLittermanModel():
#     def __init__(
#         self,
#         prices,
#         w_mkt=None,
#         pirce_data=False,
        
                
#     )


def get_bl_return_cov(
        prices,
        market_caps,
        price_data=False,
        delta=None,
        frequency=252,
        tau=1,
        report=False,
        cov_method = 'single_factor',
        f_views=None
    ):
    r"""
    get expected returns and covariance matrix from Black-Litterman Model
    with views is histrical mean and covariance.
    """
    prices = prices.copy()
    market_caps = market_caps.copy()
    market_caps = market_caps / market_caps.sum()
    
    prices_with_mkt = prices.copy()
    prices_with_mkt['mkt'] = (market_caps * prices_with_mkt).sum(axis=1)
    market_prices = prices_with_mkt['mkt']
    if price_data:
        prices = prices.pct_change()
        # market_prices = market_prices.pct_change()

    returns_hist = prices.mean()*frequency
    cov_matrix_hist =  prices.cov()*frequency
    
    if delta is None:
        delta = get_implied_risk_aversion(market_prices=market_prices,frequency=frequency)
    beta,cov_matrix_capm = get_beta_cov_matrix(prices=prices_with_mkt,market_returns='mkt',frequency=frequency)
    Pi = get_market_implied_prior_returns(market_caps=market_caps,risk_aversion=delta,cov_matrix=cov_matrix_capm)
    
    cov_matrix_capm_inv = np.linalg.inv(cov_matrix_capm)
    cov_matrix_hist_inv = np.linalg.inv(cov_matrix_hist)
    
    cov_matrix_bl = np.linalg.inv(cov_matrix_capm_inv * tau + cov_matrix_hist_inv)
    returns_bl = pd.Series(cov_matrix_bl.dot(tau * cov_matrix_capm_inv.dot(Pi) + cov_matrix_hist_inv.dot(returns_hist)),index=returns_hist.index)
    cov_matrix_bl = fix_nonpositive_semidefinite(cov_matrix_bl)
    
    if f_views is not None:
        returns_bl = (tau * Pi + f_views) / (tau + 1)

    if cov_method is not None:
        cov_shrinkage =  CovarianceShrinkage(prices=prices,frequency=frequency,returns_data=True)
        cov_matrix_bl = cov_shrinkage.ledoit_wolf(cov_method)
        
    if report:
        returns_bl = pd.concat([returns_hist, Pi,returns_bl],axis=1)
        cov_matrix_bl = np.c_[np.diagonal(cov_matrix_hist),np.diagonal(cov_matrix_capm),np.diagonal(cov_matrix_bl)]
    return returns_bl,cov_matrix_bl


def get_rolling_weights(prices,length_min=None,length_add=1,linspace=1000,method='bl',**kwargs):
    weights_dict = {}
    all_periods = prices.index
    length_all = len(all_periods)
    if length_min is None:
        length_min = int(1/2*length_all)
    round_need_renew =  math.ceil((length_all - length_min) / length_add)-1
    
    for round in range(round_need_renew):
        _prices = prices.iloc[:length_min + round * length_add] 
        _returns_bl,_cov_matrix_bl = get_bl_return_cov(prices=_prices,**kwargs)
        can_use_date = all_periods[length_min + round * length_add]
        if method == 'bl':
            weights_dict[can_use_date] = get_efficient_frontier(
                linspace=linspace,expected_returns=_returns_bl, cov_matrix=_cov_matrix_bl)[1]
        # elif method == 'risk-parity':
            
    weights_df = pd.concat(weights_dict)
    return weights_df
           
        
    
    
    
        








