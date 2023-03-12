import logging
import pandas as pd
import numpy as np
import alphalens
try:
    from factor_formulas import *
except:
    from .factor_formulas import *
try:
    from utility import parLapply
except:
    from .utility import parLapply
from functools import partial
from itertools import product

def quantile_turnover(quantile_factor,period=1):
    def __quantile_turnover(d,period=period):
        """
        Computes the proportion of names in a factor quantile that were
        not in that quantile in the previous period.

        Parameters
        ----------
        quantile_factor : pd.Series
            DataFrame with date, asset and factor quantile.
        quantile : int
            Quantile on which to perform turnover analysis.
        period: int, optional
            Number of days over which to calculate the turnover.

        Returns
        -------
        quant_turnover : pd.Series
            Period by period turnover for that quantile.
        """

        quant_names = d
        quant_name_sets = quant_names.groupby(level=['date']).apply(
            lambda x: set(x.index.get_level_values('asset')))

        name_shifted = quant_name_sets.shift(period)

        new_names = (quant_name_sets - name_shifted).dropna()
        quant_turnover = new_names.apply(
            lambda x: len(x)) / quant_name_sets.apply(lambda x: len(x))
        return quant_turnover
    return quantile_factor.groupby(quantile_factor).apply(lambda x:__quantile_turnover(x)).unstack().T

class FactorEvaluation:
    """
    因子组合构建
    """
    def __init__(self, bar_data, factor_data):
        bar_data = pd.DataFrame(bar_data)
        factor_data = pd.DataFrame(factor_data)
        self.bar_data = bar_data
        self.raw_factor_data = factor_data.add_prefix('raw_')
        self.factor_data = factor_data
        self.all_factors = self.factor_data.columns


    def compute_forward_returns(self,
                                factor,
                                prices,
                                periods=(1, 5, 10),
                                filter_zscore=None,
                                cumulative_returns=True,
                                price_type='open',
                                add_shift=1):
        # prices = self.bar_data[price_type].unstack()
        returns_df = alphalens.utils.compute_forward_returns(
            self.factor_data,
            prices=prices,
            periods=periods,
            filter_zscore=filter_zscore,
            cumulative_returns=cumulative_returns
        )
        return returns_df.groupby(level='asset').apply(lambda x:x.shift(-add_shift))

    def get_clean_factor(self,
                        forward_returns=None,
                        groupby=None,
                        binning_by_group=False,
                        quantiles=5,
                        bins=None,
                        groupby_labels=None,
                        max_loss=0.35,
                        zero_aware=False):
        if forward_returns is None:
            forward_returns = self.compute_forward_returns()
        clean_factor = {}
        for factor in self.all_factors:
            factor_data = self.factor_data[factor]
            clean_factor[factor] = alphalens.utils.get_clean_factor(factor_data,
                        forward_returns,
                        groupby,
                        binning_by_group,
                        quantiles,
                        bins,
                        groupby_labels,
                        max_loss,
                        zero_aware)
        return clean_factor


    def get_clean_factor_and_forward_returns(self,
        groupby=None,
        binning_by_group=False,
        quantiles=5,
        bins=None,
        periods=(1, 5, 10),
        filter_zscore=20,
        groupby_labels=None,
        max_loss=0.35,
        zero_aware=False,
        cumulative_returns=True,
        price_type='open',
        add_shift=1
    ):
        prices = self.bar_data[price_type].unstack()
        clean_factor = {}
        for factor in self.all_factors:
            factor_data = self.factor_data[factor]
            forward_returns = self.compute_forward_returns(
                factor_data,
                prices,
                periods,
                filter_zscore,
                cumulative_returns,
                add_shift
            )
            clean_factor[factor] = alphalens.utils.get_clean_factor(factor_data, forward_returns, groupby=groupby,
                                   groupby_labels=groupby_labels,
                                   quantiles=quantiles, bins=bins,
                                   binning_by_group=binning_by_group,
                                   max_loss=max_loss, zero_aware=zero_aware)
            self.clean_factor = clean_factor
        return clean_factor
    
    
    
    def factor_rank_autocorrelation(self, period=1):
        """因子排序自相关系数"""
        factor_data = self.factor_data
        grouper = [factor_data.index.get_level_values('date')]

        
        res = []
        for factor in factor_data.columns:
            ranks = factor_data.groupby(grouper)[factor].rank()
            asset_factor_rank = ranks.reset_index().pivot(index='date',
                                                            columns='asset',
                                                            values=factor)

            asset_shifted = asset_factor_rank.shift(period)
            autocorr = asset_factor_rank.corrwith(asset_shifted, axis=1)
            autocorr.name = period
            res.append(autocorr)
        res = pd.concat(res,axis=1)
        res.columns = factor_data.columns
            
        return res
    
    def factor_information_coefficient(self,
                                   group_adjust=False,
                                   by_group=False,
                                   **kwargs):
        clean_factor = getattr(self,'clean_factor', self.get_clean_factor_and_forward_returns(**kwargs))
        ics = {}
        for factor in self.all_factors:
            factor_data = clean_factor[factor]
            ics[factor] = alphalens.performance.factor_information_coefficient(factor_data)
        ics = pd.concat(ics,axis=1)
        return ics


    def factor_returns(self,
                    demeaned=True,
                   group_adjust=False,
                   equal_weight=False,
                   by_asset=False,
                   **kwargs):
        clean_factor = getattr(self,'clean_factor', self.get_clean_factor_and_forward_returns(**kwargs))
        factor_returns = {}
        for factor in self.all_factors:
            factor_data = clean_factor[factor]
            factor_returns[factor] = alphalens.performance.factor_returns(factor_data)
        
        factor_returns = pd.concat(factor_returns,axis=1)
        return factor_returns
    


    def quantile_turnover(self,**kwargs):
        clean_factor = getattr(self,'clean_factor', self.get_clean_factor_and_forward_returns(**kwargs))
        turnovers = {}
        for factor in self.all_factors:
            factor_data = clean_factor[factor]
            quantile_factor = factor_data['factor_quantile']
            # turnovers[factor] = alphalens.performance.quantile_turnover(quantile_factor,quantile, period)
            turnovers[factor] = quantile_turnover(quantile_factor)
        
        turnovers = pd.concat(turnovers,axis=1)
        return turnovers