import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm
from datetime import datetime
import sys

import trade.self_uqer as qs

def _get_assets_series(start, end):
    assets_series = pd.read_csv('./data/index_industry_d.csv',index_col=0)
    assets_series.index = pd.to_datetime(assets_series.index)
    assets_series = assets_series.loc[start:end]
    assets_names = assets_series.columns.to_list()
    return assets_series, assets_names

## 获取benchmark_series指数日度收盘数据
def _get_benchmark_series(start, end, benchmark):
    benchmark_series = qs.get_data_indexes(benchmark, 'close', start, end)
    benchmark_series = benchmark_series.loc[start:end]
    benchmark_return_series = benchmark_series.pct_change()
    benchmark_return_series.fillna(0,inplace=True)
    return benchmark_series, benchmark_return_series

## 获取无风险收益率
def _get_rf_series(start, end):
    rf_return_series = qs.get_data_InterestRate(start,end,period='10Y') / 252
    rf_return_series.fillna(method='ffill',inplace=True)
    return rf_return_series

## 交易日数据
def _get_tradentransfer_date(stard, end):
    trade_date = qs.get_TradeDate(start, end)
    transfer_date = qs.get_TransferDate(start, end, end=True, freq='month')
    return trade_date, transfer_date
### /* 2015-11-30 当天有因子数据, 2022-02-28为最后一个交易日 */
## 单因子读取
# 确保因子的columns和assets的columns必须对应
def _get_factor_series(start, end, assets_names, trade_date,  transfer_date):
    factors = dict()
    factor = pd.read_csv('./factors/industry_sarimaxANDsarima_predictions.csv',index_col=0)
    factor = factor.loc[start:end]
    factor = factor.loc[:,assets_names]
    factor.index = [transfer_date.asof(x) for x in factor.index] # 转换为真实的交易日
    factor.index = pd.to_datetime(factor.index)

    temp = pd.DataFrame(data=factor, index=trade_date) # 转换因子频率, 试试reindex
    temp.fillna(method='ffill', inplace=True)
    # TODO
    temp.dropna(inplace=True) # 我觉得没有必要dropna()
    factors['pred_pe_yoy'] = temp
    return factors