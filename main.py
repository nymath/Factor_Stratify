#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
sudo chmod a+x main.py

'''
#%% Loading Modules
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


#%%

qs.set_font() # 设置中文字体

#%% 回测参数设置
try:
    start = sys.argv[1]
    end = sys.argv[2]
    benchmark = sys.argv[3]
except:
    start = '2017-10-01'
    end = '2022-01-15'
    benchmark = '000300.XSHG'
if len(start)!= 10 or len(end)!=10:
    raise ValueError
if benchmark not in ['000300.XSHG','000016.XSHG','000905.XSHG','000906.XSHG']:
    print('系统暂不支持输入的benchmark')
    raise ValueError

try:
    whetherplot = sys.argv[4]
except:
    whetherplot = "plot"
    
# TODO
# freq = 'd'  
# refresh_rate = 'Monthly(-1)' 


#%% Data Loading 
## 获取申万一级行业的日度收盘数据
# TODO 把这些函数封装到一个文件里, 减少主体框架的代码
from main_dataloading import _get_assets_series
assets_series, assets_names = _get_assets_series(start, end)

## 获取benchmark_series指数日度收盘数据
from main_dataloading import _get_benchmark_series
benchmark_series, benchmark_return_series = _get_benchmark_series(start, end, benchmark)

## 获取无风险收益率
from main_dataloading import _get_rf_series
rf_return_series = _get_rf_series(start, end)

## 交易日数据
trade_date = qs.get_TradeDate(start, end)
transfer_date = qs.get_TransferDate(start, end, end=True, freq='month')

## 读取因子
from main_dataloading import _get_factor_series
factors = _get_factor_series(start, end, assets_names, trade_date, transfer_date)


if __name__ == '__main__':

    dict_weight_series, simple_return_series = qs.StratifiedBackTest(assets_series, factors['pred_pe_yoy'], transfer_date,n=10)
    result = {}
    for col in simple_return_series.columns:
        result[col] = qs.RiskAssessor(simple_return_series[col], benchmark_return_series, rf_return_series)

    
    resultprint = pd.DataFrame(columns=list(result.keys()),dtype='float64')
    resultprint.loc['mean',:] = [result[col].rp_a_mean for col in result.keys()]
    resultprint.loc['std',:] = [result[col].rp_a_std for col in result.keys()]
    resultprint.loc['alpha',:] = [result[col].reg_alpha for col in result.keys()]
    resultprint.loc['beta',:] = [result[col].reg_beta for col in result.keys()]
    resultprint.loc['InfomationRatio',:] = [result[col].reg_infomation_ratio for col in result.keys()]
    resultprint.loc['SharpeRatio',:] = [result[col].SharpeRatio for col in result.keys()]
    resultprint.loc['maxdrawdown',:] = [result[col].drawdownmax for col in result.keys()]
    resultprint.loc['maxduration',:] = [result[col].durationmax for col in result.keys()]
    print(resultprint)
    import pprint 

    color_lst = []
    fig, ax = plt.subplots(figsize=(8,5))
    for i in simple_return_series.columns:
        ax.plot((1+simple_return_series[i]).cumprod(), label=i)
    ax.grid()
    ax.legend(bbox_to_anchor=(1.01, 1),loc='upper left',borderaxespad=0.)
    ax.set_title('因子分层测试, 升序分组')
    ax.set_ylabel('净值')
    plt.tight_layout()
    plt.show()

    ## 各年收益率

    ## 总收益率

    ## 夏普/总夏普, 这里来一个.values.print('-----总夏普-----')

    print(test['sharpe'])
    # 最大回撤
    print('-----最大回撤-----')
    print(test['max_drawdown'])
    print('-----alpha-----')
    print(test['alpha'])
    print('--------年化换手率---------')
    print(cal_turnover(dict_weight_series['G10']))
    print('-------pe_yoy_pred----------IC,ICIR')
    IC_mean = IC_series['Pearson'].mean()
    IC_std = IC_series['Pearson'].std()
    print(f'IC_mean:{IC_mean}')

    print(f'ICIR:{IC_mean/IC_std}')
    def set_color(test):
        if test==1:
            return 'red'
        else:
            return 'blue'
        
    # if whetherplot == "plot":
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.plot(pnl.index,pnl['G10'], lw = 2, label = '多',color = 'tab:blue')
    ax.plot(pnl.index,pnl['G1'], lw = 2, label = '空',color = 'tab:red')
    ax.set_xlabel("时间（年）", fontsize=10) 
    ax.set_ylabel("净值", fontsize=10)#, color="blue") 
    ax.legend(loc = 'upper left')
    ax.grid()
    plt.title('净值曲线')
    plt.show()

    # fig, ax = plt.subplots(figsize=(10, 6)) 
    # ax.plot(pnl.index,pnl['ls'], lw = 2, label = '多空组合净值',color = 'tab:red')
    # ax.plot(pnl.index,pnl['benchmark'], lw = 2, label= f"{benchmark}", color = 'tab:blue')
    # ax.set_xlabel("时间（年）", fontsize=10) 
    # ax.set_ylabel("净值", fontsize=10)#, color="blue") 
    # ax.legend(loc = 'upper left')
    # ax.grid()
    # plt.show()

    # IC_Series
    color_series = np.abs(IC_series['pvalue']) < 0.2
    color_series = list(map(set_color,color_series))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(IC_series.index,IC_series['Pearson'], color=color_series)
    ax.grid()
    plt.show()


