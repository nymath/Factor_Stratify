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
from trade import self_uqer
from trade.self_uqer import refresh_rate_map
from trade.self_uqer import RiskAssessment
from trade.self_uqer import cal_turnover
from trade.self_uqer import cal_ReturnSeries
from trade.self_uqer import get_data_indexes
from trade.self_uqer import get_data_InterestRate
from trade.self_uqer import get_TradeDate
from trade.self_uqer import get_TransferDate
from trade.self_uqer import get_NextTradeDate
from trade.self_uqer import StratifiedBackTest
from trade.self_uqer import cal_IC_series

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

# try:
#     whetherplot = sys.argv[4]
# except:
#     whetherplot = "plot"
    
# freq = 'd'  
# refresh_rate = 'Monthly(-1)' 


#%% Data Loading 
## 获取申万一级行业的日度收盘数据

assets_series = pd.read_csv('./data/index_industry_d.csv',index_col=0)
assets_series.index = pd.to_datetime(assets_series.index)
assets_series = assets_series.loc[start:end]
assets_names = assets_series.columns.to_list()

## 获取benchmark_series指数日度收盘数据

benchmark_series = get_data_indexes(benchmark, 'close', start, end)
benchmark_series = benchmark_series.loc[start:end]

## 获取无风险收益率
rf_return_series = get_data_InterestRate(start,end,period='10Y') / 252
rf_return_series.fillna(method='ffill',inplace=True)

## 交易日数据
trade_date = get_TradeDate(start, end)
transfer_date = get_TransferDate(start, end, end=True, freq='month')

# n2d = pd.Series(data=trade_date,index=range(len(trade_date))) # 输入number 得到 date
# d2n = pd.Series(data=range(len(trade_date)),index=trade_date) # 输入date 得到 number


### /* 2015-11-30 当天有因子数据, 2022-02-28为最后一个交易日 */
## 单因子读取
# 确保因子的columns和assets的columns必须对应
factors = dict()
factor = pd.read_csv('./factors/industry_sarimaxANDsarima_predictions.csv',index_col=0)
factor = factor.loc[start:end]
factor = factor.loc[:,assets_names]
factor.index = [transfer_date.asof(x) for x in factor.index] # 转换为真实的交易日
factor.index = pd.to_datetime(factor.index)

temp = pd.DataFrame(data=factor, index=trade_date) # 转换因子频率, 试试reindex
temp.fillna(method='ffill', inplace=True)

####################!!!!!!!!!!!!!!!!!!!!################
temp.dropna(inplace=True) # 我觉得没有必要dropna()
factors['pred_pe_yoy'] = temp

#%% 分层回测
IC_series = cal_IC_series(assets_series, factors['pred_pe_yoy'], transfer_date)  
assets_series.index
factors['pred_pe_yoy'].index
dict_weight_series, simple_return_series = StratifiedBackTest(assets_series,factors['pred_pe_yoy'],transfer_date,n=10)

temp = benchmark_series / benchmark_series.shift(1) - 1
temp.fillna(0,inplace=True)
simple_return_series['benchmark'] = temp

color_lst = []
fig, ax = plt.subplots(figsize=(8,5))
for i in simple_return_series.columns[:-1]:
    ax.plot((1+simple_return_series[i]).cumprod(), label=i)
ax.grid()
ax.legend(bbox_to_anchor=(1.01, 1),loc='upper left',borderaxespad=0.)
ax.set_title('因子分层测试, 升序分组')
ax.set_ylabel('净值')
plt.tight_layout()
plt.show()

#%% 相关评判指标计算
simple_return = simple_return_series
log_return = np.log(1+simple_return) 

test = RiskAssessment(simple_return, rf_return_series)
result1 = self_uqer.RiskAssessor(simple_return['G10'], simple_return['benchmark'], rf_return_series)

result1.durationmax

## 各年收益率
annual_log_return_mean = log_return.resample('y').mean() * 252
annual_log_return_std = log_return.resample('y').std() * np.sqrt(252)
annual_riskfree_mean = rf_return_series.resample('y').mean() * 252
all_riskfree_mean = rf_return_series.mean() * 252

## 总收益率
all_log_return_mean = log_return.mean() * 252
all_log_return_std = log_return.std() * np.sqrt(252)
print(f'收益率为\n{all_log_return_mean} \n标准差为 \n{all_log_return_std}')
# 相关评测指标计算
## 净值
pnl = np.exp(log_return).cumprod()
pnl.loc[transfer_date[0]] = np.ones(pnl.shape[1])
pnl.sort_index(inplace=True)

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

#%% 测试功能
# 权重变换 & 交易成本测算




