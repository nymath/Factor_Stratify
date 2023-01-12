import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from os.path import expanduser
import h5py
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import tee
from itertools import compress
current_time = datetime.now().strftime('%Y-%m-%d')
import scipy.stats
def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

if os.name == 'posix':
    plt.rcParams['font.sans-serif'] = ['Songti SC']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False


HOME = expanduser("~")
rawpath = f'{HOME}/.rqalpha/bundle'
if not os.path.exists(rawpath):
    print('数据源路径不存在, 请在terminal中依次输入\n1. pip install rqalpha \n2. rqalpha download-bundle\n完成数据导入')
    
refresh_rate_map = {
    'Monthly(-1)': 'is_monthend',
    'Weekly(-1)': 'is_weekend',
    'Yearly(-1)': 'is_yearend',
}
name2code_index = {
    'SH50': '000016.XSHG',
    'HS300': '000300.XSHG',
    'ZZ500': '000905.XSHG',
    'ZZ800': '000906.XSHG',
    'ZZ1000': '000852.XSHG'
}

def get_TradeDate(start_date: str, 
                  end_date: str) -> pd.core.indexes.datetimes.DatetimeIndex:
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    test = pd.to_datetime(np.array(np.load(f'{rawpath}/trading_dates.npy'),dtype='U8'))
    trade_dates = pd.Series(0,index=test)[start_date:end_date].index
    return trade_dates

def get_TransferDate(start_date: str, 
                  end_date: str, end=True, freq='month') -> pd.core.indexes.datetimes.DatetimeIndex:
    def pairwise(iterable):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    s = get_TradeDate(start_date, end_date)[0]
    e = get_TradeDate(start_date, end_date)[-1]
    if end:
        date_s = pd.to_datetime(s)
        date_e = pd.to_datetime(get_NextTradeDate(current_date=e,period=1))
        trade_date = get_TradeDate(date_s,date_e)
        exec(f'lst = list(map(lambda x:x[0]!=x[1],pairwise(map(lambda x:x.{freq},trade_date))))',locals())
        res = pd.to_datetime(list(compress(trade_date, locals()['lst'])))
        return res
    if not end:
        date_s = pd.to_datetime(get_NextTradeDate(current_date=s,period=-1))
        date_e = pd.to_datetime(e)
        trade_date = get_TradeDate(date_s,date_e)
        exec(f'lst = list(map(lambda x:x[0]!=x[1],pairwise(map(lambda x:x.{freq},trade_date))))',locals())
        res = pd.to_datetime(list(compress(trade_date, locals()['lst'])))
        return res   

def get_ex_factor(secID, start_date, end_date):
    def handle_data(s):
        df = pd.DataFrame(hf[s][:])
        df.iloc[0,0] = '19900101000000'
        df.index = pd.to_datetime(np.array(df['start_date'],dtype='U8'))
        df = df[['ex_cum_factor']]
        true_time = [df.index.asof(t) for t in trade_date] # 这个for循环应该太浪费时间了
        ex_factor = df.iloc[-1,:]/df.loc[true_time]
        ex_factor.index = pd.to_datetime(trade_date)
        return ex_factor
    trade_date = get_TradeDate(start_date, end_date)
    with h5py.File(f'{rawpath}/ex_cum_factor.h5',mode='r') as hf:
        if not isinstance(secID, list):
            ex_factor = handle_data(secID)
        else:
            ex_factor = pd.concat([handle_data(s) for s in secID],axis=1)
            ex_factor.columns = secID
    return ex_factor


def get_NextTradeDate(current_date: str,period:int = -1) ->pd._libs.tslibs.timestamps.Timestamp:
    # 目前的版本要求current_date是一个tradedate
    current_date =  pd.to_datetime(current_date)
    shifts = 3*period + int(np.sign(period)*10)
    next = current_date + timedelta(days=shifts)
    days = get_TradeDate(min(current_date,next),max(current_date, next))
    index = np.where(days == current_date)[0] + period
    result = days[int(index)]
    return result

def get_data_indexes(secID: str, fields: str,
                       start_date: str, end_date: str):
    '''
    secID: 传入标准的指数代码, 例如 000300.XSHG
    fields: open, high, low, close, volumn, total_turnover
    '''
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    with h5py.File(f'{rawpath}/indexes.h5',mode='r') as hf:
        df = pd.DataFrame(hf[secID][:])
    data = df[fields]
    data.index = pd.to_datetime(np.array(df['datetime'],dtype='U8'))
    df = data.loc[start_date:end_date]
    return df

def get_data_stocks(secID, fields, start_date, end_date, exclude_ds=False):
    def handle_data(s,field):
        df = pd.DataFrame(hf[s][:])
        data = df.loc[:,field]
        data.index = pd.to_datetime(np.array(df['datetime'],dtype='U8'))
        return pd.DataFrame(data.values,index=data.index,columns=[s])
    with h5py.File(f'{rawpath}/stocks.h5',mode='r') as hf:
        if (not isinstance(secID, list)) and (not isinstance(fields, list)):
            data = handle_data(secID,fields)
            data.loc[start_date:end_date]
        elif (isinstance(secID, list)) and (not isinstance(fields, list)):
            data = pd.concat([handle_data(s,fields) for s in secID],axis=1)
            data.columns = secID
            data = data.loc[start_date:end_date]
        elif (not isinstance(secID, list)) and (isinstance(fields, list)):
            data = pd.concat([handle_data(secID,f) for f in fields],axis=1)
            data.columns = fields
            data = data.loc[start_date:end_date]
        else: 
            data = {}
            for f in fields:
                data[f] = pd.concat([handle_data(s,f) for s in secID],axis=1)
                data[f].columns = secID
                data[f] = data[f].loc[start_date:end_date]
        return data


def get_data_index_components(secID='000300.XSHG', current_date=None) -> list:
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./data/index_components'):
        os.mkdir('./data/index_components')
    try:
        res = pd.read_csv(f'./data/index_components/{secID}.csv',index_col=0)
        res.index = pd.to_datetime(res.index)
        truetime = res.index.asof(f'{current_date}')
        return list(res.loc[truetime])
    except:
        print('本地数据不存在, 尝试下载')
        res = pd.read_csv(f'https://raw.githubusercontent.com/nymath/financial_data/main/data/index_components/{secID}.csv',index_col=0,encoding='utf8')
        res.index = pd.to_datetime(res.index)
        print('下载成功')
        res.index = pd.to_datetime(res.index)
        res.to_csv(f'./data/index_components/{secID}.csv',encoding='utf8')
        print('保存成功')
        truetime = res.index.asof(f'{current_date}')
        return list(res.loc[current_date])

def get_data_InterestRate(start_date: str,
                          end_date: str,period: str ='10Y') -> pd.Series:
    with h5py.File(f'{rawpath}/yield_curve.h5',mode='r') as hf:
        df = pd.DataFrame(hf['data'][:])
    data = df[period]
    data.index = pd.to_datetime(np.array(df['date'],dtype='U8'))
    trade_date = get_TradeDate(start_date, end_date)
    df = data[trade_date]
    return df


class RiskAssessor(object):

    def __init__(self, simple_return_series, simple_benchmark_series, simple_riskfree_series, freq = 252, logarithm = False):
        try:
            self.name = simple_return_series.name
        except:
            print("最好给你的策略起一个名字")
        if logarithm:
            self.rp = np.log(1+simple_return_series)
            self.rm = np.log(1+simple_benchmark_series)
            self.rf = np.log(1+simple_riskfree_series)
        else:
            self.rp = simple_return_series
            self.rm = simple_benchmark_series
            self.rf = simple_riskfree_series
            
        self.freq = freq
        self.pnl = (1+simple_return_series).cumprod()
        
        self._cal_annulized_info()
        self._cal_regression_based_info()
        self._cal_maxdrawdown(self.pnl)

    def __getattr__(self):
        pass
    
    def _cal_annulized_info(self):
        """
        Calculate the annualized return, the volatility and the Sharpe ratio.
        """
        self.rf_a_mean = self.rf.mean() * self.freq
        self.rp_a_mean = self.rp.mean() * self.freq
        self.rp_a_std = self.rp.std() * np.sqrt(self.freq)
        self.SharpeRatio = (self.rp_a_mean - self.rf_a_mean) / self.rp_a_std
    
    def _cal_regression_based_info(self):
        """
        Calculate annualized alpha, sigmahat and beta.
        """
        X = self.rm - self.rf
        X = sm.add_constant(X,prepend=True)
        Y = self.rp - self.rf
        model = sm.OLS(Y,X)
        res = model.fit()
        self.reg_alpha = np.array(res.params)[0] * self.freq
        self.reg_beta = np.array(res.params)[1]
        self.reg_sigmahat = np.sqrt(res.mse_resid) * np.sqrt(self.freq)
        self.reg_infomation_ratio = self.reg_alpha / self.reg_sigmahat
    
    def _cal_maxdrawdown(self, pnl):
        """
        Calculate the largest peak-to-through drawdown of the PnL curve as well as 
        the duration of the drawdown. Requires that the pnl_returns is a Pandas Series.
        Parameters:
        pnl: A pandas Series representing the PnL curve
        """
        hwm = [0]
        idx = pnl.index
        self.drawdown = pd.Series(index=idx, dtype='float64')
        self.duration = pd.Series(index=idx, dtype='float64')
        for t in range(1, len(idx)):
            hwm.append(max(hwm[t-1], pnl[t]))
            self.drawdown[t] = (hwm[t]-pnl[t])
            self.duration[t] = (0 if self.drawdown[t] == 0 else self.duration[t-1]+1)
        self.drawdownmax = self.drawdown.max()
        self.durationmax = self.duration.max()
    




class MultiRiskAssessor(RiskAssessor):
    pass
    
    

def RiskAssessment(simple_return_series: pd.DataFrame,
                   simple_riskfree_series: pd.DataFrame) -> dict:
    trade_date = simple_return_series.index.to_list()
    riskfree = simple_riskfree_series.loc[trade_date]
    portfolio_names0 = simple_return_series.columns
    portfolio_names1 = portfolio_names0.to_list().copy()
    portfolio_names1.remove('benchmark')
    
    template0 = pd.Series(data=0,index=portfolio_names0)
    template1 = pd.Series(data=0,index=portfolio_names1)
    
    log_return = np.log(1+simple_return_series)
    
    pnl = (1+simple_return_series).cumprod()
    assess = dict()
    # mean return  
    all_log_return_mean = log_return.mean() * 252
    all_log_return_std = log_return.std() * np.sqrt(252)
    assess['annulized_return'] = all_log_return_mean
    assess['annulized_volatility'] = all_log_return_std
    
    # Sharpe ratio
    all_riskfree_mean = riskfree.mean() * 252
    all_log_return_Sharpe = ( all_log_return_mean - np.array(all_riskfree_mean))/ all_log_return_std
    assess['sharpe'] = all_log_return_Sharpe
    
    # max_drawdown
    MDD_list = template0.copy()
    for s in portfolio_names0:
        temp_list = pnl[s]
        MDD_list[s] = np.max((temp_list.cummax() - temp_list)/temp_list.cummax())
    assess['max_drawdown'] = MDD_list

    # Alpha Beta Information Ratio
    X = log_return['benchmark'].values.reshape(-1,1) - riskfree.values.reshape(-1,1)
    X = sm.add_constant(X,prepend=True)
    Y = log_return.loc[:,portfolio_names1] - riskfree.values.reshape(-1,1)
    beta_list = template1.copy()
    alpha_list = template1.copy()
    sigma_hat_list = template1.copy()
    for s in portfolio_names1:
        model = sm.OLS(Y[s],X)
        res = model.fit()
        alpha_list[s] = res.params[0] 
        beta_list[s] = res.params[1]
        sigma_hat_list[s] = np.sqrt(res.mse_resid)
    alpha_list = alpha_list * 252 
    sigma_hat_list = sigma_hat_list * np.sqrt(252) 
    assess['alpha'] = alpha_list
    assess['beta'] = beta_list
    assess['information_ratio'] = alpha_list / sigma_hat_list
    return assess

def cal_ReturnSeries(assets_series: pd.DataFrame, weight_series: pd.DataFrame) -> pd.Series:
                    #  trade_date: pd.core.indexes.datetimes.DatetimeIndex, 
                    #  transfer_date: pd.core.indexes.datetimes.DatetimeIndex
    trade_date = assets_series.index.to_list()
    transfer_date = weight_series.index.to_list()
    simple_return_series = pd.Series(index=trade_date, name='test', dtype='float64')
    
    for i in range(len(transfer_date)):
        if i < len(transfer_date)-1:     
            current_time = transfer_date[i]
            next_time = transfer_date[i+1]
        else:
            current_time = transfer_date[i]
            next_time = trade_date[-1]   
        assets_data = assets_series.loc[current_time:next_time]
        assets_pnl = assets_data / assets_data.iloc[0,:]
        future_portfolio_return = (assets_pnl * np.array(weight_series.loc[current_time]).reshape(1,-1)).sum(axis=1) # 用一次broadcast
        future_portfolio_return = future_portfolio_return / future_portfolio_return.shift(1) - 1
        future_portfolio_return.dropna(inplace=True)
        simple_return_series[future_portfolio_return.index] = future_portfolio_return     
    # zero padding
    simple_return_series.loc[transfer_date[0]] = 0
    simple_return_series.dropna(inplace=True)
    temp = pd.Series(simple_return_series,index=trade_date)
    temp.fillna(0,inplace=True)
    return temp


def StratifiedBackTest(assets_series, factor_series, transfer_date, n=5):
    '''
    单因子分层回测
    '''
    num_assets = assets_series.shape[1]
    num_base = int(num_assets/n)
    portfolios_name = [f'G{i}' for i in range(1,n+1)]
    template = pd.DataFrame(0,index=transfer_date,columns=assets_series.columns, dtype='float64')
    dict_weight_series = {}
    for s in portfolios_name:
        dict_weight_series[s] = template.copy()
    for i in range(len(transfer_date)):
        current_time = transfer_date[i]
        print(current_time)
        factor_current = factor_series.loc[current_time,:]
        for j in range(1,n+1):
            if j < n:
                temp_componets = list(factor_current.sort_values(ascending=True)[(j-1)*num_base:j*num_base].index)
            else:
                temp_componets = list(factor_current.sort_values(ascending=True)[(j-1)*num_base:].index)
            dict_weight_series[f'G{j}'].loc[current_time,temp_componets] = 1 / len(temp_componets)
            
    simple_return_series = pd.concat((cal_ReturnSeries(assets_series,dict_weight_series[s]) for s in dict_weight_series.keys()),axis=1)
    simple_return_series.columns = dict_weight_series.keys()
    return dict_weight_series, simple_return_series


def cal_IC_series(assets_series: pd.DataFrame, factor_series: pd.DataFrame, 
                  transfer_date: pd.DatetimeIndex) -> pd.Series:
    IC_series = pd.DataFrame(columns=['Pearson','pvalue'], dtype='float64')
    for i in range(len(transfer_date)-1):
        current_time = transfer_date[i]
        next_time = transfer_date[i+1]
        factor_current = factor_series.loc[current_time,:]
        next_monthly_return = assets_series.loc[next_time] / assets_series.loc[current_time] - 1
        res = scipy.stats.pearsonr(factor_current,next_monthly_return)
        IC_series.loc[current_time] = np.array(res)
    return IC_series


def cal_turnover(weight_series: pd.DataFrame) -> np.ndarray:
    weight_series_l1 = weight_series.shift(1)
    weight_series_l1.fillna(0,inplace=True)
    _ = np.abs(weight_series-weight_series_l1).mean(axis=0).sum()*250
    return _

# 接下来就剩一个因子库构建了


# 海龟交易系统搭建
# class VirtualAccount(__builtin__.object):
#     def __init__(self):
#         pass
#     self.current_date
#     self.previous_date
    
#     def get_universe(self, asset_type, exclude_halt=False):
#         pass
    
# class RiskAssessment():
#     def __init__(self):
#         pass


# def frequency_convert(multi_series, frequency_list):
#     try:
#         if '__iter__' in dir(frequency_list):
#             frequency_list = list(iter(frequency_list))
#             if multi_series.shape[0] < len(frequency_list):
#                 xx = pd.Series(data=range(len(frequency_list)),index=frequency_list)
#                 temp = pd.concat([multi_series,xx],axis=1)
#                 temp = temp.fillna(method='ffill').dropna()
#                 return temp.loc[:,list(multi_series.columns)]
#             else:
#                 return 0
#         else:
#             print(f'ERROR: {frequency_list} is not iterable.')
#     except:
#         print('请传入DataFrame以及iterator')


# cond = {}
# cond['synthesis'] = """
# factor_current = factors_series['pred_pe_yoy'].loc[current_time,:] # 不用合成
# """
# cond['select'] = """
# assets_selected = list(factor_current.sort_values(ascending=True)[-3:].index) # 预测值最大的三个行业
# """
# cond['optimize'] = """
# weight_series.loc[current_time,assets_selected] = 1 / len(assets_selected)
# """

# def cal_WeightSeries(assets_series: pd.DataFrame, factors_series: dict,
#                      transfer_date: list, cond:dict, disp = False) -> pd.DataFrame:
#     '''
#     获得一个选股策略的权重
#     '''
#     weight_series = pd.DataFrame(0,index=transfer_date,columns=assets_series.columns, dtype='float64')
#     for i in range(len(transfer_date)):
#         # 获取当前时间
#         current_time = transfer_date[i]
#         # 获取当前股票池
#         current_stocks = assets_series.columns.to_list()
#         # 获取因子数据
        
#         ## 因子中性化(neutralize) 去极值(winsorize)，标准化(standardize)
#         if disp:
#             print(current_time)
#         # 因子合成
#         exec(cond['synthesis'])
        
#         # 选股: $ (factor_current, cond) \mapto assets_selected $
#         exec(cond['select']) 
        
#         # 配权: $ (assets_selected, alpha_factor, CovMatrix, RiskAdverse, optimize_mothod) \mapsto  weight_vector $
#         exec(cond['optimize'])
        
#     return weight_series
