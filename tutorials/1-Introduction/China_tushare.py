import warnings

warnings.filterwarnings("ignore")
import sys
import datetime
import argparse
import pandas as pd
from IPython import display
from meta import config
from meta.data_processor import DataProcessor
from main import check_and_make_directories
from meta.data_processors.tushare import Tushare, ReturnPlotter
from meta.env_stock_trading.env_stocktrading_China_A_shares import (
    StockTradingEnv,
)
from agents.stablebaselines3_models import DRLAgent
import os
from typing import List
from argparse import ArgumentParser
from meta import config
from meta.config_tickers import DOW_30_TICKER
from meta.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_BASE_URL,
)
import pyfolio
from pyfolio import timeseries
from data_download import get_daily_stock_and_indicator, INDICATORS

pd.options.display.max_columns = None

print("ALL Modules have been imported!")


### Create folders

import os

"""
use check_and_make_directories() to replace the following

if not os.path.exists("./datasets"):
    os.makedirs("./datasets")
if not os.path.exists("./trained_models"):
    os.makedirs("./trained_models")
if not os.path.exists("./tensorboard_log"):
    os.makedirs("./tensorboard_log")
if not os.path.exists("./results"):
    os.makedirs("./results")
"""
def prepare_dir():
    check_and_make_directories(
        [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
    )


def download_data(TRAIN_START_DATE, TRADE_END_DATE, mini=False, stock_config={}):
    """
    data_is: test, longtou, growth
    获取数据
    """
    ### Download data, cleaning and feature engineering
    # At Oct.22 2022, trade date available span is [2020-04-22, 2022-10-21]
    TIME_INTERVAL = "1d"
    kwargs = {}
    kwargs["token"] = "27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5"
    p = DataProcessor(
        data_source="tushare",
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        time_interval=TIME_INTERVAL,
        **kwargs,
    )
    if stock_config["stock_type"] == "test":
        ticker_list = [
        "600000.SH",
        "600009.SH",
        "600016.SH",
        "600028.SH",
        "600030.SH",
        "600031.SH",
        "600036.SH",
        "600050.SH",
        "600104.SH",
        "600196.SH",
        "600276.SH",
        "600309.SH",
        "600519.SH",
        "600547.SH",
        "600570.SH",
    ]
        # download and clean
        p.download_data(ticker_list=ticker_list)
    else:
        data_df = get_daily_stock_and_indicator(TRAIN_START_DATE,TRADE_END_DATE,mini,stock_config=stock_config)
        p.dataframe = data_df
    p.clean_data()
    # add_technical_indicator
    p.add_technical_indicator(config.INDICATORS)
    p.clean_data()
    print(f"p.dataframe: {p.dataframe}")

    return p


def preprocess_data(p):
    ### Split traning dataset

    train = p.data_split(p.dataframe, TRAIN_START_DATE, TRAIN_END_DATE)
    print(f"len(train.tic.unique()): {len(train.tic.unique())}")

    print(f"train.tic.unique(): {train.tic.unique()}")

    print(f"train.head(): {train.head()}")

    print(f"train.shape: {train.shape}")

    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension * (len(config.INDICATORS) + 2) + 1
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    return train, stock_dimension,state_space

def setup_env(train_data, stock_dimension,state_space):
    ### Train
    env_kwargs = {
        "stock_dim": stock_dimension,
        "hmax": 1000,
        "initial_amount": 1000000,
        "buy_cost_pct": 6.87e-5,
        "sell_cost_pct": 1.0687e-3,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dimension,
        "tech_indicator_list": config.INDICATORS,
        "print_verbosity": 1,
        "initial_buy": True,
        "hundred_each_trade": True,
    }

    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)

    ## DDPG

    env_train, _ = e_train_gym.get_sb_env()
    print(f"print(type(env_train)): {print(type(env_train))}")
    return env_train, env_kwargs

def ddpg(env_train, total_timesteps=10000):
    agent = DRLAgent(env=env_train)
    DDPG_PARAMS = {
        "batch_size": 256,
        "buffer_size": 50000,
        "learning_rate": 0.0005,
        "action_noise": "normal",
    }
    POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
    model_ddpg = agent.get_model(
        "ddpg", model_kwargs=DDPG_PARAMS, policy_kwargs=POLICY_KWARGS
    )

    trained_ddpg = agent.train_model(
        model=model_ddpg, tb_log_name="ddpg", total_timesteps=total_timesteps
    )
    return trained_ddpg

## A2C
def a2c(env_train,total_timesteps=50000):
    agent = DRLAgent(env=env_train)
    model_a2c = agent.get_model("a2c")

    trained_a2c = agent.train_model(
        model=model_a2c, tb_log_name="a2c", total_timesteps=total_timesteps
    )
    return trained_a2c

def ppo(env_train, total_timesteps):
    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

    trained_ppo = agent.train_model(model=model_ppo,
                                    tb_log_name='ppo',
                                    total_timesteps=total_timesteps)
    return trained_ppo

def td3(env_train, total_timesteps):
    agent = DRLAgent(env = env_train)
    TD3_PARAMS = {"batch_size": 100,
                  "buffer_size": 1000000,
                  "learning_rate": 0.001}

    model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)


    trained_td3 = agent.train_model(model=model_td3,
                                 tb_log_name='td3',
                                 total_timesteps=total_timesteps)
    return trained_td3


def sac(env_train, total_timesteps=30000):
    # ### Agent 5: SAC
    agent = DRLAgent(env=env_train)
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 1000000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }

    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

    trained_sac = agent.train_model(model=model_sac,
                                    tb_log_name='sac',
                                    total_timesteps=total_timesteps)
    return trained_sac

def trade_test_data(trained_model, data_p,stock_dimension,state_space):
    ### Trade
    trade = data_p.data_split(data_p.dataframe, TRADE_START_DATE, TRADE_END_DATE)
    env_kwargs = {
        "stock_dim": stock_dimension,
        "hmax": 1000,
        "initial_amount": 1000000,
        "buy_cost_pct": 6.87e-5,
        "sell_cost_pct": 1.0687e-3,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dimension,
        "tech_indicator_list": config.INDICATORS,
        "print_verbosity": 1,
        "initial_buy": False,
        "hundred_each_trade": True,
    }
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_model, environment=e_trade_gym
    )

    df_actions.to_csv("action.csv", index=False)
    print(f"df_actions: {df_actions}")
    return df_account_value, df_actions, trade

def backtest(df_account_value, trade_data, result_file):
    ### Backtest
    # matplotlib inline
    from matplotlib import pyplot as plt

    plt.clf()
    plotter = ReturnPlotter(df_account_value, trade_data, TRADE_START_DATE, TRADE_END_DATE)
    plotter.plot_all()

    plt.clf()
    plotter.plot()

    # matplotlib inline
    # # ticket: SSE 50：000016
    plt.clf()
    plotter.plot("000016")

    #### Use pyfolio

    # CSI 300
    baseline_df = plotter.get_baseline("399300")


    daily_return = plotter.get_return(df_account_value)
    daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(
        returns=daily_return,
        factor_returns=daily_return_base,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print("==============DRL Strategy Stats===========")
    print(f"perf_stats_all: {perf_stats_all}")


    daily_return = plotter.get_return(df_account_value)
    daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(
        returns=daily_return_base,
        factor_returns=daily_return_base,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print("==============Baseline Strategy Stats===========")

    print(f"perf_stats_all: {perf_stats_all}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="强化学习预测")
    parser.add_argument('-m', '--model', type=str, default="sac",choices=("sac","ppo","a2c","ddpg","td3","all", "ensemble") ,help='使用哪个模型进行训练')
    parser.add_argument('-st', '--start_train', default='2010-01-01', help='训练的开始时间')
    parser.add_argument('-et', '--end_train', default='2020-05-31', help='训练的结束时间')
    parser.add_argument('-se', '--start_test', default='2020-06-01', help='测试的开始时间')
    parser.add_argument('-ee', '--end_test', default='2022-05-31', help='测试的结束时间')
    parser.add_argument('-t', '--timesteps', type=int, default=30000, help='训练的时间步')
    parser.add_argument('-mi', '--mini', action='store_true', help='迷你数据集')
    parser.add_argument('-so', '--stock_type',  type=str, default="growth", help='股票的类型，支持,growth,longtou, test')
    parser.add_argument('-sn', '--stock_num', type=int, default=20, help='股票数量，默认20')
    args = parser.parse_args()
    # Step3 下载数据集
    TRAIN_START_DATE = args.start_train
    TRAIN_END_DATE = args.end_train
    TRADE_START_DATE = args.start_test
    TRADE_END_DATE = args.end_test
    model = args.model
    prepare_dir()
    print(f"训练日期是: {TRAIN_START_DATE} 到 {TRAIN_END_DATE}, 预测日期是: {TRADE_START_DATE} 到 {TRADE_END_DATE}")
    print(f"使用的模型是: {model}")
    stock_config = {"stock_type": args.stock_type, "stock_num": args.stock_num}
    data_p = download_data(TRAIN_START_DATE, TRADE_END_DATE, mini=args.mini, stock_config=stock_config)
    train_data, stock_dimension,state_space= preprocess_data(data_p)
    env_train, env_kwargs = setup_env(train_data,stock_dimension,state_space)
    if model == "sac":
        trained_model = sac(env_train, total_timesteps=args.timesteps)
    elif model =="td3":
        trained_model = td3(env_train, total_timesteps=args.timesteps)
    elif model =="ppo":
        trained_model = ppo(env_train, total_timesteps=args.timesteps)
    elif model =="a2c":
        trained_model = a2c(env_train)
    elif model =="ddpg":
        trained_model = ddpg(env_train)
    elif model == "all":
        for model_name in ["sac","td3","ppo","a2c","ddpg"]:
            print(f"使用的模型是: {model_name}")
            model_func = eval(model_name)
            trained_model = model_func(env_train, total_timesteps=args.timesteps)
            df_account_value, df_actions, trade_data = trade_test_data(trained_model, data_p,stock_dimension,state_space)
            now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
            trade_action_file = f"action_{model_name}_{now}.xlsx"
            df_actions.to_excel(trade_action_file, index=False)
            # 缓存df_account_value到本地
            df_account_value_pkl_file = "cache/df_account_value.pkl"
            df_account_value.to_pickle(df_account_value_pkl_file)
            csv_file = f"backtest_{model_name}_{now}.xlsx"
            backtest(df_account_value, trade_data, result_file=csv_file)
        print(f"结束所有模型的训练学习")
        sys.exit(0)
    elif model == "ensemble":
        print(f"进行组合式模型的训练")
        raise NotImplementedError("暂未完成")
        # df_account_value, df_summary = ensemble_model(processed=processed_full)
        # 缓存df_account_value到本地
        now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
        csv_file = f"backtest_{model}_{now}.xlsx"
        backtest(df_account_value, result_file=csv_file)
        sys.exit(0)
    else:
        print(f"不支持的模型,退出")
        sys.exit(0)
    df_account_value, df_actions, trade_data = trade_test_data(trained_model, data_p,stock_dimension,state_space)
    # 缓存df_account_value到本地
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    trade_action_file = f"action_{model}_{now}.xlsx"
    df_actions.to_excel(trade_action_file, index=False)
    csv_file = f"backtest_{model}_{now}.xlsx"
    backtest(df_account_value, trade_data, result_file=csv_file)