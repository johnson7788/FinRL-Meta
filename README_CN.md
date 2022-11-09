# FinRL-Meta: A Universe of Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning


[![Downloads](https://pepy.tech/badge/finrl_meta)](https://pepy.tech/project/finrl_meta)
[![Downloads](https://pepy.tech/badge/finrl_meta/week)](https://pepy.tech/project/finrl_meta)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl_meta.svg)](https://pypi.org/project/finrl_meta/)

FinRL Meta（[文档网站](https://finrl.readthedocs.io/en/latest/finrl_meta/background.html))为数据驱动的金融强化学习构建市场环境。我们的目标是帮助社区中的用户轻松构建环境。

1.FinRL Meta提供了数百种市场环境。
2.FinRL Meta复制现有论文作为基准。
3.FinRL Meta提供数十个演示/教程，以课程形式组织。

安装talib:
brew install ta-lib
pip install ta-lib

Previously called **Neo_FinRL**: **N**ear real-market **E**nvironments f**o**r data-driven **Fin**ancial **R**einforcement **L**earning.

## Outline
- [News and Tutorials](#news-and-tutorials)
- [Our Goals](#our-goals)
- [Design Principles](#design-principles)
- [Overview](#overview)
- [Plug-and-Play](#plug-and-play)
- [Training-Testing-Trading](#training-testing-trading-pipeline)
- [Our Vision](#our-vision)

## News and Tutorials

+ [MLearning.ai] [Financial Metaverse as a Playground for Financial Machine Learning](https://medium.com/@zx2325/finrl-meta-from-market-environments-to-a-financial-metaverse-5db8490a83df)
+ [DataDrivenInvestor] [FinRL-Meta: A Universe of Near Real-Market En­vironments for Data­-Driven Financial Reinforcement Learning](https://medium.datadriveninvestor.com/finrl-meta-a-universe-of-near-real-market-en-vironments-for-data-driven-financial-reinforcement-e1894e1ebfbd)
+ [深圳特区报] [深港两城深度融合 用“创新”续写“春天的故事”](https://www.sh-stic.com/news_72/515.html) [香港特别行政区联络办公室](http://www.locpg.gov.cn/jsdt/2022-06/06/c_1211654176.htm)
+ [央广网] [2021 IDEA大会于福田圆满落幕：群英荟萃论道AI 多项目发布亮点纷呈](http://tech.cnr.cn/techph/20211123/t20211123_525669092.shtml)
+ [央广网] [2021 IDEA大会开启AI思想盛宴 沈向洋理事长发布六大前沿产品](https://baijiahao.baidu.com/s?id=1717101783873523790&wfr=spider&for=pc)
+ [IDEA新闻] [2021 IDEA大会发布产品FinRL-Meta——基于数据驱动的强化学习金融风险模拟系统](https://idea.edu.cn/news/20211213143128.html)
+ [知乎] [FinRL-Meta基于数据驱动的强化学习金融元宇宙](https://zhuanlan.zhihu.com/p/437804814)

## Our Goals
+为了提供基准并促进公平比较，我们允许研究人员在同一数据集上评估不同的策略。
此外，这将有助于研究人员更好地理解DRL算法的“黑盒”性质（基于深度神经网络）。
+为了减少模拟现实的差距：现有的工作使用历史数据的回溯测试，而实际的表现可能截然不同。
+减少数据预处理负担，使量化人员能够专注于制定和优化策略。

## Design Principles
+**即插即用（PnP）**：模块化；处理不同的市场（例如T0与T+1）
+**完整性和通用性：**多个市场；各种数据源（API、Excel等）；用户友好变量。
+**层结构和可扩展性
**：三层，包括：数据层、环境层和agent层。层通过端到端接口进行交互，实现高扩展性。
+**“训练-测试-交易”pipeline**：模拟训练和连接实时API进行测试/交易，缩小模拟真实差距。
+**高效的数据采样**：加快数据采样过程是DRL训练的关键！来自ElegantRL项目。我们知道多处理对于减少训练时间（CPU+GPU之间的调度）非常有效。
+**透明度**：上层不可见的虚拟环境+**灵活性和可扩展性**：继承在这里可能有帮助

## Overview
![Overview image of FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/figs/neofinrl_overview.png)
我们在FinRL Meta中使用了分层结构，如上图所示，它由三层组成：数据层、环境层和agent层。每一层都执行其功能，并且是独立的。同时，各层通过端到端接口进行交互，以实现算法交易的完整工作流。此外，层结构允许轻松扩展用户定义的功能。

## DataOps
DataOps将精益开发和DevOps的理念应用于数据分析领域。公司和组织已经开发了DataOps实践，以提高数据分析的质量和效率。
这些实现整合了各种数据源，统一并自动化了数据分析pipeline，包括数据访问、清理、分析和可视化。
然而，DataOps方法尚未应用于金融强化学习研究。
大多数研究人员以个案的方式访问数据、清理数据并提取技术指标（特征），这涉及繁重的人工工作，可能无法保证数据质量。
为了处理金融大数据（非结构化），我们遵循DataOps范式，并在下图中实现了一个自动pipeline：任务规划、数据处理、训练测试交易和监控agent的性能。
通过这条pipeline，我们不断在动态市场数据集上制作DRL基准。

<div align="center">
<img align="center" src=figs/finrl_meta_dataops.png width="80%">
</div>


Supported Data Sources:
|Data Source |Type |Range and Frequency |Request Limits|Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|[Alpaca](https://alpaca.markets/docs/introduction/)| US Stocks, ETFs| 2015-now, 1min| Account-specific| OHLCV| Prices&Indicators|
|[Baostock](http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3)| CN Securities| 1990-12-19-now, 5min| Account-specific| OHLCV| Prices&Indicators|
|[Binance](https://binance-docs.github.io/apidocs/spot/en/#public-api-definitions)| Cryptocurrency| API-specific, 1s, 1min| API-specific| Tick-level daily aggegrated trades, OHLCV| Prices&Indicators|
|[CCXT](https://docs.ccxt.com/en/latest/manual.html)| Cryptocurrency| API-specific, 1min| API-specific| OHLCV| Prices&Indicators|
|[IEXCloud](https://iexcloud.io/docs/api/)| NMS US securities|1970-now, 1 day|100 per second per IP|OHLCV| Prices&Indicators|
|[JoinQuant](https://www.joinquant.com/)| CN Securities| 2005-now, 1min| 3 requests each time| OHLCV| Prices&Indicators|
|[QuantConnect](https://www.quantconnect.com/docs/home/home)| US Securities| 1998-now, 1s| NA| OHLCV| Prices&Indicators|
|[RiceQuant](https://www.ricequant.com/doc/rqdata/python/)| CN Securities| 2005-now, 1ms| Account-specific| OHLCV| Prices&Indicators|
|[Tushare](https://tushare.pro/document/1?doc_id=131)| CN Securities, A share| -now, 1 min| Account-specific| OHLCV| Prices&Indicators|
|[WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/)| US Securities| 2003-now, 1ms| 5 requests each time| Intraday Trades|Prices&Indicators|
|[YahooFinance](https://pypi.org/project/yfinance/)| US Securities| Frequency-specific, 1min| 2,000/hour| OHLCV | Prices&Indicators|


OHLCV: open, high, low, and close prices; volume

adjusted_close: adjusted close price

Technical indicators users can add: 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'. Users also can add their features.


## Plug-and-Play (PnP)
In the development pipeline, we separate market environments from the data layer and the agent layer. A DRL agent can be directly plugged into our environments. Different agents/algorithms can be compared by running on the same benchmark environment for fair evaluations.

支持以下DRL库：9
+ [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL): Lightweight, efficient and stable DRL implementation using PyTorch.
+ [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3): Improved DRL algorithms based on OpenAI Baselines.
+ [RLlib](https://github.com/ray-project/ray): An open-source DRL library that offers high scalability and unified APIs.

A demonstration notebook for plug-and-play with ElegantRL, Stable Baselines3 and RLlib: [Plug and Play with DRL Agents](https://colab.research.google.com/github/AI4Finance-Foundation/FinRL-Meta/blob/main/Demo_Plug_and_Play_with_DRL_Libraries.ipynb)

## "Training-Testing-Trading" Pipeline

<div align="center">
<img align="center" src=figs/timeline.png width="80%">
</div>

我们采用训练测试交易pipeline。首先，在训练数据集中训练DR  Lagent，并在测试数据集中微调（调整超参数）。
然后，对agent进行回溯测试（在历史数据集上），或在纸质/实时交易市场中进行存款。
该pipeline通过将训练/测试和交易期分开来解决**信息泄露问题**。
这种统一的pipeline还允许不同算法之间的公平比较。

## Our Vision
对于未来的工作，我们计划构建一个基于多agent的市场模拟器，该模拟器由超过一万个agent组成，即FinRL Metaverse。
首先，FinRL Metaverse旨在构建一个市场环境的宇宙，如XLand环境（[来源](https://deep mind.com/research/publications/2021/open-ended-learning-leads-to-generally-capable-agent s))和
行星规模气候预测（[来源](https://www.nature.com/articles/s41586-021-03854-z))由DeepMind开发。为了提高大规模市场的性能，我们将采用基于GPU的大规模并行模拟，
就像Isaac Gym（[来源](https://arxiv.org/abs/2108.10470)). 
此外，探索深度进化RL框架将很有趣（[来源](https://doaj.org/article/4dd31838732842439cc1301e52613d1c))以模拟市场。
我们的最终目标是通过FinRL Meta深入了解复杂的市场现象，并为金融监管提供指导。

<div align="center">
<img align="center" src=figs/finrl_metaverse.png width="80%">
</div>


## Citing FinRL-Meta
```
@article{finrl_meta_2021,
    author = {Liu, Xiao-Yang and Rui, Jingyang and Gao, Jiechao and Yang, Liuqing and Yang, Hongyang and Wang, Zhaoran and Wang, Christina Dan and Guo Jian},
    title   = {{FinRL-Meta}: Data-Driven Deep ReinforcementLearning in Quantitative Finance},
    journal = {Data-Centric AI Workshop, NeurIPS},
    year    = {2021}
}

```

## Collaborators

<div align="center">
<img align="center" src=figs/Columbia_logo.jpg width="120"> &nbsp;&nbsp;
<img align="center" src=figs/IDEA_Logo.png width="200"> &nbsp;&nbsp;
<img align="center" src=figs/Northwestern_University.png width="120"> &nbsp;&nbsp;
<img align="center" src=figs/NYU_Shanghai_Logo.png width="200">	&nbsp;&nbsp;
</div>


**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
