import numpy as np
import covasim as cv
import Enums
import sciris as sc
import os
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import ContactNetwork
import CrossNetwork
import MyPlot

# 设置 matplotlib 显示中文（Windows 常用 SimHei / 微软雅黑）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框

# 定义层级配置
custom_config={
    'base': {
        'network_type': Enums.NetWorkType.scale_free.name,
        'n_contacts': None,
        'm_connections': 3,  # 每个人加入网络时连接的边数 (决定了网络的平均密度)
        'age_range': None,
        'cluster_size': None
    }
}

# 定义国家配置：比例式 2:1 表示 A 占 2/3、B 占 1/3（也可写概率式如 {'A': 0.5, 'B': 0.5}）
countries_config = {
    'A': 2,
    'B': 1
}

# 随机种子：可分别指定，便于复现或做敏感性分析
seed_population = 0      # 人口与区内接触网（create_custom_population）
seed_cross_layer = 42    # 跨区层（流动者选取与跨区边）

# 创建自定义人口（pop_size 须与下方 custom_pars['pop_size'] 一致）
pop_size = 30000
popdict_base, custom_keys = ContactNetwork.create_custom_population(
    pop_size, custom_config, countries_config, seed=seed_population
)

# 创建自定义参数（pop_size 须与上面 create_custom_population 的 pop_size 一致）
custom_pars = {
    # Population parameters
    'pop_size': pop_size,
    'pop_infected': {'A':0, 'B': 20},
    'pop_infected_region_key': 'country',

    # Simulation parameters
    'start_day': '2021-07-04',
    'n_days': 60,

    # Rescaling parameters
    'rescale': False,        # 禁用动态调整人口大小

    # Network parameters（base 与 cross 层权重；cross 可设低一些表示跨区接触强度）
    'beta_layer': {'base': 1.0, 'cross': 0.6},
    # Basic disease transmission parameters
    'beta': 0.036,
}

# 无跨区层时 A/B 两区不接触；加上跨区层后可观察跨境传播
# frac_travelers 为每区流动人口比例 (0~1)，0.01 表示每区 1% 为流动者，总跨区人数约等于总人口的 1%
popdict = CrossNetwork.add_cross_layer(
    popdict_base, frac_travelers=0.01, n_cross_per_person=10, cross_beta=0.6, cross_layer_seed=seed_cross_layer
)

sim = cv.Sim(
    pars=custom_pars,
    label='无干预',
    analyzers=[MyPlot.CountryRegionAnalyzer(country_key='country', regions=('A', 'B'))],
)
sim.popdict = popdict
sim.reset_layer_pars(force=True)
sim.initialize()
sim.run()

# 保存模拟结果与图片到指定目录
results_dir = r'E:\大论文相关\covasim\myproject\results\双耦合网络图片\无干预模拟'
os.makedirs(results_dir, exist_ok=True)
sim.save(os.path.join(results_dir, 'cross_1%.sim'))

# 按 A/B 两区域分别绘制：左上/右上为 A 区 SEIR+病程，左下/右下为 B 区，并保存图片
MyPlot.plot_two_country_epidemic_curves(
    sim, country_key='country', regions=('A', 'B'),
    save_path=os.path.join(results_dir, '两区域疫情曲线_1%.png'),
)

