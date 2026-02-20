"""
第五章多层网络模拟脚本。

使用方式：
  1. 直接运行：python multi_layer_case.py
  2. 在项目根目录下执行，确保能 import covasim、ContactNetwork、CrossNetwork 等

功能：
  - 创建四层区内网络（home/school/work/community），A/B 两区各自独立
  - 添加基于出行目的的跨区层（cross_work/cross_community/cross_home）
  - 候鸟跨境移动：出境时原属地权重冻结，跨区层按 purpose 激活

验证：
  运行结束后自动输出各层边数统计，用于确认：
  - 区内层：仅含 A 区内边与 B 区内边，跨区=0
  - 跨区层：仅含 A-B 跨区边，跨区>0
"""
import sys
import os

# 将项目根目录加入 path，使 import covasim 使用本地的 E:\my_paper\covasim\covasim
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import covasim as cv
import covasim.utils as cvu
import Enums
import sciris as sc
import os
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import ContactNetwork
import CrossNetwork
import MyPlot
from my_intervention import (
    ContactTracingAOnly,
    reduce_region_a_contacts,
    ScaleRegionBaseBetaByPhase,
    CrosserTravel,
    CrosserTravelMultilayer,
    WorkFromHomeA,
    CommunityRestrictA,
    MaskWearing,
    MaskWearingTwoPhase,
    MaskRelax,
    InjectUndocumentedInfectious,
)
from my_utils import (
    create_vaccination_schedule,
    sequence_random,
    sequence_crosser_first_then_random_a,
    is_position_a,
    is_position_b,
    make_subtarget_position,
    make_subtarget_crosser,
    make_subtarget_position_exclude_undocumented,
    make_subtarget_crosser_exclude_undocumented,
)

# 设置 matplotlib 显示中文（Windows 常用 SimHei / 微软雅黑）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框

# ================== 1. 网络层配置 ==================
custom_config={
    'community': {
        'network_type': Enums.NetWorkType.scale_free.name,
        'n_contacts': None,
        'm_connections': 3,  # 每个人加入网络时连接的边数 (决定了网络的平均密度)
        'age_range': None,
        'cluster_size': None
    },
    'work': {
        'network_type': Enums.NetWorkType.random.name,
        'n_contacts': 10,
        'age_range': (22, 65),
    },
    'school': {
        'network_type': Enums.NetWorkType.random.name,
        'n_contacts': 10,
        'age_range': (6, 22),
    },
    'home': {
        'network_type': Enums.NetWorkType.microstructured.name,
        'cluster_size': 3.0,
    }
}

# ================== 2. 人口与跨区层创建 ==================
# 国家配置：比例式 2:1 表示 A 占 2/3、B 占 1/3（也可写概率式如 {'A': 0.5, 'B': 0.5}）
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
# 添加多层跨境层（基于出行目的的精准嵌入）
popdict_base = CrossNetwork.add_cross_layer_multilayer(
    popdict_base,
    frac_travelers=0.01,
    n_cross_per_person=10,
    cross_beta=0.6,
    frac_work=0.7,
    frac_visit=0.25,
    frac_undocumented=0.05,
    cross_layer_seed=seed_cross_layer,
    region_a='A',
    region_b='B',
)

# ================== 3. 仿真参数与干预 ==================
custom_pars = {
    # Population parameters
    'pop_size': pop_size,
    'pop_infected': {'A':20, 'B': 20},
    'pop_infected_region_key': 'country',

    # Simulation parameters
    'start_day': '2021-07-04',
    'n_days': 180,

    # Rescaling parameters
    'rescale': False,        # 禁用动态调整人口大小

    # Network parameters（区内层 + 跨区层权重）
    'beta_layer': {
        'home': 3, 'school': 0.6, 'work': 0.6, 'community': 0.3,
        'cross_work': 0.6, 'cross_community': 0.6, 'cross_home': 0.6,
    },
    'beta': 0.036,
}

# 多层跨境移动（每日 10% 候鸟出境，停留 1~7 天）
crosser_travel_ml = CrosserTravelMultilayer(
    frac_cross_per_day=0.1,
    duration_min=1,
    duration_max=7,
    start_day=0,
)
# A 区居家办公（工作层移除 70% 边）+ 社区限制（社区层移除 50% 边），须在 CrosserTravelMultilayer 之后
interventions = [
    crosser_travel_ml,
    WorkFromHomeA(start_day=0, fraction=0.3),
    # CommunityRestrictA(start_day=0, fraction=0.5),
]

# ================== 4. 运行模拟 ==================
sim = cv.Sim(
    pars=custom_pars,
    label='多层网络模拟',
    interventions=interventions,
    analyzers=[MyPlot.CountryRegionAnalyzer(country_key='country', regions=('A', 'B'))],
)
sim.popdict = popdict_base
sim.reset_layer_pars(force=True)
sim.initialize()
sim.run()

results_dir = r'myproject\results\多层耦合网络图片\居家办公干预'
os.makedirs(results_dir, exist_ok=True)
sim_basename = 'case01'
sim_path = os.path.join(results_dir, sim_basename + '.sim')
sim.save(filename=sim_path, keep_people=True)  # 保留 people 与 infection_log，供 plot_case 等后续绘图使用

# 各层每日新感染人数（按区域、按传播层）
MyPlot.plot_layer_region_infections(
    sim,
    country_key='country',
    regions=('A', 'B'),
    layers=['home', 'school', 'work', 'community'],
    show_regions=('A'),
    save_path=os.path.join(results_dir, sim_basename + '.png'),
)
