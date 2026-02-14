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
    'pop_infected': 20,

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

# 三个 sim：仅 frac_travelers 不同（0.01, 0.03, 0.06），共用一个基础人口再分别加跨区层
frac_travelers_list = [0.01, 0.03, 0.06]
sims = []
for ft in frac_travelers_list:
    popdict_copy = sc.dcp(popdict_base)
    popdict_copy = CrossNetwork.add_cross_layer(
        popdict_copy,
        frac_travelers=ft,
        n_cross_per_person=2,
        cross_beta=0.6,
        cross_layer_seed=seed_cross_layer,
    )
    sim = cv.Sim(pars=custom_pars, label=f'流动比例 {ft*100:.0f}%')
    sim.popdict = popdict_copy
    sim.reset_layer_pars(force=True)
    sim.initialize()
    sim.run()
    sims.append(sim)

# 保存完整模拟结果到「跨境传播敏感性」目录，下次可用 cv.MultiSim.load(...) 加载后直接画图
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', '双耦合网络图片', '跨境传播敏感性')
os.makedirs(results_dir, exist_ok=True)
msim = cv.MultiSim(sims)
msim.save(os.path.join(results_dir, '跨境传播敏感性.msim'))

# 绘制三个 sim 的累计感染人数
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for sim in msim.sims:
    ax.plot(sim.results['t'], sim.results['cum_infections'].values, label=sim.label)
ax.set_xlabel('天数')
ax.set_ylabel('累计感染人数')
ax.set_title('不同流动人口比例下累计感染人数对比')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
