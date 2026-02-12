import numpy as np
import covasim as cv
import Enums
import sciris as sc
import os
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import ContactNetwork

# 设置 matplotlib 显示中文（Windows 常用微软雅黑/黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框



# 定义层级配置
custom_config={
    'base': {
        'network_type': Enums.NetWorkType.scale_free.name,
        'n_contacts': None,
        'm_connections': 3,  # 每个人加入网络时连接的边数 (决定了网络的平均密度)
        'beta': 0.3,
        'age_range': None,
        'cluster_size': None
    }   
}

# 定义国家配置 不配置默认为一个国家
countries_config = {
    'A': 1.0
}

# 创建自定义人口
custom_popdict, custom_keys = ContactNetwork.create_custom_population(1000, custom_config, countries_config)
# 创建自定义参数
basepars = {
    # Population parameters
    'pop_size': 1000,
    'pop_infected': 20,

    # Simulation parameters
    'start_day': '2021-07-04',
    'n_days': 60,

    # Rescaling parameters
    'rescale': False,        # 禁用动态调整人口大小

    # Network parameters  无变化
    # Basic disease transmission parameters
    'beta': 0.016,

}
# 干预开始日（统一从第10天开始便于比较）
intervention_start = 10

# 1. 检测隔离：检测阳性者被隔离 20%
test_isolate_20 = cv.test_prob(
    symp_prob=0.2,           # 有症状者检测概率
    asymp_prob=0.02,         # 无症状者检测概率
    start_day=intervention_start,
    test_delay=6,
)

# 2. 检测隔离：检测阳性者被隔离 40%
test_isolate_40 = cv.test_prob(
    symp_prob=0.4,           # 有症状者检测概率
    asymp_prob=0.02,         # 无症状者检测概率
    start_day=intervention_start,
    test_delay=6,
)

# 2. 检测隔离：检测阳性者被隔离
test_isolate_60 = cv.test_prob(
    symp_prob=0.6,           # 有症状者检测概率
    asymp_prob=0.02,         # 无症状者检测概率
    start_day=intervention_start,
    test_delay=6,
)


# 创建四个情景的 Sim（共用同一套 basepars）
sim_test_isolate_20 = cv.Sim(
    pars=basepars,
    interventions=test_isolate_20,
    label='检测隔离 20%',
)
sim_test_isolate_40 = cv.Sim(
    pars=basepars,
    interventions=test_isolate_40,
    label='检测隔离 40%',
)
sim_test_isolate_60 = cv.Sim(
    pars=basepars,
    interventions=test_isolate_60,
    label='检测隔离 60%',
)
sim_base = cv.Sim(
    pars=basepars,
    label='基线情景',
)

# 并行运行四个模拟
if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results/单区域网络图片/检测隔离情况模拟/结果延迟6天')
    msim = cv.parallel([sim_base, sim_test_isolate_20, sim_test_isolate_40, sim_test_isolate_60])
    msim.save(os.path.join(results_dir, 'test_isolate_different_ratio_cum_infections.msim'))

    # 绘制累计感染曲线对比
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for sim in msim.sims:
        ax.plot(
            sim.results['t'],  # 天数 0, 1, 2, ...
            sim.results['cum_infections'].values,
            label=sim.label,
        )
        sim.to_excel(os.path.join(results_dir, f'{sim.label}.xlsx'))
    ax.set_xlabel('天数')
    ax.set_ylabel('累计感染人数')
    ax.set_title('检测隔离不同比例下累计感染人数对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'test_isolate_different_ratio_cum_infections.png'), dpi=150)
    plt.show()

