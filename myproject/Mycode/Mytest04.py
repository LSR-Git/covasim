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

# 1. 检测隔离：检测阳性者被隔离
test_isolate = cv.test_prob(
    symp_prob=0.5,           # 有症状者检测概率
    asymp_prob=0.02,         # 无症状者检测概率
    start_day=intervention_start,
    test_delay=1,
)

# 2. 接触者追踪：检测 + 追踪密切接触者并隔离（需与检测配合）
test_for_ct = cv.test_prob(
    symp_prob=0.4, asymp_prob=0.01,
    start_day=intervention_start, test_delay=1,
)
contact_tracing = cv.contact_tracing(
    trace_probs=0.5,         # 追踪到的接触者比例
    trace_time=2,            # 追踪延迟天数
    start_day=intervention_start,
)

# 3. 疫苗接种（days 为开始接种的日期或日期列表，从第 intervention_start 天起每天可接种）
vaccination = cv.vaccinate_prob(
    vaccine='pfizer',
    days=list(range(intervention_start, basepars['n_days'])),  # 从第10天到模拟结束每天可接种
    prob=0.08,               # 每日接种概率
)

# 4. 口罩佩戴：降低传播率
mask_wearing = cv.change_beta(
    days=intervention_start,
    changes=0.7,             # 传播率降为原来的70%（即降低30%）
    layers=None,
)

# 创建四个情景的 Sim（共用同一套 basepars）
sim_test_isolate = cv.Sim(
    pars=basepars,
    interventions=test_isolate,
    label='检测隔离',
)
sim_contact_trace = cv.Sim(
    pars=basepars,
    interventions=[test_for_ct, contact_tracing],
    label='接触者追踪',
)
sim_vaccination = cv.Sim(
    pars=basepars,
    interventions=vaccination,
    label='疫苗接种',
)
sim_mask = cv.Sim(
    pars=basepars,
    interventions=mask_wearing,
    label='口罩佩戴',
)
sim_base = cv.Sim(
    pars=basepars,
    label='基线情景',
)

# 并行运行四个模拟
if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    msim = cv.parallel([sim_base, sim_test_isolate, sim_contact_trace, sim_vaccination, sim_mask])
    msim.save(os.path.join(results_dir, 'four_interventions_results.msim'))

    # 绘制累计感染曲线对比
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for sim in msim.sims:
        ax.plot(
            sim.results['t'],  # 天数 0, 1, 2, ...
            sim.results['cum_infections'].values,
            label=sim.label,
        )
    ax.set_xlabel('天数')
    ax.set_ylabel('累计感染人数')
    ax.set_title('四种干预措施下累计感染人数对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'four_interventions_cum_infections.png'), dpi=150)
    plt.show()

