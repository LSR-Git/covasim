import numpy as np
import covasim as cv
import Enums
import sciris as sc
import os
import matplotlib.pyplot as plt
import networkx as nx
import ContactNetwork

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
custom_pars = {
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
    'beta': 0.036,
}



# 创建模拟（只初始化一次，作为后续复制的"干净"模板）
base_sim = cv.Sim(pars=custom_pars)
base_sim.popdict = custom_popdict
base_sim.reset_layer_pars(force=True)  # 自定义网络需要 force=True
base_sim.initialize()


n_sims = 3
betas = [0.016, 0.026, 0.036]
# 用 MultiSim 同时跑三个模拟
# 注意：Windows 下多进程必须在 if __name__ == '__main__' 内执行
if __name__ == '__main__':
    msims = []
    for beta in betas:
        sims = []
        for s in range(n_sims):
            # 必须从 base_sim 复制，不能从上次循环的 sim 复制（否则会复制已运行完的人群状态）
            sim = base_sim.copy()
            sim.pars['beta'] = beta
            sim.label = f'Beta = {beta}'
            sim.pars['rand_seed'] = s
            sims.append(sim)
        msim = cv.MultiSim(sims)
        msim.run()
        msim.mean()
        msims.append(msim)

    merged = cv.MultiSim.merge(msims, base=True)
    merged.plot(color_by_sim=True)
