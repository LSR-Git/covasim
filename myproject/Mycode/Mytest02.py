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
countries_config = {}

# 创建自定义人口
custom_popdict, custom_keys = ContactNetwork.create_custom_population(1000, custom_config, countries_config)

# 创建自定义参数
custom_pars = {
    # Population parameters
    'pop_size': 1000,
    'pop_infected': 10,

    # Simulation parameters
    'start_day': '2021-07-04',
    'n_days': 60,

    # Rescaling parameters
    'rescale': False,        # 禁用动态调整人口大小

    # Network parameters  无变化

    # Basic disease transmission parameters
    'beta_dist': {'dist': 'uniform', 'par1': 1.0, 'par2': 1.0},
    'beta': 0.036,

    # Parameters that control settings and defaults for multi-variant runs 无变化

    # Parameters used to calculate immunity
    
    # 添加干预措施
    # 'interventions': [tp, ct],  # 包含测试和接触者追踪
    # 'interventions': [tp, ct, vx],  # 如果启用疫苗，使用这行替换上面那行
}

# 创建模拟
sim = cv.Sim(pars=custom_pars)
sim.popdict = custom_popdict
sim.reset_layer_pars(force=True) 
sim.initialize()

sim.run()
sim.save(f'E:/大论文相关/covasim/myproject/results/test.sim')
sim.plot(to_plot=['cum_infections', 'new_infections', 'cum_deaths', 'n_infectious'])