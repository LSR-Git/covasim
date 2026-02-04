import numpy as np
import covasim as cv
import Enums
import sciris as sc
import os
import matplotlib.pyplot as plt
import networkx as nx
import ContactNetwork

# 定义层级配置
custom_config_test={
    'country': {
        'network_type': Enums.NetWorkType.scale_free.name,
        'n_contacts': None,
        'm_connections': 3,  # 每个人加入网络时连接的边数 (决定了网络的平均密度)
        'beta': 0.3,
        'age_range': None,
        'cluster_size': None
    }   
}

# 定义国家配置（比例总和必须等于1.0）
countries_config = {
    'A': 0.6,  # 60%
    'B': 0.4   # 40%
}

# 创建自定义人口
custom_popdict, custom_keys = ContactNetwork.create_custom_population(1000, custom_config_test, countries_config)

# 创建自定义参数
custom_pars = {
    # Population parameters
    'pop_size': 1000,
    'pop_infected': 10,

    # Simulation parameters
    'start_day': '2022-02-14',
    'end_day': '2022-03-29',

    # Rescaling parameters
    'rescale': False,        # 禁用动态调整人口大小

    # Network parameters  无变化

    # Basic disease transmission parameters
    'beta': 0.036,


}

# 创建模拟
sim = cv.Sim(pars=custom_pars)
sim.popdict = custom_popdict
sim.reset_layer_pars() 
sim.initialize()

sim.run()
print(f"pop_type: {sim['pop_type']}")
print(f"层键: {sim.layer_keys()}")
print(f"contacts: {sim['contacts']}")
print(f"beta_layer: {sim['beta_layer']}")
print(f"dynam_layer: {sim['dynam_layer']}")
sim.plot()
# 转换为包含所有层的图
# G = sim.people.contacts.to_graph()

# 绘制
# nx.draw(G, with_labels=False, node_size=30, alpha=0.6)
# plt.title('All Contact Layers')
# plt.show()