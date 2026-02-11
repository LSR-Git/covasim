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
custom_popdict, custom_keys = ContactNetwork.create_custom_population(int(1000), custom_config, countries_config)

# 创建自定义参数
custom_pars = {
    # Population parameters
    'pop_size': int(1000),  # 必须为整数，10e3 在 Python 里是 10000.0（float）
    'pop_infected': 20,

    # Simulation parameters
    'start_day': '2021-07-04',
    'n_days': 60,

    # Rescaling parameters
    'rescale': False,        # 禁用动态调整人口大小

    # Network parameters  无变化

    # Basic disease transmission parameters
    'beta': 0.026,
}



# 创建模拟
sim_base = cv.Sim(pars=custom_pars)

sim_base.popdict = custom_popdict
sim_base.reset_layer_pars(force=True) 
sim_base.initialize()

sim_base.run()

# sim_base.plot()
G = sim_base.people.to_graph()
print(G)

# 画网络：感染者红色，其余白色（节点属性来自 people，如 naive=True 表示从未感染）
node_colors = [
    'red' if not G.nodes[n].get('naive', True) else 'white'
    for n in G.nodes()
]
plt.figure(figsize=(10, 8))
nx.draw(
    G,
    node_color=node_colors,
    node_size=20,
    edge_color='gray',
    alpha=0.6,
    with_labels=False,
    pos=nx.spring_layout(G, seed=42, k=0.5),  # 布局可改为 nx.kamada_kawai_layout(G) 等
)
plt.title('Contact network: red=infected, white=naive')
plt.axis('off')
plt.tight_layout()
plt.show()
