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
        'age_range': None,
        'cluster_size': None
    }

}

# 定义国家配置 不配置默认为一个国家
countries_config = {
    'A': 0.5,
    'B': 0.5
}

# 跨区层参数（方式二：仅流动人口有跨区边）
frac_travelers = 0.08       # 每区流动人口比例
n_cross_per_person = 2      # 每个流动者在本层的跨区边数
cross_beta = 0.6            # 跨区边的传播权重（相对区内可略低）
cross_layer_seed = 42       # 随机种子，便于复现

# 创建自定义人口（pop_size 须与下方 custom_pars['pop_size'] 一致）
pop_size = 10000
custom_popdict, custom_keys = ContactNetwork.create_custom_population(pop_size, custom_config, countries_config)

# 添加跨区接触层（仅当存在 A、B 两区时）
countries = custom_popdict['country']
unique_countries = np.unique(countries)
if len(unique_countries) >= 2:
    rng = np.random.RandomState(cross_layer_seed)
    inds_A = np.where(countries == 'A')[0]
    inds_B = np.where(countries == 'B')[0]
    if len(inds_A) > 0 and len(inds_B) > 0:
        n_travelers_A = max(1, int(frac_travelers * len(inds_A)))
        n_travelers_B = max(1, int(frac_travelers * len(inds_B)))
        travelers_A = rng.choice(inds_A, size=n_travelers_A, replace=False)
        travelers_B = rng.choice(inds_B, size=n_travelers_B, replace=False)

        p1_cross_list = []
        p2_cross_list = []
        for a in travelers_A:
            partners = rng.choice(inds_B, size=n_cross_per_person, replace=True)
            for b in partners:
                p1_cross_list.append(a)
                p2_cross_list.append(b)
        for b in travelers_B:
            partners = rng.choice(inds_A, size=n_cross_per_person, replace=True)
            for a in partners:
                p1_cross_list.append(b)
                p2_cross_list.append(a)

        p1_cross = np.array(p1_cross_list, dtype=cv.default_int)
        p2_cross = np.array(p2_cross_list, dtype=cv.default_int)
        n_cross = len(p1_cross)
        beta_cross = np.full(n_cross, cross_beta, dtype=cv.default_float)

        cross_layer = cv.Layer(p1=p1_cross, p2=p2_cross, beta=beta_cross, label='cross')
        custom_popdict['contacts'].add_layer(cross=cross_layer)
        custom_popdict['layer_keys'].append('cross')
        custom_keys = list(custom_popdict['layer_keys'])

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

# 创建模拟
sim_base = cv.Sim(pars=custom_pars)

sim_base.popdict = custom_popdict
sim_base.reset_layer_pars(force=True) 
sim_base.initialize()

sim_base.run()
sim_base.people.to_graph()
print(sim_base.layer_keys())
# sim_base.plot()