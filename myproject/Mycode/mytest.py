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
    'A': 1.0,  # 60%
}

# 创建自定义人口
custom_popdict, custom_keys = ContactNetwork.create_custom_population(1000, custom_config_test, countries_config)

# 定义干预措施
# 1. 测试干预：用于产生诊断（diagnoses）
#    - symp_prob: 有症状者被测试的概率
#    - asymp_prob: 无症状者被测试的概率
#    - test_delay: 测试结果延迟天数
tp = cv.test_prob(symp_prob=0.2, asymp_prob=0.01, start_day='2022-02-14', test_delay=2)

# 2. 接触者追踪干预：用于产生隔离（quarantined）
#    - trace_probs: 各层的追踪概率（需要根据你的层键调整）
#    注意：如果你的层键是 'country'，需要相应调整
ct = cv.contact_tracing(trace_probs=0.3, start_day='2022-02-14')

# 3. 疫苗干预（可选）：如果需要疫苗数据，取消下面的注释
# vx = cv.vaccinate_prob('pfizer', days=5, prob=0.1, start_day='2022-02-20')

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
    'beta_dist': {'dist': 'uniform', 'par1': 1.0, 'par2': 1.0},
    'beta': 0.036,

    # Parameters that control settings and defaults for multi-variant runs 无变化

    # Parameters used to calculate immunity
    
    # 添加干预措施
    'interventions': [tp, ct],  # 包含测试和接触者追踪
    # 'interventions': [tp, ct, vx],  # 如果启用疫苗，使用这行替换上面那行

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
#转换为包含所有层的图
G = sim.people.contacts.to_graph

#绘制
# nx.draw(G, with_labels=False, node_size=30, alpha=0.6)
# plt.title('All Contact Layers')
# plt.show()
