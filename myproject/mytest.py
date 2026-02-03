import numpy as np
import covasim as cv
import enums
import sciris as sc
import os
import matplotlib.pyplot as plt
import networkx as nx

def create_custom_population(pop_size, layer_config):
    '''
    创建完全自定义的人口
    
    Args:
        pop_size: 人口大小
        layer_config: 层配置字典，格式为：
            {
                'layer_name': {
                    'n_contacts': 平均接触数,
                    'beta': 传播率,
                    'age_range': (min_age, max_age) 或 None 表示所有年龄,
                    'cluster_size': 如果是聚类结构，指定聚类大小；否则为 None
                }
            }
    '''
    # 创建基本属性
    uids = np.arange(pop_size, dtype=cv.default_int)
    ages = np.random.uniform(18, 65, pop_size)
    sexes = np.random.binomial(1, 0.5, pop_size)
    countries = np.random.choice(['A','B'], pop_size)
    
    # 创建接触网络
    contacts = cv.Contacts()
    layer_keys = []
    
    for layer_name, config in layer_config.items():
        layer_keys.append(layer_name)
        
        # 根据年龄范围筛选人员
        if config.get('age_range') is not None:
            min_age, max_age = config['age_range']
            age_mask = (ages >= min_age) & (ages < max_age)
            indices = np.where(age_mask)[0]
        else:
            indices = None
        
        # 创建接触
        if config.get('network_type') == enums.NetWorkType.scale_free.name:
            # 使用无标度网络
            # 获取参数 m (每个新节点的连接数)
            m = config.get('m_connections', 2) 
            
            if indices is not None:
                # 针对特定人群建立无标度网络 (传入 indices 进行映射)
                layer_contacts = cv.make_scale_free_contacts(len(indices), m_connections=m, mapping=indices)
            else:
                # 全员无标度
                layer_contacts = cv.make_scale_free_contacts(pop_size, m_connections=m)          
        elif config.get('network_type') == enums.NetWorkType.microstructured.name:
            # 使用聚类结构
            if indices is not None:
                # 需要先创建完整网络，然后筛选
                temp_contacts = cv.make_microstructured_contacts(len(indices), cluster_size=config['cluster_size'])
                # 映射回原始索引
                temp_contacts['p1'] = indices[temp_contacts['p1']]
                temp_contacts['p2'] = indices[temp_contacts['p2']]
                layer_contacts = temp_contacts
            else:
                layer_contacts = cv.make_microstructured_contacts(pop_size, cluster_size=config['cluster_size'])
        elif config.get('network_type') == enums.NetWorkType.random.name:
            # 使用随机接触
            n_contacts = config.get('n_contacts', 10)
            if indices is not None:
                layer_contacts = cv.make_random_contacts(len(indices), n=n_contacts, mapping=indices)
            else:
                layer_contacts = cv.make_random_contacts(pop_size, n=n_contacts)
        
        # 创建层
        layer = cv.Layer(**layer_contacts, label=layer_name)
        contacts.add_layer(**{layer_name: layer})
    
    # 创建人口字典
    popdict = {
        'uid': uids,
        'age': ages,
        'sex': sexes,
        'contacts': contacts,
        'layer_keys': layer_keys
    }
    
    return popdict, layer_keys

custom_config_test={
    'country': {
        'network_type': enums.NetWorkType.scale_free.name,
        'n_contacts': None,
        'm_connections': 3,  # 每个人加入网络时连接的边数 (决定了网络的平均密度)
        'beta': 0.3,
        'age_range': None,
        'cluster_size': None
    }   
}

# 创建自定义人口
custom_popdict, custom_keys = create_custom_population(100, custom_config_test)

# 创建模拟
sim = cv.Sim(pop_size=100, n_days=90)
sim.popdict = custom_popdict

# 设置网络参数
sim['contacts'] = {name: config.get('n_contacts', config.get('cluster_size', 10)) 
                     for name, config in custom_config_test.items()}
sim['beta_layer'] = {name: config['beta'] for name, config in custom_config_test.items()}
sim['dynam_layer'] = {name: 0 for name in custom_keys}  # 默认都不是动态的
sim['iso_factor'] = {name: 0.1 for name in custom_keys}
sim['quar_factor'] = {name: 0.2 for name in custom_keys}

sim.reset_layer_pars() 

sim.initialize()
sim.run()
# 转换为包含所有层的图
G = sim.people.contacts.to_graph()

# 绘制
nx.draw(G, with_labels=False, node_size=30, alpha=0.6)
plt.title('All Contact Layers')
plt.show()