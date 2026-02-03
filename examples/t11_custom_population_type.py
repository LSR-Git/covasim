'''
演示如何创建完全自定义的人口类型和层结构
不使用默认的 h, s, w, c 层，而是创建自己的层键
'''

import numpy as np
import covasim as cv
import sciris as sc
import os

# 检查导入的 covasim 版本（可选，用于调试）
# 取消下面的注释来查看导入的是本地还是 pip 安装的版本
# print("="*50)
# print("Covasim 导入信息:")
# print(f"模块路径: {os.path.dirname(cv.__file__)}")
# print(f"版本: {cv.__version__}")
# if 'site-packages' in os.path.dirname(cv.__file__):
#     print("来源: pip 安装的版本")
# else:
#     print("来源: 本地开发版本")
# print("="*50)

# 方法1：创建完全自定义的人口（适合简单场景，代码直观）
# 直接编写代码创建每一层，适合层数少、结构简单的情况
# 定义自定义的层键，例如：'home'（家庭）, 'office'（办公室）, 'restaurant'（餐厅）, 'gym'（健身房）
custom_layer_keys = ['home', 'office', 'restaurant', 'gym']

# 创建人口大小
pop_size = 10000

# 创建基本人口属性
uids = np.arange(pop_size, dtype=cv.default_int)
ages = np.random.uniform(18, 65, pop_size)
sexes = np.random.binomial(1, 0.5, pop_size)

# 创建自定义的接触网络
contacts = cv.Contacts()

# 1. 家庭层（home）- 使用微结构化接触（聚类）
home_contacts = cv.make_microstructured_contacts(pop_size, cluster_size=3.0)
contacts.add_layer(home=cv.Layer(**home_contacts, label='home'))

# 2. 办公室层（office）- 随机接触，但只针对工作年龄的人
work_ages = (ages >= 22) & (ages < 65)
work_indices = np.where(work_ages)[0]
office_contacts = cv.make_random_contacts(len(work_indices), n=15, mapping=work_indices)
contacts.add_layer(office=cv.Layer(**office_contacts, label='office'))

# 3. 餐厅层（restaurant）- 随机接触
restaurant_contacts = cv.make_random_contacts(pop_size, n=5)
contacts.add_layer(restaurant=cv.Layer(**restaurant_contacts, label='restaurant'))

# 4. 健身房层（gym）- 随机接触，但只针对特定年龄
gym_ages = (ages >= 20) & (ages < 50)
gym_indices = np.where(gym_ages)[0]
gym_contacts = cv.make_random_contacts(len(gym_indices), n=8, mapping=gym_indices)
contacts.add_layer(gym=cv.Layer(**gym_contacts, label='gym'))

# 创建人口字典
popdict = {
    'uid': uids,
    'age': ages,
    'sex': sexes,
    'contacts': contacts,
    'layer_keys': custom_layer_keys
}

# 创建模拟，使用自定义人口
sim = cv.Sim(
    pop_size=pop_size,
    n_days=90,
    pop_type='random'  # 使用 random 作为基础，但会被自定义人口覆盖
)

# 设置自定义人口
sim.popdict = popdict

# 手动设置网络参数（因为不是标准的 h, s, w, c）
# 必须为每个自定义层设置参数
sim['contacts'] = {
    'home': 2.0,        # 家庭平均接触数
    'office': 15,       # 办公室平均接触数
    'restaurant': 5,    # 餐厅平均接触数
    'gym': 8           # 健身房平均接触数
}

sim['beta_layer'] = {
    'home': 4.0,        # 家庭传播率最高
    'office': 0.8,     # 办公室传播率中等
    'restaurant': 0.5,  # 餐厅传播率较低
    'gym': 0.6         # 健身房传播率中等
}

sim['dynam_layer'] = {
    'home': 0,         # 家庭层不是动态的
    'office': 0,       # 办公室层不是动态的
    'restaurant': 1,   # 餐厅层是动态的（可能因为营业时间变化）
    'gym': 1          # 健身房层是动态的
}

sim['iso_factor'] = {
    'home': 0.3,
    'office': 0.05,
    'restaurant': 0.05,
    'gym': 0.05
}

sim['quar_factor'] = {
    'home': 0.6,
    'office': 0.1,
    'restaurant': 0.1,
    'gym': 0.1
}

# 初始化并运行
sim.initialize()
sim.run()
sim.plot()

print(f"自定义层键: {sim.layer_keys()}")
print(f"各层接触数: {sim['contacts']}")
print(f"各层传播率: {sim['beta_layer']}")


# 方法2：使用 reset_layer_pars 自动设置（更简单）
# 如果你已经创建了自定义人口，可以让系统自动检测层键并设置默认参数

print("\n" + "="*50)
print("方法2：使用 reset_layer_pars 自动设置")
print("="*50)

# 创建另一个自定义人口
sim2 = cv.Sim(pop_size=5000, n_days=60, pop_type='random')
sim2.popdict = popdict  # 使用相同的自定义人口

# 让系统自动检测层键并设置默认参数
sim2.reset_layer_pars()  # 这会自动为所有层设置默认参数

# 然后可以手动调整特定层的参数
sim2['beta_layer']['home'] = 5.0  # 只修改家庭传播率
sim2['contacts']['gym'] = 10     # 只修改健身房接触数

sim2.initialize()
sim2.run()

print(f"自动检测的层键: {sim2.layer_keys()}")
print(f"自动设置的参数: {sim2['beta_layer']}")


# 方法3：使用配置字典+辅助函数（适合复杂场景，可复用）
# 通过配置字典定义层结构，函数封装了创建逻辑，代码更简洁、可复用
print("\n" + "="*50)
print("方法3：使用配置字典+辅助函数（适合复杂场景）")
print("="*50)

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
        if config.get('cluster_size') is not None:
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
        else:
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


# 定义自定义层配置
custom_config = {
    'family': {
        'n_contacts': None,  # 使用聚类，不需要指定接触数
        'beta': 4.0,
        'age_range': None,  # 所有年龄
        'cluster_size': 3.5  # 平均每个家庭3.5人
    },
    'workplace': {
        'n_contacts': 20,
        'beta': 0.8,
        'age_range': (22, 65),  # 工作年龄
        'cluster_size': None  # 随机接触
    },
    'shopping': {
        'n_contacts': 10,
        'beta': 0.4,
        'age_range': None,  # 所有年龄
        'cluster_size': None
    },
    'transport': {
        'n_contacts': 15,
        'beta': 0.3,
        'age_range': None,
        'cluster_size': None
    }
}

# 创建自定义人口
custom_popdict, custom_keys = create_custom_population(8000, custom_config)

# 创建模拟
sim3 = cv.Sim(pop_size=8000, n_days=90, pop_type='random')
sim3.popdict = custom_popdict

# 设置网络参数
sim3['contacts'] = {name: config.get('n_contacts', config.get('cluster_size', 10)) 
                     for name, config in custom_config.items()}
sim3['beta_layer'] = {name: config['beta'] for name, config in custom_config.items()}
sim3['dynam_layer'] = {name: 0 for name in custom_keys}  # 默认都不是动态的
sim3['iso_factor'] = {name: 0.1 for name in custom_keys}
sim3['quar_factor'] = {name: 0.2 for name in custom_keys}

sim3.initialize()
sim3.run()

print(f"自定义层键: {sim3.layer_keys()}")
print(f"各层配置:")
for layer_name in custom_keys:
    print(f"  {layer_name}:")
    print(f"    接触数: {sim3['contacts'][layer_name]}")
    print(f"    传播率: {sim3['beta_layer'][layer_name]}")
