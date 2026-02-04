'''
演示如何自定义人员的传播参数

在Covasim中，有两个关键的个人传播参数：
1. rel_sus (相对易感性) - 控制个人被感染的概率
   - 值越大，越容易被感染
   - 默认值基于年龄，通常在0.5-2.0之间
   - 设置为0表示完全免疫，设置为1表示正常易感性

2. rel_trans (相对传播性) - 控制个人感染他人的概率
   - 值越大，传播能力越强
   - 默认值基于年龄和病毒载量分布
'''

import numpy as np
import covasim as cv
import ContactNetwork
import Enums
import matplotlib.pyplot as plt

# ============================================================================
# 方法1：在初始化后、运行前设置传播参数（推荐）
# ============================================================================

print("="*60)
print("方法1：在初始化后设置传播参数")
print("="*60)

# 创建自定义人口
custom_config = {
    'country': {
        'network_type': Enums.NetWorkType.scale_free.name,
        'n_contacts': None,
        'm_connections': 3,
        'beta': 0.3,
        'age_range': None,
        'cluster_size': None
    }   
}

pop_size = 100

# 定义国家配置（比例总和必须等于1.0）
countries_config = {
    'A': 0.6,  # 60%
    'B': 0.4   # 40%
}

custom_popdict, custom_keys = ContactNetwork.create_custom_population(pop_size, custom_config, countries_config)

# 创建模拟
sim = cv.Sim(pop_size=pop_size, n_days=90)
sim.popdict = custom_popdict
sim.reset_layer_pars()

# 初始化（这一步会创建 people 对象并设置默认的传播参数）
sim.initialize()

# 现在可以自定义传播参数了
# 方法1.1：根据年龄设置
young = sim.people.age < 30
middle = (sim.people.age >= 30) & (sim.people.age < 50)
old = sim.people.age >= 50

# 年轻人：易感性高，传播性强
sim.people.rel_sus[young] = 1.5
sim.people.rel_trans[young] = 1.2

# 中年人：正常
sim.people.rel_sus[middle] = 1.0
sim.people.rel_trans[middle] = 1.0

# 老年人：易感性高，但传播性低
sim.people.rel_sus[old] = 1.8
sim.people.rel_trans[old] = 0.8

print(f"年轻人数量: {young.sum()}, 易感性: {sim.people.rel_sus[young].mean():.2f}, 传播性: {sim.people.rel_trans[young].mean():.2f}")
print(f"中年人数量: {middle.sum()}, 易感性: {sim.people.rel_sus[middle].mean():.2f}, 传播性: {sim.people.rel_trans[middle].mean():.2f}")
print(f"老年人数量: {old.sum()}, 易感性: {sim.people.rel_sus[old].mean():.2f}, 传播性: {sim.people.rel_trans[old].mean():.2f}")

# 方法1.2：根据自定义属性设置（例如：国家）
# 假设我们在创建人口时添加了 'country' 属性
# 注意：需要在创建 popdict 时添加这个属性
if 'country' in custom_popdict:
    country_A = custom_popdict['country'] == 'A'
    country_B = custom_popdict['country'] == 'B'
    
    # 国家A：易感性高
    sim.people.rel_sus[country_A] = 1.3
    # 国家B：易感性低
    sim.people.rel_sus[country_B] = 0.7

# 方法1.3：随机设置（例如：模拟基因差异）
np.random.seed(42)
# 易感性：0.5-1.5之间随机
sim.people.rel_sus = np.random.uniform(0.5, 1.5, pop_size)
# 传播性：0.8-1.2之间随机
sim.people.rel_trans = np.random.uniform(0.8, 1.2, pop_size)

print(f"\n随机设置后的统计:")
print(f"易感性 - 均值: {sim.people.rel_sus.mean():.2f}, 范围: [{sim.people.rel_sus.min():.2f}, {sim.people.rel_sus.max():.2f}]")
print(f"传播性 - 均值: {sim.people.rel_trans.mean():.2f}, 范围: [{sim.people.rel_trans.min():.2f}, {sim.people.rel_trans.max():.2f}]")

# 运行模拟
sim.run()

print(f"\n最终感染数: {sim.results['cum_infections'][-1]}")


# ============================================================================
# 方法2：通过干预措施动态设置（在运行过程中修改）
# ============================================================================

print("\n" + "="*60)
print("方法2：通过干预措施动态设置传播参数")
print("="*60)

def dynamic_transmission_params(sim):
    '''
    在模拟运行过程中动态修改传播参数
    '''
    # 在第30天，降低年轻人的易感性（例如：开始接种疫苗）
    if sim.t == sim.day(30):
        young = sim.people.age < 30
        sim.people.rel_sus[young] *= 0.3  # 降低70%的易感性
        print(f"第30天：降低年轻人的易感性")
    
    # 在第60天，降低老年人的传播性（例如：加强防护措施）
    if sim.t == sim.day(60):
        old = sim.people.age >= 50
        sim.people.rel_trans[old] *= 0.5  # 降低50%的传播性
        print(f"第60天：降低老年人的传播性")

# 创建新的模拟
sim2 = cv.Sim(pop_size=pop_size, n_days=90, interventions=dynamic_transmission_params)
sim2.popdict = custom_popdict
sim2.reset_layer_pars()
sim2.initialize()

# 初始设置
sim2.people.rel_sus = np.random.uniform(0.8, 1.2, pop_size)
sim2.people.rel_trans = np.random.uniform(0.9, 1.1, pop_size)

print(f"初始易感性均值: {sim2.people.rel_sus.mean():.2f}")
print(f"初始传播性均值: {sim2.people.rel_trans.mean():.2f}")

sim2.run()

print(f"最终易感性均值: {sim2.people.rel_sus.mean():.2f}")
print(f"最终传播性均值: {sim2.people.rel_trans.mean():.2f}")
print(f"最终感染数: {sim2.results['cum_infections'][-1]}")


# ============================================================================
# 方法3：在创建人口时添加自定义属性，然后根据属性设置参数
# ============================================================================

print("\n" + "="*60)
print("方法3：根据自定义属性设置传播参数")
print("="*60)

def create_population_with_attributes(pop_size, layer_config):
    '''
    创建带自定义属性的人口
    
    注意：必需的基本属性只有三个（在 covasim/population.py 的 validate_popdict 函数中定义）：
    1. uid: 唯一标识符数组（整数）
    2. age: 年龄数组（浮点数）
    3. sex: 性别数组（整数，当前版本未使用，但必需）
    
    其他属性（如 contacts, layer_keys, country 等）是可选的自定义属性。
    参考：covasim/population.py 第118行 required_keys = ['uid', 'age', 'sex']
    '''
    # 创建必需的基本属性（这三个是必须的）
    uids = np.arange(pop_size, dtype=cv.default_int)
    ages = np.random.uniform(18, 65, pop_size)
    sexes = np.random.binomial(1, 0.5, pop_size)
    
    # 添加自定义属性：国家
    countries = np.random.choice(['A', 'B'], pop_size, p=[0.6, 0.4])
    
    # 添加自定义属性：健康状况（0=健康, 1=有基础疾病）
    health_status = np.random.binomial(1, 0.2, pop_size)  # 20%的人有基础疾病
    
    # 创建接触网络
    contacts = cv.Contacts()
    layer_keys = []
    
    for layer_name, config in layer_config.items():
        layer_keys.append(layer_name)
        
        if config.get('age_range') is not None:
            min_age, max_age = config['age_range']
            age_mask = (ages >= min_age) & (ages < max_age)
            indices = np.where(age_mask)[0]
        else:
            indices = None
        
        if config.get('network_type') == Enums.NetWorkType.scale_free.name:
            m = config.get('m_connections', 2)
            if indices is not None:
                layer_contacts = cv.make_scale_free_contacts(len(indices), m_connections=m, mapping=indices)
            else:
                layer_contacts = cv.make_scale_free_contacts(pop_size, m_connections=m)
        elif config.get('network_type') == Enums.NetWorkType.random.name:
            n_contacts = config.get('n_contacts', 10)
            if indices is not None:
                layer_contacts = cv.make_random_contacts(len(indices), n=n_contacts, mapping=indices)
            else:
                layer_contacts = cv.make_random_contacts(pop_size, n=n_contacts)
        
        layer = cv.Layer(**layer_contacts, label=layer_name)
        contacts.add_layer(**{layer_name: layer})
    
    # 创建人口字典（包含自定义属性）
    popdict = {
        'uid': uids,
        'age': ages,
        'sex': sexes,
        'country': countries,  # 自定义属性
        'health_status': health_status,  # 自定义属性
        'contacts': contacts,
        'layer_keys': layer_keys
    }
    
    return popdict, layer_keys

# 创建带自定义属性的人口
custom_popdict3, custom_keys3 = create_population_with_attributes(100, custom_config)

# 创建模拟
sim3 = cv.Sim(pop_size=100, n_days=90)
sim3.popdict = custom_popdict3
sim3.reset_layer_pars()
sim3.initialize()

# 将自定义属性添加到 people 对象（这样可以在干预措施中使用）
sim3.people.country = custom_popdict3['country']
sim3.people.health_status = custom_popdict3['health_status']

# 根据自定义属性设置传播参数
# 国家A：易感性高
country_A_mask = sim3.people.country == 'A'
sim3.people.rel_sus[country_A_mask] = 1.3

# 国家B：易感性低
country_B_mask = sim3.people.country == 'B'
sim3.people.rel_sus[country_B_mask] = 0.7

# 有基础疾病的人：易感性更高，传播性也更高
health_issue_mask = sim3.people.health_status == 1
sim3.people.rel_sus[health_issue_mask] *= 1.5
sim3.people.rel_trans[health_issue_mask] *= 1.3

print(f"国家A人数: {country_A_mask.sum()}, 易感性: {sim3.people.rel_sus[country_A_mask].mean():.2f}")
print(f"国家B人数: {country_B_mask.sum()}, 易感性: {sim3.people.rel_sus[country_B_mask].mean():.2f}")
print(f"有基础疾病人数: {health_issue_mask.sum()}, 易感性: {sim3.people.rel_sus[health_issue_mask].mean():.2f}, 传播性: {sim3.people.rel_trans[health_issue_mask].mean():.2f}")

sim3.run()
print(f"最终感染数: {sim3.results['cum_infections'][-1]}")


# ============================================================================
# 方法4：创建辅助函数，方便批量设置
# ============================================================================

print("\n" + "="*60)
print("方法4：使用辅助函数批量设置")
print("="*60)

def set_transmission_by_age(people, age_ranges, rel_sus_values, rel_trans_values):
    '''
    根据年龄范围批量设置传播参数
    
    Args:
        people: sim.people 对象
        age_ranges: 年龄范围列表，例如 [(0, 30), (30, 50), (50, 100)]
        rel_sus_values: 对应的易感性值列表
        rel_trans_values: 对应的传播性值列表
    '''
    for (min_age, max_age), sus, trans in zip(age_ranges, rel_sus_values, rel_trans_values):
        mask = (people.age >= min_age) & (people.age < max_age)
        people.rel_sus[mask] = sus
        people.rel_trans[mask] = trans
        print(f"年龄 {min_age}-{max_age}: 易感性={sus}, 传播性={trans}, 人数={mask.sum()}")

def set_transmission_by_custom_attribute(people, attribute_name, attribute_values, rel_sus_values, rel_trans_values):
    '''
    根据自定义属性批量设置传播参数
    
    Args:
        people: sim.people 对象
        attribute_name: 属性名称（字符串）
        attribute_values: 属性值列表
        rel_sus_values: 对应的易感性值列表
        rel_trans_values: 对应的传播性值列表
    '''
    if not hasattr(people, attribute_name):
        raise ValueError(f"People对象没有属性 '{attribute_name}'")
    
    attribute_array = getattr(people, attribute_name)
    for attr_val, sus, trans in zip(attribute_values, rel_sus_values, rel_trans_values):
        mask = attribute_array == attr_val
        people.rel_sus[mask] = sus
        people.rel_trans[mask] = trans
        print(f"{attribute_name}={attr_val}: 易感性={sus}, 传播性={trans}, 人数={mask.sum()}")

# 使用辅助函数
sim4 = cv.Sim(pop_size=100, n_days=90)
sim4.popdict = custom_popdict3
sim4.reset_layer_pars()
sim4.initialize()
sim4.people.country = custom_popdict3['country']
sim4.people.health_status = custom_popdict3['health_status']

# 按年龄设置
set_transmission_by_age(
    sim4.people,
    age_ranges=[(0, 30), (30, 50), (50, 100)],
    rel_sus_values=[1.5, 1.0, 1.8],
    rel_trans_values=[1.2, 1.0, 0.8]
)

# 按国家设置
set_transmission_by_custom_attribute(
    sim4.people,
    attribute_name='country',
    attribute_values=['A', 'B'],
    rel_sus_values=[1.3, 0.7],
    rel_trans_values=[1.0, 1.0]
)

sim4.run()
print(f"最终感染数: {sim4.results['cum_infections'][-1]}")

print("\n" + "="*60)
print("所有方法演示完成！")
print("="*60)
