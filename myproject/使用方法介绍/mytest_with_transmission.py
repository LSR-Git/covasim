'''
演示如何在自定义人口中设置个人传播参数
'''
import numpy as np
import covasim as cv
import Enums
import ContactNetwork
import matplotlib.pyplot as plt
import networkx as nx

# 创建自定义人口配置
custom_config_test = {
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

# 创建自定义人口
custom_popdict, custom_keys = ContactNetwork.create_custom_population(pop_size, custom_config_test, countries_config)

# 创建模拟
sim = cv.Sim(pop_size=pop_size, n_days=90)
sim.popdict = custom_popdict
sim.reset_layer_pars()

# 初始化（这一步会创建 people 对象并设置默认的传播参数）
sim.initialize()

# ============================================================================
# 方法1：根据年龄设置传播参数
# ============================================================================
print("="*60)
print("方法1：根据年龄设置传播参数")
print("="*60)

young = sim.people.age < 30
middle = (sim.people.age >= 30) & (sim.people.age < 50)
old = sim.people.age >= 50

# 设置易感性（rel_sus）和传播性（rel_trans）
sim.people.rel_sus[young] = 1.5   # 年轻人易感性高
sim.people.rel_trans[young] = 1.2  # 年轻人传播性强

sim.people.rel_sus[middle] = 1.0   # 中年人正常
sim.people.rel_trans[middle] = 1.0

sim.people.rel_sus[old] = 1.8      # 老年人易感性高
sim.people.rel_trans[old] = 0.8     # 但传播性低

print(f"年轻人 ({young.sum()}人): 易感性={sim.people.rel_sus[young].mean():.2f}, 传播性={sim.people.rel_trans[young].mean():.2f}")
print(f"中年人 ({middle.sum()}人): 易感性={sim.people.rel_sus[middle].mean():.2f}, 传播性={sim.people.rel_trans[middle].mean():.2f}")
print(f"老年人 ({old.sum()}人): 易感性={sim.people.rel_sus[old].mean():.2f}, 传播性={sim.people.rel_trans[old].mean():.2f}")

# ============================================================================
# 方法2：根据自定义属性（国家）设置传播参数
# ============================================================================
print("\n" + "="*60)
print("方法2：根据国家设置传播参数")
print("="*60)

# 将自定义属性添加到 people 对象
sim.people.country = custom_popdict['country']

# 根据国家设置不同的传播参数
country_A = sim.people.country == 'A'
country_B = sim.people.country == 'B'

# 国家A：易感性高
sim.people.rel_sus[country_A] = 1.3
# 国家B：易感性低（例如：更好的防护措施）
sim.people.rel_sus[country_B] = 0.7

print(f"国家A ({country_A.sum()}人): 易感性={sim.people.rel_sus[country_A].mean():.2f}")
print(f"国家B ({country_B.sum()}人): 易感性={sim.people.rel_sus[country_B].mean():.2f}")

# ============================================================================
# 方法3：随机设置（模拟个体差异）
# ============================================================================
print("\n" + "="*60)
print("方法3：随机设置传播参数（模拟个体差异）")
print("="*60)

np.random.seed(42)
# 易感性：0.5-1.5之间随机
sim.people.rel_sus = np.random.uniform(0.5, 1.5, pop_size)
# 传播性：0.8-1.2之间随机
sim.people.rel_trans = np.random.uniform(0.8, 1.2, pop_size)

print(f"易感性 - 均值: {sim.people.rel_sus.mean():.2f}, 范围: [{sim.people.rel_sus.min():.2f}, {sim.people.rel_sus.max():.2f}]")
print(f"传播性 - 均值: {sim.people.rel_trans.mean():.2f}, 范围: [{sim.people.rel_trans.min():.2f}, {sim.people.rel_trans.max():.2f}]")

# ============================================================================
# 运行模拟
# ============================================================================
print("\n" + "="*60)
print("运行模拟...")
print("="*60)

sim.run()

print(f"最终感染数: {sim.results['cum_infections'][-1]}")
print(f"最终死亡数: {sim.results['cum_deaths'][-1]}")

# ============================================================================
# 可视化网络（可选）
# ============================================================================
# 转换为包含所有层的图
G = sim.people.contacts.to_graph()

# 绘制
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, k=0.3, iterations=50)
nx.draw(G, pos, with_labels=False, node_size=30, alpha=0.6, width=0.5)
plt.title('Contact Network with Custom Transmission Parameters')
plt.show()

# ============================================================================
# 绘制结果（可选）
# ============================================================================
sim.plot()
