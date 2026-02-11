'''
测试按 country 分组的网络生成功能
验证连接只发生在相同 country 的人之间
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

# 验证 country 分布
countries = custom_popdict['country']
unique_countries, counts = np.unique(countries, return_counts=True)
print("="*60)
print("Country 分布:")
for country, count in zip(unique_countries, counts):
    print(f"  Country {country}: {count} 人")
print("="*60)

# 创建模拟
sim = cv.Sim(pop_size=pop_size, n_days=90)
sim.popdict = custom_popdict
sim.reset_layer_pars()

# 初始化
sim.initialize()

# 将 country 属性添加到 people 对象
sim.people.country = custom_popdict['country']

# 转换为图
G = sim.people.contacts.to_graph()

# 验证连接是否只发生在相同 country 的人之间
print("\n验证连接是否只发生在相同 country 之间:")
print("-"*60)

# 获取所有边
edges = list(G.edges())
country_A = set(np.where(countries == 'A')[0])
country_B = set(np.where(countries == 'B')[0])

# 检查每条边
cross_country_edges = []
same_country_A_edges = []
same_country_B_edges = []

for edge in edges:
    u, v = edge
    u_country = countries[u]
    v_country = countries[v]
    
    if u_country != v_country:
        cross_country_edges.append(edge)
    elif u_country == 'A':
        same_country_A_edges.append(edge)
    else:
        same_country_B_edges.append(edge)

print(f"总边数: {len(edges)}")
print(f"Country A 内部连接: {len(same_country_A_edges)}")
print(f"Country B 内部连接: {len(same_country_B_edges)}")
print(f"跨 Country 连接: {len(cross_country_edges)}")

if len(cross_country_edges) == 0:
    print("\n✓ 验证通过：没有跨 country 的连接！")
else:
    print(f"\n✗ 验证失败：发现 {len(cross_country_edges)} 条跨 country 连接")
    print("前5条跨 country 连接:")
    for i, edge in enumerate(cross_country_edges[:5]):
        u, v = edge
        print(f"  {edge}: Country {countries[u]} <-> Country {countries[v]}")

# 可视化网络（按 country 着色）
print("\n生成网络可视化图...")
plt.figure(figsize=(12, 8))

# 使用 spring layout
pos = nx.spring_layout(G, k=0.3, iterations=50)

# 按 country 着色节点
node_colors = ['red' if countries[i] == 'A' else 'blue' for i in G.nodes()]

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, alpha=0.8)

# 绘制边
nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', label='Country A'),
    Patch(facecolor='blue', label='Country B')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.title('Contact Network (Red=Country A, Blue=Country B)\nConnections only within same country')
plt.axis('off')
plt.tight_layout()
plt.savefig('country_network_test.png', dpi=150, bbox_inches='tight')
print("网络图已保存为 country_network_test.png")

# 测试其他网络类型
print("\n" + "="*60)
print("测试随机网络类型:")
print("="*60)

custom_config_random = {
    'random_layer': {
        'network_type': Enums.NetWorkType.random.name,
        'n_contacts': 5,
        'beta': 0.3,
        'age_range': None,
    }   
}

popdict_random, keys_random = ContactNetwork.create_custom_population(50, custom_config_random, countries_config)
countries_random = popdict_random['country']

sim_random = cv.Sim(pop_size=50, n_days=1)
sim_random.popdict = popdict_random
sim_random.reset_layer_pars()
sim_random.initialize()

G_random = sim_random.people.contacts.to_graph()
edges_random = list(G_random.edges())

cross_country_random = []
for edge in edges_random:
    u, v = edge
    if countries_random[u] != countries_random[v]:
        cross_country_random.append(edge)

print(f"随机网络总边数: {len(edges_random)}")
print(f"跨 Country 连接: {len(cross_country_random)}")
if len(cross_country_random) == 0:
    print("✓ 随机网络验证通过：没有跨 country 的连接！")
else:
    print(f"✗ 随机网络验证失败：发现 {len(cross_country_random)} 条跨 country 连接")

print("\n" + "="*60)
print("测试微结构化网络类型:")
print("="*60)

custom_config_micro = {
    'micro_layer': {
        'network_type': Enums.NetWorkType.microstructured.name,
        'cluster_size': 3.0,
        'beta': 0.3,
        'age_range': None,
    }   
}

popdict_micro, keys_micro = ContactNetwork.create_custom_population(50, custom_config_micro, countries_config)
countries_micro = popdict_micro['country']

sim_micro = cv.Sim(pop_size=50, n_days=1)
sim_micro.popdict = popdict_micro
sim_micro.reset_layer_pars()
sim_micro.initialize()

G_micro = sim_micro.people.contacts.to_graph()
edges_micro = list(G_micro.edges())

cross_country_micro = []
for edge in edges_micro:
    u, v = edge
    if countries_micro[u] != countries_micro[v]:
        cross_country_micro.append(edge)

print(f"微结构化网络总边数: {len(edges_micro)}")
print(f"跨 Country 连接: {len(cross_country_micro)}")
if len(cross_country_micro) == 0:
    print("✓ 微结构化网络验证通过：没有跨 country 的连接！")
else:
    print(f"✗ 微结构化网络验证失败：发现 {len(cross_country_micro)} 条跨 country 连接")

print("\n" + "="*60)
print("所有测试完成！")
print("="*60)
