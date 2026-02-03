
import numpy as np
import covasim as cv
import sciris as sc
# 方法1：创建完全自定义的人口（推荐）
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