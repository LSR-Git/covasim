import numpy as np
import covasim as cv


def add_cross_layer(
    popdict,
    frac_travelers=0.03,
    n_cross_per_person=2,
    cross_beta=0.6,
    cross_layer_seed=42,
    region_a=None,
    region_b=None,
):
    '''
    在已有 popdict 上添加跨区接触层（仅流动人口有跨区边）。

    会就地修改 popdict：新增 'crosser' 布尔属性；在 contacts 中新增 'cross' 层；
    若有至少两个区域则向 popdict['layer_keys'] 追加 'cross'。

    注：本函数仅做静态网络与 crosser 标记。若需「跨境时 cross 权重 1、base 权重 0，回国时相反」
    以及「每日从境内候鸟中按比例随机出境、境外停留 1–7 天」等动态行为，请在仿真中使用
    CrosserTravel 干预（见 compose_intervention.py）实现。

    Args:
        popdict: 人口字典，须含 'country' 数组、'contacts' 与 'layer_keys'
        frac_travelers: 每区流动人口比例 (0~1)
        n_cross_per_person: 每个流动者在本层的跨区边数
        cross_beta: 跨区边的传播权重
        cross_layer_seed: 随机种子，便于复现
        region_a: 第一个区域名，默认取 unique_countries[0]
        region_b: 第二个区域名，默认取 unique_countries[1]

    Returns:
        popdict: 修改后的 popdict（就地修改，返回值便于链式调用）
    '''
    pop_size = len(popdict['country'])

    popdict['crosser'] = np.zeros(pop_size, dtype=bool)
    countries = popdict['country']
    unique_countries = np.unique(countries)

    if len(unique_countries) < 2:
        return popdict

    name_a = region_a if region_a is not None else unique_countries[0]
    name_b = region_b if region_b is not None else unique_countries[1]
    inds_A = np.where(countries == name_a)[0]
    inds_B = np.where(countries == name_b)[0]

    if len(inds_A) == 0 or len(inds_B) == 0:
        return popdict

    rng = np.random.RandomState(cross_layer_seed)
    n_travelers_A = max(1, int(frac_travelers * len(inds_A)))
    n_travelers_B = max(1, int(frac_travelers * len(inds_B)))
    travelers_A = rng.choice(inds_A, size=n_travelers_A, replace=False)
    travelers_B = rng.choice(inds_B, size=n_travelers_B, replace=False)
    popdict['crosser'][travelers_A] = True
    popdict['crosser'][travelers_B] = True

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
    popdict['contacts'].add_layer(cross=cross_layer)
    popdict['layer_keys'].append('cross')

    return popdict
