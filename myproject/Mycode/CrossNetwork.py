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


def add_cross_layer_multilayer(
    popdict,
    frac_travelers=0.03,
    n_cross_per_person=2,
    cross_beta=0.6,
    frac_work=0.7,
    frac_visit=0.25,
    frac_undocumented=0.05,
    cross_layer_seed=42,
    region_a=None,
    region_b=None,
):
    '''
    为多层网络（home/school/work/community）添加基于出行目的的跨区层。

    候鸟按目的分为务工、探亲、偷渡，预建 cross_work、cross_community、cross_home 三个静态跨区层。
    务工：cross_work + cross_community；探亲：cross_home + cross_community；偷渡：仅 cross_community。
    跨境时的激活由 CrosserTravelMultilayer 通过 beta 控制实现。

    Args:
        popdict: 人口字典，须含 country、age、contacts、layer_keys，且 contacts 含 home/school/work/community
        frac_travelers: 每区流动人口比例 (0~1)
        n_cross_per_person: 每个流动者在各目标层的跨区边数
        cross_beta: 跨区边的传播权重
        frac_work: 务工候鸟比例（与 frac_visit、frac_undocumented 归一化后使用）
        frac_visit: 探亲候鸟比例
        frac_undocumented: 偷渡候鸟比例
        cross_layer_seed: 随机种子
        region_a: 区域 A 名称
        region_b: 区域 B 名称

    Returns:
        popdict: 修改后的 popdict（就地修改）
    '''
    pop_size = len(popdict['country'])
    countries = np.asarray(popdict['country'])
    ages = np.asarray(popdict['age'])
    unique_countries = np.unique(countries)

    if len(unique_countries) < 2:
        return popdict

    name_a = region_a if region_a is not None else unique_countries[0]
    name_b = region_b if region_b is not None else unique_countries[1]
    inds_A = np.where(countries == name_a)[0]
    inds_B = np.where(countries == name_b)[0]

    if len(inds_A) == 0 or len(inds_B) == 0:
        return popdict

    # B 区目标人员：工作层 22-65 岁，社区/家庭层全员
    inds_B_work = inds_B[(ages[inds_B] >= 22) & (ages[inds_B] < 65)]
    if len(inds_B_work) == 0:
        inds_B_work = inds_B
    inds_A_work = inds_A[(ages[inds_A] >= 22) & (ages[inds_A] < 65)]
    if len(inds_A_work) == 0:
        inds_A_work = inds_A

    rng = np.random.RandomState(cross_layer_seed)
    n_travelers_A = max(1, int(frac_travelers * len(inds_A)))
    n_travelers_B = max(1, int(frac_travelers * len(inds_B)))
    travelers_A = rng.choice(inds_A, size=n_travelers_A, replace=False)
    travelers_B = rng.choice(inds_B, size=n_travelers_B, replace=False)

    # 归一化目的比例
    total_frac = frac_work + frac_visit + frac_undocumented
    if total_frac <= 0:
        p_work, p_visit, p_undoc = 1.0, 0.0, 0.0
    else:
        p_work = frac_work / total_frac
        p_visit = frac_visit / total_frac
        p_undoc = frac_undocumented / total_frac

    popdict['crosser'] = np.zeros(pop_size, dtype=bool)
    popdict['crosser'][travelers_A] = True
    popdict['crosser'][travelers_B] = True

    # crosser_purpose: 'work' | 'visit' | 'undocumented'
    crosser_purpose = np.empty(pop_size, dtype=object)
    crosser_purpose[:] = ''
    for tinds in [travelers_A, travelers_B]:
        n = len(tinds)
        r = rng.random(n)
        work_mask = r < p_work
        visit_mask = (r >= p_work) & (r < p_work + p_visit)
        undoc_mask = r >= p_work + p_visit
        crosser_purpose[tinds] = np.where(work_mask, 'work', np.where(visit_mask, 'visit', 'undocumented'))
    popdict['crosser_purpose'] = crosser_purpose

    # 预建跨区层
    def make_cross_edges(crosser_inds, partner_inds, rng, n_per_person):
        p1_list, p2_list = [], []
        for c in crosser_inds:
            partners = rng.choice(partner_inds, size=n_per_person, replace=True)
            for p in partners:
                p1_list.append(c)
                p2_list.append(p)
        return np.array(p1_list, dtype=cv.default_int), np.array(p2_list, dtype=cv.default_int)

    # cross_work: 务工候鸟 <-> 对方工作层人员
    work_A = travelers_A[crosser_purpose[travelers_A] == 'work']
    work_B = travelers_B[crosser_purpose[travelers_B] == 'work']
    p1_w, p2_w = [], []
    if len(work_A) > 0:
        a1, a2 = make_cross_edges(work_A, inds_B_work, rng, n_cross_per_person)
        p1_w.extend(a1)
        p2_w.extend(a2)
    if len(work_B) > 0:
        b1, b2 = make_cross_edges(work_B, inds_A_work, rng, n_cross_per_person)
        p1_w.extend(b1)
        p2_w.extend(b2)
    if len(p1_w) > 0:
        p1_w = np.array(p1_w, dtype=cv.default_int)
        p2_w = np.array(p2_w, dtype=cv.default_int)
        beta_w = np.full(len(p1_w), cross_beta, dtype=cv.default_float)
        layer_w = cv.Layer(p1=p1_w, p2=p2_w, beta=beta_w, label='cross_work')
        popdict['contacts'].add_layer(cross_work=layer_w)
        popdict['layer_keys'].append('cross_work')

    # cross_community: 所有候鸟 <-> 对方全员
    p1_c, p2_c = [], []
    if len(travelers_A) > 0:
        a1, a2 = make_cross_edges(travelers_A, inds_B, rng, n_cross_per_person)
        p1_c.extend(a1)
        p2_c.extend(a2)
    if len(travelers_B) > 0:
        b1, b2 = make_cross_edges(travelers_B, inds_A, rng, n_cross_per_person)
        p1_c.extend(b1)
        p2_c.extend(b2)
    if len(p1_c) > 0:
        p1_c = np.array(p1_c, dtype=cv.default_int)
        p2_c = np.array(p2_c, dtype=cv.default_int)
        beta_c = np.full(len(p1_c), cross_beta, dtype=cv.default_float)
        layer_c = cv.Layer(p1=p1_c, p2=p2_c, beta=beta_c, label='cross_community')
        popdict['contacts'].add_layer(cross_community=layer_c)
        popdict['layer_keys'].append('cross_community')

    # cross_home: 探亲候鸟 <-> 对方全员
    visit_A = travelers_A[crosser_purpose[travelers_A] == 'visit']
    visit_B = travelers_B[crosser_purpose[travelers_B] == 'visit']
    p1_h, p2_h = [], []
    if len(visit_A) > 0:
        a1, a2 = make_cross_edges(visit_A, inds_B, rng, n_cross_per_person)
        p1_h.extend(a1)
        p2_h.extend(a2)
    if len(visit_B) > 0:
        b1, b2 = make_cross_edges(visit_B, inds_A, rng, n_cross_per_person)
        p1_h.extend(b1)
        p2_h.extend(b2)
    if len(p1_h) > 0:
        p1_h = np.array(p1_h, dtype=cv.default_int)
        p2_h = np.array(p2_h, dtype=cv.default_int)
        beta_h = np.full(len(p1_h), cross_beta, dtype=cv.default_float)
        layer_h = cv.Layer(p1=p1_h, p2=p2_h, beta=beta_h, label='cross_home')
        popdict['contacts'].add_layer(cross_home=layer_h)
        popdict['layer_keys'].append('cross_home')

    return popdict
