# 组合干预情景用到的辅助函数与 subtarget 构造
import numpy as np

# 默认区域键与名称（与 compose_intervention 中一致，可按需覆盖）
REGION_KEY = 'position'
REGION_NAME_A = 'A'
REGION_NAME_B = 'B'


def _default_region_key(region_key):
    return REGION_KEY if region_key is None else region_key


def _default_region_name_a(region_name_a):
    return REGION_NAME_A if region_name_a is None else region_name_a


def _default_region_name_b(region_name_b):
    return REGION_NAME_B if region_name_b is None else region_name_b


def is_position_a(sim, region_key=None, region_name=None):
    """当前所在地为 A 区（默认 position=='A'）。"""
    rk = _default_region_key(region_key)
    rn = REGION_NAME_A if region_name is None else region_name
    return np.asarray(getattr(sim.people, rk)) == rn


def is_position_b(sim, region_key=None, region_name=None):
    """当前所在地为 B 区（默认 position=='B'）。"""
    rk = _default_region_key(region_key)
    rn = REGION_NAME_B if region_name is None else region_name
    return np.asarray(getattr(sim.people, rk)) == rn


def get_crosser_inds(sim, region_key=None, region_name_a=None):
    """候鸟：当前在 A 区且为跨境人员（crosser）。"""
    in_a = is_position_a(sim, region_key=region_key, region_name=region_name_a or REGION_NAME_A)
    return np.where(in_a & sim.people.crosser)[0]


def is_country_a_crosser(sim, region_name_a=None):
    """A 区户籍且为跨境人员（crosser），用于边境检测仅对 A 区候鸟生效。"""
    rn = _default_region_name_a(region_name_a)
    return (np.asarray(sim.people.country) == rn) & np.asarray(sim.people.crosser)


def is_position_a_crosser(sim, region_key=None, region_name_a=None):
    """当前在 A 区且为跨境人员（crosser），用于边境检测包含所有在 A 区的候鸟。"""
    in_a = is_position_a(sim, region_key=region_key, region_name=region_name_a or REGION_NAME_A)
    return in_a & np.asarray(sim.people.crosser)


def make_subtarget_position(region_key=None, region_name=None):
    """构造按区域筛选的 subtarget（检测/疫苗接种等共用）。"""
    rk = _default_region_key(region_key)
    rn = REGION_NAME_A if region_name is None else region_name

    def inds(sim):
        return np.arange(sim.n)

    def vals(sim):
        return (np.asarray(getattr(sim.people, rk)) == rn).astype(float)

    return {'inds': inds, 'vals': vals}


def make_subtarget_crosser(crosser_prob=0.5, region_key=None, region_name_a=None):
    """边境检测 subtarget：在 A 区的候鸟为 crosser_prob，其余人 0。"""
    rk = _default_region_key(region_key)
    rna = _default_region_name_a(region_name_a)

    def inds(sim):
        return np.arange(sim.n)

    def vals(sim):
        in_a_crosser = is_position_a_crosser(sim, region_key=rk, region_name_a=rna)
        return np.where(in_a_crosser, float(crosser_prob), 0.0).astype(float)

    return {'inds': inds, 'vals': vals}


def make_subtarget_position_exclude_undocumented(region_key=None, region_name=None):
    """构造按区域筛选且排除 undocumented 的 subtarget（case05 境内检测用）。"""
    rk = _default_region_key(region_key)
    rn = REGION_NAME_A if region_name is None else region_name

    def inds(sim):
        undocumented = getattr(sim.people, 'undocumented', np.zeros(sim.n, dtype=bool))
        in_region = (np.asarray(getattr(sim.people, rk)) == rn)
        return np.where(in_region & ~undocumented)[0]

    def vals(sim):
        return np.ones(len(inds(sim)), dtype=float)

    return {'inds': inds, 'vals': vals}


def make_subtarget_crosser_exclude_undocumented(crosser_prob=0.5, region_key=None, region_name_a=None):
    """边境检测 subtarget：在 A 区的候鸟且非 undocumented 为 crosser_prob，其余 0（case05 用）。"""
    rk = _default_region_key(region_key)
    rna = _default_region_name_a(region_name_a)

    def inds(sim):
        return np.arange(sim.n)

    def vals(sim):
        undocumented = getattr(sim.people, 'undocumented', np.zeros(sim.n, dtype=bool))
        in_a_crosser = is_position_a_crosser(sim, region_key=rk, region_name_a=rna)
        return np.where(in_a_crosser & ~undocumented, float(crosser_prob), 0.0).astype(float)

    return {'inds': inds, 'vals': vals}


def sequence_random(people):
    """疫苗接种顺序：随机排列。"""
    return np.random.permutation(len(people.uid))


def sequence_crosser_first_then_random_a(people, region_key=None, region_name_a=None):
    """A 区优先候鸟接种，多余剂量对 A 区其他人随机。"""
    rk = _default_region_key(region_key)
    rna = _default_region_name_a(region_name_a)
    is_a = np.asarray(getattr(people, rk)) == rna
    inds_crosser = np.where(is_a & people.crosser)[0]
    inds_other_a = np.where(is_a & ~people.crosser)[0]
    np.random.shuffle(inds_crosser)
    np.random.shuffle(inds_other_a)
    return np.concatenate([inds_crosser, inds_other_a])


def create_vaccination_schedule(total_doses, daily_doses, start_day=0):
    """
    创建疫苗接种计划：从 start_day 开始，每天接种 daily_doses 剂，直到总疫苗量用完。

    Args:
        total_doses: 总疫苗量
        daily_doses: 每天接种数量
        start_day: 开始接种的日期（天数或日期字符串）

    Returns:
        dict: num_doses 字典，格式为 {day: doses, ...}

    Example:
        num_doses = create_vaccination_schedule(total_doses=10000, daily_doses=500, start_day=0)
        # 返回 {0: 500, 1: 500, ..., 19: 500}（共20天，每天500剂）
    """
    num_doses_dict = {}
    remaining = total_doses
    day = start_day
    while remaining > 0:
        doses_today = min(daily_doses, remaining)
        num_doses_dict[day] = doses_today
        remaining -= doses_today
        day += 1
    return num_doses_dict
