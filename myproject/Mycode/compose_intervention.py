import sys
import os

# 将项目根目录加入 path，使 import covasim 使用本地的 E:\my_paper\covasim\covasim
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import covasim as cv
import covasim.utils as cvu
import Enums
import sciris as sc
import os
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import ContactNetwork
import CrossNetwork
import MyPlot

# 设置 matplotlib 显示中文（Windows 常用 SimHei / 微软雅黑）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框

# 定义层级配置
custom_config={
    'base': {
        'network_type': Enums.NetWorkType.scale_free.name,
        'n_contacts': None,
        'm_connections': 3,  # 每个人加入网络时连接的边数 (决定了网络的平均密度)
        'age_range': None,
        'cluster_size': None
    }
}

# 定义国家配置：比例式 2:1 表示 A 占 2/3、B 占 1/3（也可写概率式如 {'A': 0.5, 'B': 0.5}）
countries_config = {
    'A': 2,
    'B': 1
}

# 随机种子：可分别指定，便于复现或做敏感性分析
seed_population = 0      # 人口与区内接触网（create_custom_population）
seed_cross_layer = 42    # 跨区层（流动者选取与跨区边）

# 创建自定义人口（pop_size 须与下方 custom_pars['pop_size'] 一致）
pop_size = 30000
popdict_base, custom_keys = ContactNetwork.create_custom_population(
    pop_size, custom_config, countries_config, seed=seed_population
)

# 创建自定义参数（pop_size 须与上面 create_custom_population 的 pop_size 一致）
custom_pars = {
    # Population parameters
    'pop_size': pop_size,
    'pop_infected': {'A':0, 'B': 20},
    'pop_infected_region_key': 'country',

    # Simulation parameters
    'start_day': '2021-07-04',
    'n_days': 180,

    # Rescaling parameters
    'rescale': False,        # 禁用动态调整人口大小

    # Network parameters（base 与 cross 层权重；cross 可设低一些表示跨区接触强度）
    'beta_layer': {'base': 1.0, 'cross': 0.6},
    # Basic disease transmission parameters
    'beta': 0.036,
}

# 无跨区层时 A/B 两区不接触；加上跨区层后可观察跨境传播
# frac_travelers 为每区流动人口比例 (0~1)，0.01 表示每区 1% 为流动者，总跨区人数约等于总人口的 1%
popdict = CrossNetwork.add_cross_layer(
    popdict_base, frac_travelers=0.01, n_cross_per_person=10, cross_beta=0.6, cross_layer_seed=seed_cross_layer
)

# ========== 干预开始日（可按需修改） ==========
intervention_start = 0
_scenario_a_start_round1 = 0
_scenario_a_start_round2 = 17
_scenario_a_start_round3 = 34
_scenario_a_start_round4 = 42

# 统一：所有干预的“A/B 区”均按当前所在地 position=='A'/'B' 判定（与 country 户籍区分，便于跨境场景）
_region_key = 'position'
_region_name_a = 'A'
_region_name_b = 'B'

def _is_position_a(sim):
    return (np.asarray(getattr(sim.people, _region_key)) == _region_name_a)


def _is_position_b(sim):
    """当前所在地为 B 区（position=='B'）。"""
    return (np.asarray(getattr(sim.people, _region_key)) == _region_name_b)


# 仅 A 区有资格的 subtarget（检测/追踪/疫苗接种等共用）
_subtarget_position_a = {
    'inds': lambda sim: np.arange(sim.n),
    'vals': lambda sim: _is_position_a(sim).astype(float),
}


def _get_crosser_inds(sim):
    """候鸟：当前在 A 区且为跨境人员（crosser）。"""
    return np.where(_is_position_a(sim) & sim.people.crosser)[0]


def _is_country_a_crosser(sim):
    """A 区户籍且为跨境人员（crosser），用于边境检测仅对 A 区候鸟生效。"""
    return (np.asarray(sim.people.country) == _region_name_a) & np.asarray(sim.people.crosser)


def _is_position_a_crosser(sim):
    """当前在 A 区且为跨境人员（crosser），用于边境检测包含所有在 A 区的候鸟（包括 A 区候鸟和 B 区候鸟）。"""
    return _is_position_a(sim) & np.asarray(sim.people.crosser)


# 边境检测 subtarget：所有在 A 区的候鸟（position=='A' 且 crosser，包括 A 区候鸟和 B 区候鸟）有 50% 检测概率，其余人 0
_subtarget_crosser = {
    'inds': lambda sim: np.arange(sim.n),
    'vals': lambda sim: np.where(_is_position_a_crosser(sim), 0.5, 0.0).astype(float),
}

# ========== 1. 检测隔离：两种检测 ==========
# 1a. 境内检测：对 A 区所在人员（position=='A'）的日常国内检测，20% 有症状、5% 无症状，延迟 2 天
test_isolate_a = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.05,
    start_day=intervention_start,
    test_delay=2,
    subtarget=_subtarget_position_a,
)

# 1b. 边境检测：对所有来到 A 区的候鸟（position=='A' 且 crosser=True，包括 A 区候鸟和 B 区候鸟）的例行检测，50% 概率，延迟 1 天（本情景自第 0 天开始）

test_isolate_crosser = cv.test_prob(
    symp_prob=0.8,
    asymp_prob=0.1,
    start_day=_scenario_a_start_round1,
    test_delay=1,
    subtarget=_subtarget_crosser,
)

# 旧版（保留兼容）
test_isolate = test_isolate_a

# ========== 2. 接触者追踪：仅对 A 区检测 + 50% 接触者追踪，追踪延迟 2 天 ==========
# 仅 position=='A' 者被检测，确诊者（均为 A 区）的接触者被追踪并隔离
# 自定义接触者追踪：只追踪 A 区的接触者（position=='A'），避免追踪到 B 区人员
class ContactTracingAOnly(cv.contact_tracing):
    '''接触者追踪：只追踪 A 区的接触者（position=='A'），避免追踪到 B 区人员。'''
    def __init__(self, region_key='position', region_name='A', **kwargs):
        super().__init__(**kwargs)
        self.region_key = region_key
        self.region_name = region_name

    def notify_contacts(self, sim, contacts):
        '''只通知 A 区的接触者'''
        is_dead = np.where(sim.people.dead)[0]  # 已死亡人员的索引
        position = getattr(sim.people, self.region_key, None)
        if position is None:
            # 如果没有 position 属性，回退到原始行为
            super().notify_contacts(sim, contacts)
            return
        
        is_in_a = (np.asarray(position) == self.region_name)
        for trace_time, contact_inds in contacts.items():
            contact_inds = np.setdiff1d(contact_inds, is_dead)  # 排除已死亡人员
            # 只通知 A 区的接触者
            contact_inds_a = contact_inds[is_in_a[contact_inds]]
            if len(contact_inds_a) > 0:
                sim.people.known_contact[contact_inds_a] = True
                sim.people.date_known_contact[contact_inds_a] = np.fmin(
                    sim.people.date_known_contact[contact_inds_a], 
                    sim.t + trace_time
                )
                sim.people.schedule_quarantine(
                    contact_inds_a, 
                    start_date=sim.t + trace_time, 
                    period=self.quar_period - trace_time
                )
        return

test_for_ct = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.05,
    start_day=_scenario_a_start_round2,
    test_delay=2,
    subtarget=_subtarget_position_a,
)
contact_tracing_50 = ContactTracingAOnly(
    trace_probs=0.2,
    trace_time=2,
    start_day=_scenario_a_start_round2,
    region_key=_region_key,
    region_name=_region_name_a,
)
intervention_contact_tracing = [test_for_ct, contact_tracing_50]

# ========== 场景三专用：第三阶段（round3 起）A 区检测/追踪概率提升一倍 ==========
# 境内检测：阶段 1–2 为 0.2/0.05、阶段 3 为 0.4/0.1；接触者追踪：阶段 1–2 为 0.2、阶段 3 为 0.4
test_isolate_a_case03_phase12 = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.05,
    start_day=intervention_start,
    end_day=_scenario_a_start_round3 - 1,
    test_delay=2,
    subtarget=_subtarget_position_a,
)
test_isolate_a_case03_phase3 = cv.test_prob(
    symp_prob=0.4,
    asymp_prob=0.1,
    start_day=_scenario_a_start_round3,
    test_delay=2,
    subtarget=_subtarget_position_a,
)
test_for_ct_case03_phase12 = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.05,
    start_day=_scenario_a_start_round2,
    end_day=_scenario_a_start_round3 - 1,
    test_delay=2,
    subtarget=_subtarget_position_a,
)
test_for_ct_case03_phase3 = cv.test_prob(
    symp_prob=0.4,
    asymp_prob=0.1,
    start_day=_scenario_a_start_round3,
    test_delay=2,
    subtarget=_subtarget_position_a,
)
contact_tracing_50_case03_phase12 = ContactTracingAOnly(
    trace_probs=0.2,
    trace_time=2,
    start_day=_scenario_a_start_round2,
    end_day=_scenario_a_start_round3 - 1,
    region_key=_region_key,
    region_name=_region_name_a,
)
contact_tracing_50_case03_phase3 = ContactTracingAOnly(
    trace_probs=0.4,
    trace_time=2,
    start_day=_scenario_a_start_round3,
    region_key=_region_key,
    region_name=_region_name_a,
)

# ========== 3. 疫苗接种 ==========
# 3a. A 区随机接种 3000 剂（干预开始日当天 3000 剂，仅 position=='A' 有资格）
def _sequence_random(people):
    return np.random.permutation(len(people.uid))

# 3b. A 区 300 剂：优先对候鸟（position=='A' 且 crosser）接种，多余剂量对 A 区其他人员随机接种
def _sequence_crosser_first_then_random_a(people):
    is_a = (np.asarray(getattr(people, _region_key)) == _region_name_a)
    inds_crosser = np.where(is_a & people.crosser)[0]
    inds_other_a = np.where(is_a & ~people.crosser)[0]
    np.random.shuffle(inds_crosser)
    np.random.shuffle(inds_other_a)
    return np.concatenate([inds_crosser, inds_other_a])

# 辅助函数：根据总疫苗量和每天接种数量生成 num_doses 字典
def create_vaccination_schedule(total_doses, daily_doses, start_day=0):
    '''
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
    '''
    num_doses_dict = {}
    remaining = total_doses
    day = start_day
    while remaining > 0:
        doses_today = min(daily_doses, remaining)
        num_doses_dict[day] = doses_today
        remaining -= doses_today
        day += 1
    return num_doses_dict

vaccinate_a = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses=create_vaccination_schedule(total_doses=10000, daily_doses=10000, start_day=0),  # 第0天一次性接种10000剂
    sequence=_sequence_random,
    subtarget=_subtarget_position_a,
)

# 示例：总疫苗量10000剂，每天接种500剂，从第0天开始（共20天）
# vaccinate_a_distributed = cv.vaccinate_num(
#     vaccine='pfizer',
#     num_doses=create_vaccination_schedule(total_doses=10000, daily_doses=500, start_day=0),
#     sequence=_sequence_random,
#     subtarget=_subtarget_position_a,
# )

# A 区第一批疫苗（本情景第 0 天发放；若用 10000 剂则 A 区疫情被压制得很低，可改为 3000 使 A 区疫情更明显）
vaccinate_a_10k = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={_scenario_a_start_round1: 10000},
    sequence=_sequence_random,
    subtarget=_subtarget_position_a,
)
# ================== 第二阶段疫苗接种======================
vaccinate_a_10k_round1_2 = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={_scenario_a_start_round1: 10000,_scenario_a_start_round2: 10000},
    sequence=_sequence_random,
    subtarget=_subtarget_position_a,
)
# 场景三：A 区三批疫苗（第 0/17/34 天各 10000 剂）
vaccinate_a_10k_round1_2_3 = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={
        _scenario_a_start_round1: 10000,
        _scenario_a_start_round2: 10000,
        _scenario_a_start_round3: 10000,
    },
    sequence=_sequence_random,
    subtarget=_subtarget_position_a,
)
# B 区疫苗接种（仅场景三：第三阶段起 5000 剂）
_subtarget_position_b = {
    'inds': lambda sim: np.arange(sim.n),
    'vals': lambda sim: _is_position_b(sim).astype(float),
}
vaccinate_b_5000 = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={_scenario_a_start_round3: 5000},
    sequence=_sequence_random,
    subtarget=_subtarget_position_b,
)

# ========== 4. 境内流动限制：对 A 区（position=='A'）减少 50% 的 base 层接触边 ==========
# 使用内置 clip_edges 对 base 层整体减半（若需仅 A 区减边，可改用下方 reduce_region_a_contacts_50）
clip_base_50 = cv.clip_edges(days=intervention_start, changes=0.5, layers='base')

# 仅 A 区 base 层接触减半的自定义干预（可选，与上面统一用 position 判定 A 区）
class reduce_region_a_contacts_50(cv.Intervention):
    '''对 A 区域（默认按 position）人员的 base 层接触边减少 50%，在 start_day 生效。'''
    def __init__(self, start_day=10, region_key=None, region_name=None, fraction=0.5, **kwargs):
        super().__init__(**kwargs)
        self.start_day = start_day
        self.region_key = region_key if region_key is not None else _region_key
        self.region_name = region_name if region_name is not None else _region_name_a
        self.fraction = fraction
        self._stored_contacts = None
        self._applied = False

    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)

    def apply(self, sim):
        if sim.t != self.start_day or self._applied:
            return
        if 'base' not in sim.people.contacts:
            return
        region = getattr(sim.people, self.region_key, None)
        if region is None:
            return
        in_a = (region == self.region_name)
        layer = sim.people.contacts['base']
        p1, p2 = layer['p1'][:], layer['p2'][:]
        edge_in_a = in_a[p1] | in_a[p2]
        n_total = edge_in_a.sum()
        if n_total == 0:
            return
        n_remove = int(n_total * (1 - self.fraction))
        if n_remove <= 0:
            return
        inds_all = np.where(edge_in_a)[0]
        np.random.shuffle(inds_all)
        to_remove = inds_all[:n_remove]
        self._stored_contacts = layer.pop_inds(to_remove)
        self._applied = True

# 使用仅 A 区减边的干预（二选一）
clip_base_50_region_a = reduce_region_a_contacts_50(start_day=_scenario_a_start_round2)

# ========== 5. 跨境流动限制：将候鸟比例清零 ==========
# 方式一：建 sim 时直接使用无跨区层的人口（不调用 add_cross_layer 或 frac_travelers=0）
# popdict_no_cross = CrossNetwork.add_cross_layer(popdict_base, frac_travelers=0, ...)  # 无跨区边
# 方式二：若已有跨区层，可在干预日移除 cross 层（需自定义干预）。这里仅提供方式一，按需替换上面的 popdict。
# 使用无跨境时，将上面 popdict 改为：
# popdict = CrossNetwork.add_cross_layer(popdict_base, frac_travelers=0, n_cross_per_person=10, cross_beta=0.6, cross_layer_seed=seed_cross_layer)

# ========== 5b. 候鸟动态跨境：每日境内候鸟按比例出境，境外 1–7 天，跨境时 cross 权重 1/base 权重 0，回国时相反 ==========


class CrosserTravel(cv.Intervention):
    '''候鸟动态跨境：每日先让到期者回国，再从境内候鸟中按比例随机选人出境（境外停留 duration_min~duration_max 天）；
    跨境时 cross 层权重有效、base 层权重 0，回国时 base 有效、cross 0。
    end_day_outbound：若指定，该日及之后不再派出新出境人员，仅保留到期回国逻辑。'''
    def __init__(
        self,
        frac_cross_per_day=0.1,
        duration_min=1,
        duration_max=7,
        start_day=0,
        end_day_outbound=None,
        region_key=None,
        region_name_a=None,
        region_name_b=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frac_cross_per_day = frac_cross_per_day
        self.duration_min = int(duration_min)
        self.duration_max = int(duration_max)
        self.start_day = start_day
        self.end_day_outbound = end_day_outbound
        self.region_key = region_key if region_key is not None else _region_key
        self.region_name_a = region_name_a if region_name_a is not None else _region_name_a
        self.region_name_b = region_name_b if region_name_b is not None else _region_name_b
        self._return_day = None
        self._cross_beta = None

    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)
        if self.end_day_outbound is not None:
            self.end_day_outbound = sim.day(self.end_day_outbound)
        n = sim.n
        self._return_day = np.full(n, np.nan, dtype=float)
        self._cross_beta = float(sim['beta_layer'].get('cross', 1.0))
        # 确保 base 层有 beta 数组（与 p1 等长），使用 Covasim 的默认浮点类型
        if 'base' in sim.people.contacts:
            layer = sim.people.contacts['base']
            if 'beta' not in layer or len(layer['beta']) != len(layer['p1']):
                import covasim.defaults as cvd
                layer['beta'] = np.ones(len(layer['p1']), dtype=cvd.default_float)

    def apply(self, sim):
        t = sim.t
        people = sim.people
        position = getattr(people, self.region_key, None)
        country = getattr(people, 'country', None)
        crosser = getattr(people, 'crosser', None)
        if position is None or country is None or crosser is None:
            return
        return_day = self._return_day

        # 1) 到期者回国（排除被隔离人员：quarantined 或 isolated 状态不能移动）
        returning = crosser & (return_day == t) & ~people.quarantined & ~people.isolated
        if np.any(returning):
            position[returning] = country[returning]
            return_day[returning] = np.nan

        # 2) 从境内候鸟中按比例随机选人出境（仅从 start_day 开始；end_day_outbound 之后不再派出）
        if t >= self.start_day and (self.end_day_outbound is None or t < self.end_day_outbound):
            at_home = crosser & np.isnan(return_day) & ~people.quarantined & ~people.isolated
            n_at_home = np.count_nonzero(at_home)
            if n_at_home > 0 and self.frac_cross_per_day > 0:
                n_go = max(0, int(n_at_home * self.frac_cross_per_day + 0.5))
                n_go = min(n_go, n_at_home)
                if n_go > 0:
                    at_home_inds = np.where(at_home)[0]
                    go_inds = np.random.choice(at_home_inds, size=n_go, replace=False)
                    dur = np.random.randint(self.duration_min, self.duration_max + 1, size=len(go_inds))
                    return_day[go_inds] = t + dur
                    # 对方区域：A -> B, B -> A
                    from_a = (np.asarray(country[go_inds]) == self.region_name_a)
                    position[go_inds] = np.where(from_a, self.region_name_b, self.region_name_a)

        # 3) 按 position 重算 base/cross 层 per-edge beta
        import covasim.defaults as cvd
        is_abroad = (np.asarray(position) != np.asarray(country))
        if 'base' in people.contacts:
            layer = people.contacts['base']
            p1, p2 = layer['p1'], layer['p2']
            beta = layer['beta']
            edge_abroad = is_abroad[p1] | is_abroad[p2]
            beta[edge_abroad] = cvd.default_float(0.0)
            beta[~edge_abroad] = cvd.default_float(1.0)
        if 'cross' in people.contacts:
            layer = people.contacts['cross']
            p1, p2 = layer['p1'], layer['p2']
            beta = layer['beta']
            edge_abroad = is_abroad[p1] | is_abroad[p2]
            beta[edge_abroad] = cvd.default_float(self._cross_beta)
            beta[~edge_abroad] = cvd.default_float(0.0)


# 候鸟动态跨境干预实例：每日 10% 境内候鸟出境，境外 1–7 天，第 0 天开始
crosser_travel = CrosserTravel(frac_cross_per_day=0.1, duration_min=1, duration_max=7, start_day=0)
# 场景三：第三阶段起停止派出出境（end_day_outbound=34），仅保留到期回国
crosser_travel_case03 = CrosserTravel(
    frac_cross_per_day=0.1,
    duration_min=1,
    duration_max=7,
    start_day=0,
    end_day_outbound=_scenario_a_start_round3,
)
# 场景三：边境检测仅在阶段 1–2（round3 前一天结束）
test_isolate_crosser_case03 = cv.test_prob(
    symp_prob=0.8,
    asymp_prob=0.1,
    start_day=_scenario_a_start_round1,
    end_day=_scenario_a_start_round3 - 1,
    test_delay=1,
    subtarget=_subtarget_crosser,
)

# ========== 6. 口罩佩戴：通过 rel_trans 减少 30% 传播性 ==========
class MaskWearing(cv.Intervention):
    '''通过降低传染源的 rel_trans（相对传播力）表示戴口罩，传播性降为 efficacy（默认 0.7 即减少 30%）。
    fraction：目标人群中佩戴口罩的比例（0~1），默认 1.0 表示全部佩戴。'''
    def __init__(self, start_day=10, efficacy=0.7, fraction=1.0, subtarget=None, **kwargs):
        super().__init__(**kwargs)
        self.start_day = start_day
        self.efficacy = efficacy
        self.fraction = fraction
        self.subtarget = subtarget
        self._applied = False

    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)

    def apply(self, sim):
        if sim.t != self.start_day or self._applied:
            return
        if self.subtarget is not None and 'inds' in self.subtarget:
            inds = self.subtarget['inds'](sim)
        else:
            inds = np.arange(sim.n)
        if len(inds) == 0:
            return
        if self.fraction >= 1.0:
            wear_inds = inds
        else:
            n_wear = min(len(inds), int(len(inds) * self.fraction + 0.5))
            wear_inds = np.random.choice(inds, size=n_wear, replace=False) if n_wear > 0 else np.array([], dtype=int)
        if len(wear_inds) > 0:
            sim.people.rel_trans[wear_inds] *= self.efficacy
        self._applied = True

mask_wearing = MaskWearing(start_day=intervention_start, efficacy=0.7)

# A 区 50% 口罩佩戴（仅 position=='A' 中随机 50% 降低传播性，本情景自第 0 天开始）
mask_wearing_a_50 = MaskWearing(
    start_day=_scenario_a_start_round1,
    efficacy=0.5,
    fraction=0.5,
    subtarget={'inds': lambda sim: np.where(_is_position_a(sim))[0]},
)

# ========== 两阶段口罩佩戴：第一阶段 fraction_1 比例，第二阶段达到 fraction_2 比例 ==========
class MaskWearingTwoPhase(cv.Intervention):
    '''两阶段口罩佩戴：第一阶段 start_day_1 对 subtarget 的 fraction_1 比例生效，
    第二阶段 start_day_2 对 subtarget 中剩余的人（使总比例达到 fraction_2）生效。
    通过降低传染源的 rel_trans（相对传播力）表示戴口罩，传播性降为 efficacy。
    
    Args:
        start_day_1: 第一阶段开始日期
        start_day_2: 第二阶段开始日期
        efficacy: 口罩效果（0~1），默认 0.5 表示传播性降为原来的 50%
        fraction_1: 第一阶段目标人群中佩戴口罩的比例（0~1），默认 0.5
        fraction_2: 第二阶段目标人群中佩戴口罩的总比例（0~1），默认 1.0
        subtarget: 目标人群筛选条件，格式为 {'inds': lambda sim: ...}
    '''
    def __init__(self, start_day_1, start_day_2, efficacy=0.7, fraction_1=0.5, fraction_2=1.0, subtarget=None, **kwargs):
        super().__init__(**kwargs)
        self.start_day_1 = start_day_1
        self.start_day_2 = start_day_2
        self.efficacy = efficacy
        self.fraction_1 = fraction_1
        self.fraction_2 = fraction_2
        self.subtarget = subtarget
        self._wearing_inds = None  # 已在第一阶段戴口罩的人的索引集合

    def initialize(self, sim):
        super().initialize()
        self.start_day_1 = sim.day(self.start_day_1)
        self.start_day_2 = sim.day(self.start_day_2)
        self._wearing_inds = set()

    def apply(self, sim):
        if self.subtarget is not None and 'inds' in self.subtarget:
            inds = np.array(self.subtarget['inds'](sim), dtype=int)
        else:
            inds = np.arange(sim.n)
        if len(inds) == 0:
            return

        t = sim.t
        
        # 第一阶段：在 start_day_1 对 fraction_1 比例的人应用口罩
        if t == self.start_day_1:
            n1 = min(len(inds), int(len(inds) * self.fraction_1 + 0.5))
            if n1 > 0:
                wear_1 = np.random.choice(inds, size=n1, replace=False)
                if len(wear_1) > 0:
                    sim.people.rel_trans[wear_1] *= self.efficacy
                    self._wearing_inds = set(wear_1.tolist())
        
        # 第二阶段：在 start_day_2 对剩余的人（使总比例达到 fraction_2）应用口罩
        elif t == self.start_day_2:
            # 计算第二阶段需要达到的总人数
            n_total_target = min(len(inds), int(len(inds) * self.fraction_2 + 0.5))
            # 计算还需要新增的人数
            n_already_wearing = len(self._wearing_inds)
            n_to_add = max(0, n_total_target - n_already_wearing)
            
            if n_to_add > 0:
                # 找出尚未戴口罩的人
                remaining = inds[~np.isin(inds, list(self._wearing_inds))]
                if len(remaining) > 0:
                    # 从剩余的人中随机选择需要新增的人数
                    n_select = min(n_to_add, len(remaining))
                    wear_2 = np.random.choice(remaining, size=n_select, replace=False)
                    if len(wear_2) > 0:
                        sim.people.rel_trans[wear_2] *= self.efficacy
                        self._wearing_inds.update(wear_2.tolist())

# ================== 两阶段口罩佩戴：第一阶段 A 区 0.5，第二阶段 A 区 1.0 + B 区 0.5 =======================
# 第一阶段：仅 A 区 0.5 比例；第二阶段：A 区补足到 1.0，B 区 0.5 比例（两干预需同时加入情景）
mask_wearing_a_round1_2 = MaskWearingTwoPhase(
    start_day_1=_scenario_a_start_round1,
    start_day_2=_scenario_a_start_round2,
    efficacy=0.5,
    fraction_1=0.5,
    fraction_2=1.0,
    subtarget={'inds': lambda sim: np.where(_is_position_a(sim))[0]},
)
# 第二阶段起：B 区 0.5 比例佩戴（与 mask_wearing_a_round1_2 搭配实现「A 1.0、B 0.5」）
mask_wearing_b_phase2 = MaskWearing(
    start_day=_scenario_a_start_round2,
    efficacy=0.5,
    fraction=0.5,
    subtarget={'inds': lambda sim: np.where(_is_position_b(sim))[0]},
)

# ========== 情景：A 区 50% 口罩 + 第一批疫苗 10000 剂 + 边境检测（候鸟 50% 检测隔离，延迟1天），B 区无政策 ==========
intervention_scenario_a_policies = [
    # crosser_travel,
    # mask_wearing_a_50,
    # vaccinate_a_10k,
    test_isolate_a,
    contact_tracing_50,
    test_isolate_crosser, 
]
# ==================场景模拟01 只进行第一阶段干预===========================
intervention_scenario_case01 = [
    crosser_travel,
    mask_wearing_a_50,
    vaccinate_a_10k,
]
# ==================场景模拟02 第一和第二阶段干预（A 区 0.5→1.0，第二阶段 B 区 0.5）======================
intervention_scenario_case02 = [
    crosser_travel,
    mask_wearing_a_round1_2,
    mask_wearing_b_phase2,
    vaccinate_a_10k_round1_2,
    test_isolate_a,
    contact_tracing_50,
    test_isolate_crosser,
]
# ================== 场景模拟03：第一、二阶段同 case02，第三阶段（round3=34 起）加强 =======================
# 阶段 1–2：与 case02 相同（A 区口罩 0.5→1.0，B 区 0.5，A 区两批疫苗，境内检测/追踪，边境检测，跨境流动）
# 阶段 3（day 34 起）：A 区境内检测与接触者追踪概率提升一倍；A 区第三批疫苗 10000 剂；停止跨境派出+取消边境检测；B 区接种 5000 剂
intervention_scenario_case03 = [
    crosser_travel_case03,
    mask_wearing_a_round1_2,
    mask_wearing_b_phase2,
    vaccinate_a_10k_round1_2_3,
    vaccinate_b_5000,
    test_isolate_a_case03_phase12,
    test_isolate_a_case03_phase3,
    # test_for_ct_case03_phase12,
    # test_for_ct_case03_phase3,
    contact_tracing_50_case03_phase12,
    contact_tracing_50_case03_phase3,
    test_isolate_crosser_case03,
]
# ========== 按需将上述干预加入 interventions 列表，再传入 Sim ==========
# 示例：无干预 | 仅检测隔离 | 检测+接触者追踪 | 疫苗接种(二选一) | 境内减边 | 口罩佩戴 | 候鸟动态跨境(crosser_travel) | A区情景(50%口罩+1万剂疫苗+候鸟检测隔离) | 跨境限制需用上面 popdict 建无跨区层
# 若使用候鸟动态跨境，建议将 crosser_travel 放在列表前面，以便每日先更新 position 再执行其他干预
interventions = []  # 无干预
# interventions = [test_isolate]
# interventions = intervention_contact_tracing
# interventions = [vaccinate_a]  
# interventions = [vaccinate_a_300]           # 3b：A 区 300 剂，优先候鸟，多余随机给 A 区其他人
# interventions = [mask_wearing]
# interventions = [clip_base_50_region_a]
# interventions = [crosser_travel]
# ==================场景模拟===========================
interventions = intervention_scenario_case03  # A 区 50% 口罩 + 10000 剂疫苗 + 候鸟 50% 检测隔离(延迟1天)，B 区无政策

sim = cv.Sim(
    pars=custom_pars,
    label='组合模拟',
    interventions=interventions,
    analyzers=[MyPlot.CountryRegionAnalyzer(country_key='country', regions=('A', 'B'))],
)
sim.popdict = popdict
sim.reset_layer_pars(force=True)
sim.initialize()
sim.run()

# 保存模拟结果与图片到指定目录（传完整路径，避免 sc.makefilepath 拼接时中文名被截成只剩 .sim）
results_dir = r'myproject\results\双耦合网络图片\组合模拟'
os.makedirs(results_dir, exist_ok=True)
sim_basename = 'case03'
sim_path = os.path.join(results_dir, sim_basename + '.sim')
sim.save(filename=sim_path)

# 按 A/B 两区域分别绘制：左上/右上为 A 区 SEIR+病程，左下/右下为 B 区，并保存图片
MyPlot.plot_two_country_epidemic_curves(
    sim, country_key='country', regions=('A', 'B'),
    save_path=os.path.join(results_dir, sim_basename + '.png'),
    figsize=(12, 10),
    show_severity=False,
    curves = ['n_exposed', 'n_infectious','n_quarantined','n_isolated'],  # 新增参数：只画 I 和 R
    # curves = ['cum_infections'],
    show_regions=('A','B')
)

