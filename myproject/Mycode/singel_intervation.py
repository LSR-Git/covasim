import numpy as np
import covasim as cv
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
    'pop_infected': {'A':20, 'B': 20},
    'pop_infected_region_key': 'country',

    # Simulation parameters
    'start_day': '2021-07-04',
    'n_days': 60,

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
intervention_start = 10

# 统一：所有干预的“A 区”均按当前所在地 position=='A' 判定（与 country 户籍区分，便于跨境场景）
_region_key = 'position'
_region_name_a = 'A'

def _is_position_a(sim):
    return (np.asarray(getattr(sim.people, _region_key)) == _region_name_a)

# 仅 A 区有资格的 subtarget（检测/追踪/疫苗接种等共用）
_subtarget_position_a = {
    'inds': lambda sim: np.arange(sim.n),
    'vals': lambda sim: _is_position_a(sim).astype(float),
}

# ========== 1. 检测隔离：仅对 A 区（position=='A'）50% 检测隔离，检测延迟 2 天 ==========
test_isolate = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.05,
    start_day=intervention_start,
    test_delay=2,
    subtarget=_subtarget_position_a,
)

# ========== 2. 接触者追踪：仅对 A 区检测 + 50% 接触者追踪，追踪延迟 2 天 ==========
# 仅 position=='A' 者被检测，确诊者（均为 A 区）的接触者被追踪并隔离
test_for_ct = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.05,
    start_day=intervention_start,
    test_delay=2,
    subtarget=_subtarget_position_a,
)
contact_tracing_50 = cv.contact_tracing(
    trace_probs=0.2,
    trace_time=2,
    start_day=intervention_start,
)
intervention_contact_tracing = [test_for_ct, contact_tracing_50]

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

vaccinate_a = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={intervention_start: 8000},
    sequence=_sequence_random,
    subtarget=_subtarget_position_a,
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
clip_base_50_region_a = reduce_region_a_contacts_50(start_day=intervention_start)

# ========== 5. 跨境流动限制：将候鸟比例清零 ==========
# 方式一：建 sim 时直接使用无跨区层的人口（不调用 add_cross_layer 或 frac_travelers=0）
# popdict_no_cross = CrossNetwork.add_cross_layer(popdict_base, frac_travelers=0, ...)  # 无跨区边
# 方式二：若已有跨区层，可在干预日移除 cross 层（需自定义干预）。这里仅提供方式一，按需替换上面的 popdict。
# 使用无跨境时，将上面 popdict 改为：
# popdict = CrossNetwork.add_cross_layer(popdict_base, frac_travelers=0, n_cross_per_person=10, cross_beta=0.6, cross_layer_seed=seed_cross_layer)

# ========== 按需将上述干预加入 interventions 列表，再传入 Sim ==========
# 示例：无干预 | 仅检测隔离 | 检测+接触者追踪 | 疫苗接种(二选一) | 境内减边 | 跨境限制需用上面 popdict 建无跨区层
interventions = []  # 无干预
# interventions = [test_isolate]
# interventions = intervention_contact_tracing
# interventions = [vaccinate_a]  #
# interventions = [vaccinate_a_300]           # 3b：A 区 300 剂，优先候鸟，多余随机给 A 区其他人
interventions = [clip_base_50_region_a]
# interventions = [test_isolate_50, vaccinate_a_300]

sim = cv.Sim(
    pars=custom_pars,
    label='无干预',
    interventions=interventions,
    analyzers=[MyPlot.CountryRegionAnalyzer(country_key='country', regions=('A', 'B'))],
)
sim.popdict = popdict
sim.reset_layer_pars(force=True)
sim.initialize()
sim.run()

# 保存模拟结果与图片到指定目录（传完整路径，避免 sc.makefilepath 拼接时中文名被截成只剩 .sim）
results_dir = r'E:\大论文相关\covasim\myproject\results\双耦合网络图片\单个干预模拟\境内流动限制'
os.makedirs(results_dir, exist_ok=True)
sim_path = os.path.join(results_dir, 'reduce_region_a_contacts_50.sim')
sim.save(filename=sim_path)

# 按 A/B 两区域分别绘制：左上/右上为 A 区 SEIR+病程，左下/右下为 B 区，并保存图片
MyPlot.plot_two_country_epidemic_curves(
    sim, country_key='country', regions=('A', 'B'),
    save_path=os.path.join(results_dir, 'reduce_region_a_contacts_50.png'),
    figsize=(12, 10),
    show_severity=False,
    show_regions=('A','B')
)

