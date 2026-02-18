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
from my_intervention import (
    ContactTracingAOnly,
    reduce_region_a_contacts,
    ScaleRegionBaseBetaByPhase,
    CrosserTravel,
    MaskWearing,
    MaskWearingTwoPhase,
    MaskRelax,
    InjectUndocumentedInfectious,
)
from my_utils import (
    create_vaccination_schedule,
    sequence_random,
    sequence_crosser_first_then_random_a,
    is_position_a,
    is_position_b,
    make_subtarget_position,
    make_subtarget_crosser,
    make_subtarget_position_exclude_undocumented,
    make_subtarget_crosser_exclude_undocumented,
)

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

# 场景五：偷渡者注入（可配置）
CASE05_INJECT_DAY = 80
CASE05_N_UNDOCUMENTED = 30

# 统一：所有干预的“A/B 区”均按当前所在地 position=='A'/'B' 判定（与 country 户籍区分，便于跨境场景）
_region_key = 'position'
_region_name_a = 'A'
_region_name_b = 'B'

# 区域筛选与边境检测 subtarget（由 my_utils 构造）
_subtarget_position_a = make_subtarget_position(_region_key, _region_name_a)
_subtarget_crosser = make_subtarget_crosser(0.5, _region_key, _region_name_a)
_subtarget_position_b = make_subtarget_position(_region_key, _region_name_b)


# ==================================场景一干预策略===================================
# 场景一使用：跨境流动（每天派出 10% 的境内候鸟）
crosser_travel = CrosserTravel(frac_cross_per_day=0.1, duration_min=1, duration_max=7, start_day=0)
# 场景一使用：口罩佩戴（仅 position=='A' 中随机 50% 降低传播性，本情景自第 0 天开始）
mask_wearing_a_50 = MaskWearing(
    start_day=_scenario_a_start_round1,
    efficacy=0.5,
    fraction=0.5,
    subtarget={'inds': lambda sim: np.where(is_position_a(sim))[0]},
)
# 场景一使用：A 区第一批疫苗
vaccinate_a_10k = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={_scenario_a_start_round1: 10000},
    sequence=sequence_random,
    subtarget=_subtarget_position_a,
)

# ==================================场景二干预策略===================================
# 同上 ：crosser_travel = CrosserTravel(frac_cross_per_day=0.1, duration_min=1, duration_max=7, start_day=0)
# 场景二使用： A 区第一阶段 0.5 比例；第二阶段：A 区补足到 1.0，B 区 0.5 比例（两干预需同时加入情景）
mask_wearing_a_round1_2 = MaskWearingTwoPhase(
    start_day_1=_scenario_a_start_round1,
    start_day_2=_scenario_a_start_round2,
    efficacy=0.5,
    fraction_1=0.5,
    fraction_2=1.0,
    subtarget={'inds': lambda sim: np.where(is_position_a(sim))[0]},
)
# 场景二使用：B 区第二阶段 0.5 比例佩戴（与 mask_wearing_a_round1_2 搭配实现「A 1.0、B 0.5」）
mask_wearing_b_phase2 = MaskWearing(
    start_day=_scenario_a_start_round2,
    efficacy=0.5,
    fraction=0.5,
    subtarget={'inds': lambda sim: np.where(is_position_b(sim))[0]},
)
# 场景二使用：A 区第一批疫苗
vaccinate_a_10k_round1_2 = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={_scenario_a_start_round2: 10000},
    sequence=sequence_random,
    subtarget=_subtarget_position_a,
)
# 场景二使用：境内检测 对 A 区所在人员（position=='A'）的日常国内检测，20% 有症状、5% 无症状，延迟 2 天
test_isolate_a_case02_phase2 = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.05,
    start_day=intervention_start,
    test_delay=2,
    subtarget=_subtarget_position_a,
)
# 场景二使用：边境检测对所有来到 A 区的候鸟（position=='A' 且 crosser=True，包括 A 区候鸟和 B 区候鸟）的例行检测，50% 概率，延迟 1 天（本情景自第 0 天开始）
test_isolate_crosser = cv.test_prob(
    symp_prob=0.8,
    asymp_prob=0.1,
    start_day=_scenario_a_start_round1,
    test_delay=1,
    subtarget=_subtarget_crosser,
)
# 场景二使用：接触者追踪 仅对 A 区检测 + 50% 接触者追踪，追踪延迟 2 天
contact_tracing_case02_phase2 = ContactTracingAOnly(
    trace_probs=0.2,
    trace_time=2,
    start_day=_scenario_a_start_round2,
    region_key=_region_key,
    region_name=_region_name_a,
)

# =====================================场景三干预策略============================================
# ========== 2. 接触者追踪：仅对 A 区检测 + 50% 接触者追踪，追踪延迟 2 天 ==========
# 仅 position=='A' 者被检测，确诊者（均为 A 区）的接触者被追踪并隔离
# 接触者追踪类已迁移至 my_intervention.ContactTracingAOnly
test_for_ct = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.05,
    start_day=_scenario_a_start_round2,
    test_delay=2,
    subtarget=_subtarget_position_a,
)
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
# 3a. A 区随机接种（sequence_random）；3b. 优先候鸟再 A 区其他人（sequence_crosser_first_then_random_a）见 my_utils
vaccinate_a = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses=create_vaccination_schedule(total_doses=10000, daily_doses=10000, start_day=0),  # 第0天一次性接种10000剂
    sequence=sequence_random,
    subtarget=_subtarget_position_a,
)
# 场景三使用：A 区二批疫苗
vaccinate_a_10k_round1_2_3 = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={
        _scenario_a_start_round2: 10000,
        _scenario_a_start_round3: 10000,
    },
    sequence=sequence_random,
    subtarget=_subtarget_position_a,
)
# B 区疫苗接种（仅场景三：第三阶段起 5000 剂）；_subtarget_position_b 已在上方由 make_subtarget_position 构造
vaccinate_b_5000 = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={_scenario_a_start_round3: 5000},
    sequence=sequence_random,
    subtarget=_subtarget_position_b,
)


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

# ================== 场景四专用：阶段 3 境内检测在 day 42 前结束，阶段 4 政策放松 =======================
# 阶段 3 境内检测（仅 day 34–41，阶段 4 起不做境内检测）
test_isolate_a_case04_phase3 = cv.test_prob(
    symp_prob=0.4,
    asymp_prob=0.1,
    start_day=_scenario_a_start_round3,
    end_day=_scenario_a_start_round4 - 1,
    test_delay=2,
    subtarget=_subtarget_position_a,
)
# 阶段 3 接触者追踪（仅 day 34–41，阶段 4 起恢复为部分追踪，需在 day 42 前主动结束）
contact_tracing_case04_phase3 = ContactTracingAOnly(
    trace_probs=0.4,
    trace_time=2,
    start_day=_scenario_a_start_round3,
    end_day=_scenario_a_start_round4 - 1,
    region_key=_region_key,
    region_name=_region_name_a,
)
crosser_travel_case04_resume = CrosserTravel(
    frac_cross_per_day=0.1,
    duration_min=1,
    duration_max=7,
    start_day=_scenario_a_start_round4,
    end_day_outbound=None,
)
# 阶段 4 不做境内检测，仅恢复边境检测、接触者追踪与口罩放松
test_isolate_crosser_case04_phase4 = cv.test_prob(
    symp_prob=0.8,
    asymp_prob=0.1,
    start_day=_scenario_a_start_round4,
    test_delay=1,
    subtarget=_subtarget_crosser,
)
contact_tracing_case04_phase4 = ContactTracingAOnly(
    trace_probs=0.2,
    trace_time=2,
    start_day=_scenario_a_start_round4,
    region_key=_region_key,
    region_name=_region_name_a,
)
mask_relax_a_case04 = MaskRelax(
    start_day=_scenario_a_start_round4,
    efficacy=0.5,
    fraction=1.0,
    subtarget={'inds': lambda sim: np.where(is_position_a(sim))[0]},
)
# 境内流动：阶段1 无限制(1.0)、阶段2 部分限制(0.5)、阶段3 增强限制(0.3)、阶段4 无限制(1.0)；放在 CrosserTravel 之后以便每日覆盖 A 区境内边 beta
domestic_mobility_case04 = ScaleRegionBaseBetaByPhase(
    region_key=_region_key,
    region_name=_region_name_a,
    day_scale_pairs=[
        (0, 1.0),
        (_scenario_a_start_round2, 0.5),
        (_scenario_a_start_round3, 0.3),
        (_scenario_a_start_round4, 1.0),
    ],
)

# ==================场景模拟01 只进行第一阶段干预（无疫苗，有入境检测）===========================
intervention_scenario_case01 = [
    crosser_travel,
    mask_wearing_a_50,
    test_isolate_crosser,
]
# ==================场景模拟02 第一和第二阶段干预（A 区 0.5→1.0，第二阶段 B 区 0.5）======================
intervention_scenario_case02 = [
    crosser_travel,
    mask_wearing_a_round1_2,
    mask_wearing_b_phase2,
    vaccinate_a_10k_round1_2,
    test_isolate_crosser,
    test_isolate_a_case02_phase2,
    contact_tracing_case02_phase2,
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
    test_isolate_crosser_case03,
    contact_tracing_50_case03_phase12,
    contact_tracing_50_case03_phase3,
]
# ==================场景模拟04 四阶段全流程（与四阶段策略图对应）======================
# 阶段1 常规(0–16)：境内检测 0.2/0.05、不追踪→阶段2起追踪；口罩 A 0.5；疫苗第1批；跨境 入境核检
# 阶段2 升级(17–33)：境内检测同上；接触者 部分追踪0.2；口罩 A 1.0 + B 0.5；疫苗第2批；跨境 入境核检
# 阶段3 严控(34–41)：境内检测 高频 0.4/0.1（day41 结束）；接触者 高频 0.4（day41 结束）；口罩 仍 A 1.0；疫苗第3批+B区；跨境 禁止出境；阶段4 起 境内检测/高频追踪 均不再执行
# 阶段4 温和(42+)：境内检测 无；接触者 部分追踪 0.2；口罩 无限制；跨境 恢复派出+入境核检
# 说明：阶段3 口罩保持 1.0（按你的要求）；境内流动已实现 部分/增强/无限制 三档
intervention_scenario_case04 = [
    crosser_travel_case03,
    crosser_travel_case04_resume,
    domestic_mobility_case04,
    mask_wearing_a_round1_2,
    mask_wearing_b_phase2,
    vaccinate_a_10k_round1_2_3,
    vaccinate_b_5000,
    test_isolate_a_case03_phase12,
    test_isolate_a_case04_phase3,
    test_isolate_crosser_case03,
    test_isolate_crosser_case04_phase4,
    contact_tracing_50_case03_phase12,
    contact_tracing_case04_phase3,
    contact_tracing_case04_phase4,
    mask_relax_a_case04,
]

# ================== 场景五：四阶段全流程 + 可配置日注入 n 个偷渡者（不可检测/隔离） =======================
_subtarget_position_a_case05 = make_subtarget_position_exclude_undocumented(_region_key, _region_name_a)
_subtarget_crosser_case05 = make_subtarget_crosser_exclude_undocumented(0.5, _region_key, _region_name_a)
test_isolate_a_case03_phase12_case05 = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.05,
    start_day=intervention_start,
    end_day=_scenario_a_start_round3 - 1,
    test_delay=2,
    subtarget=_subtarget_position_a_case05,
)
test_isolate_a_case04_phase3_case05 = cv.test_prob(
    symp_prob=0.4,
    asymp_prob=0.1,
    start_day=_scenario_a_start_round3,
    end_day=_scenario_a_start_round4 - 1,
    test_delay=2,
    subtarget=_subtarget_position_a_case05,
)
test_isolate_crosser_case03_case05 = cv.test_prob(
    symp_prob=0.8,
    asymp_prob=0.1,
    start_day=_scenario_a_start_round1,
    end_day=_scenario_a_start_round3 - 1,
    test_delay=1,
    subtarget=_subtarget_crosser_case05,
)
test_isolate_crosser_case04_phase4_case05 = cv.test_prob(
    symp_prob=0.8,
    asymp_prob=0.1,
    start_day=_scenario_a_start_round4,
    test_delay=1,
    subtarget=_subtarget_crosser_case05,
)
inject_undocumented_case05 = InjectUndocumentedInfectious(
    inject_day=CASE05_INJECT_DAY,
    n=CASE05_N_UNDOCUMENTED,
    region_key=_region_key,
    region_name_a=_region_name_a,
)
intervention_scenario_case05 = [
    crosser_travel_case03,
    crosser_travel_case04_resume,
    inject_undocumented_case05,
    domestic_mobility_case04,
    mask_wearing_a_round1_2,
    mask_wearing_b_phase2,
    vaccinate_a_10k_round1_2_3,
    vaccinate_b_5000,
    test_isolate_a_case03_phase12_case05,
    test_isolate_a_case04_phase3_case05,
    test_isolate_crosser_case03_case05,
    test_isolate_crosser_case04_phase4_case05,
    contact_tracing_50_case03_phase12,
    contact_tracing_case04_phase3,
    contact_tracing_case04_phase4,
    mask_relax_a_case04,
]

# ================== 场景六：在场景五基础上，第 85 天起开启境内检测、A 区口罩全员、境内流动 0.5 =======================
CASE06_DAY85 = 85
domestic_mobility_case06 = ScaleRegionBaseBetaByPhase(
    region_key=_region_key,
    region_name=_region_name_a,
    day_scale_pairs=[
        (0, 1.0),
        (_scenario_a_start_round2, 0.5),
        (_scenario_a_start_round3, 0.3),
        (_scenario_a_start_round4, 1.0),
        (CASE06_DAY85, 0.5),
    ],
)
test_isolate_a_case06_day85 = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.05,
    start_day=CASE06_DAY85,
    test_delay=2,
    subtarget=_subtarget_position_a_case05,
)
mask_wearing_a_case06_day85 = MaskWearing(
    start_day=CASE06_DAY85,
    efficacy=0.5,
    fraction=1.0,
    subtarget={'inds': lambda sim: np.where(is_position_a(sim))[0]},
)
intervention_scenario_case06 = [
    crosser_travel_case03,
    crosser_travel_case04_resume,
    inject_undocumented_case05,
    domestic_mobility_case06,
    mask_wearing_a_round1_2,
    mask_wearing_b_phase2,
    vaccinate_a_10k_round1_2_3,
    vaccinate_b_5000,
    test_isolate_a_case03_phase12_case05,
    test_isolate_a_case04_phase3_case05,
    test_isolate_crosser_case03_case05,
    test_isolate_crosser_case04_phase4_case05,
    test_isolate_a_case06_day85,
    contact_tracing_50_case03_phase12,
    contact_tracing_case04_phase3,
    contact_tracing_case04_phase4,
    mask_relax_a_case04,
    mask_wearing_a_case06_day85,
]

interventions = []  # 无干预
# ==================场景模拟===========================
interventions = intervention_scenario_case06  # A 区 50% 口罩 + 10000 剂疫苗 + 候鸟 50% 检测隔离(延迟1天)，B 区无政策

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
sim_basename = 'case06_test'
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
    show_regions=('A')
)

