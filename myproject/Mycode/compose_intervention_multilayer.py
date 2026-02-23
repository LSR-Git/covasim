"""
多层耦合网络政策干预组合脚本。

策略时间段与 compose_intervention.py 一致：
  round1=0, round2=17, round3=34, round4=42

场景一：多层级常规策略（round1 期间）
  - 口罩佩戴：工作层、学校层，仅 A 区，100% 依从性
  - 边境检测：合法入境（务工、探亲）在 A 区的候鸟，80% 有症状 / 10% 无症状检出率
  - 境内检测、接触者追踪、境内流动：无

场景二：先常规策略，round2 起升级策略至结束（类似 compose 场景02）
  - round1：同场景一
  - round2 起：境内检测 40%/1%、接触者追踪 40%、口罩扩展至社区/工作/家庭层、
    学校停课、社区/工作层各保留 50% 边、A 区疫苗 10000 剂、边境检测持续

场景三：在场景二基础上，round3 起严控策略至结束
  - round3 起：停止跨境派出、取消边境检测、境内检测提升至 0.4/0.1、
    接触者追踪 40%、工作层全面停工、社区层增强限制（约 20% 保留）、
    A 区第三批疫苗 10000 剂、B 区 5000 剂

场景四：常规 + 升级，round4 起温和策略至结束
  - round4 起：境内检测降为低强度（20%/0.5%）、停止口罩、恢复社区/工作/学校层边、
    保留接触者追踪、跨境与入境检测持续

场景五：常规 + 升级 + 严控，round4 起温和策略至结束
  - round3：同场景三（禁止出境、取消边境检测、严控境内）
  - round4 起：恢复跨境派出与入境检测、境内检测降为低强度、停止口罩、恢复各层边、保留接触者追踪

场景六：同场景五，但 round4 起完全停止境内检测（无低强度检测）
  - 用于研究温和策略中「停止检测 + 保留追踪」导致接触者追踪失效、疫情复发的对比情景
"""
import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import covasim as cv
import Enums
import ContactNetwork
import CrossNetwork
import MyPlot
from my_intervention import (
    CrosserTravelMultilayer,
    MaskWearingLayerSpecific,
    WorkFromHomeA,
    CommunityRestrictA,
    SchoolCloseA,
    ContactTracingAOnly,
)
from my_utils import (
    make_subtarget_crosser_exclude_undocumented,
    make_subtarget_position,
    sequence_random,
)

# 设置 matplotlib 显示中文
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ================== 1. 网络层配置 ==================
custom_config = {
    'community': {
        'network_type': Enums.NetWorkType.scale_free.name,
        'n_contacts': None,
        'm_connections': 3,
        'age_range': None,
        'cluster_size': None,
    },
    'work': {
        'network_type': Enums.NetWorkType.random.name,
        'n_contacts': 10,
        'age_range': (22, 65),
    },
    'school': {
        'network_type': Enums.NetWorkType.random.name,
        'n_contacts': 10,
        'age_range': (6, 22),
    },
    'home': {
        'network_type': Enums.NetWorkType.microstructured.name,
        'cluster_size': 3.0,
    },
}

# ================== 2. 人口与跨区层创建 ==================
countries_config = {'A': 2, 'B': 1}
seed_population = 0
seed_cross_layer = 42
pop_size = 30000

popdict_base, custom_keys = ContactNetwork.create_custom_population(
    pop_size, custom_config, countries_config, seed=seed_population
)
popdict_base = CrossNetwork.add_cross_layer_multilayer(
    popdict_base,
    frac_travelers=0.01,
    n_cross_per_person=10,
    cross_beta=0.6,
    frac_work=0.7,
    frac_visit=0.25,
    frac_undocumented=0.05,
    cross_layer_seed=seed_cross_layer,
    region_a='A',
    region_b='B',
)

# ================== 3. 仿真参数 ==================
custom_pars = {
    'pop_size': pop_size,
    'pop_infected': {'A': 20, 'B': 20},
    'pop_infected_region_key': 'country',
    'start_day': '2021-07-04',
    'n_days': 180,
    'rescale': False,
    'beta_layer': {
        'home': 3, 'school': 0.6, 'work': 0.6, 'community': 0.3,
        'cross_work': 0.6, 'cross_community': 0.6, 'cross_home': 0.6,
    },
    'beta': 0.036,
}

# ================== 4. 策略时间段（与 compose_intervention 一致） ==================
_scenario_a_start_round1 = 0
_scenario_a_start_round2 = 17
_scenario_a_start_round3 = 34
_scenario_a_start_round4 = 42

# ================== 5. 多层级常规策略干预 ==================
# 跨境移动
crosser_travel_ml = CrosserTravelMultilayer(
    frac_cross_per_day=0.1,
    duration_min=1,
    duration_max=7,
    start_day=_scenario_a_start_round1,
)
# 口罩：工作层、学校层，仅 A 区，100% 依从性，efficacy=0.5
mask_work_school_a = MaskWearingLayerSpecific(
    layers=['work', 'school'],
    efficacy=0.5,
    start_day=_scenario_a_start_round1,
)
# 边境检测：合法入境在 A 区的候鸟，80% 有症状 / 10% 无症状
test_crosser_legal = cv.test_prob(
    symp_prob=0.8,
    asymp_prob=0.1,
    test_delay=1,
    start_day=_scenario_a_start_round1,
    subtarget=make_subtarget_crosser_exclude_undocumented(crosser_prob=1.0),
)

# ================== 场景一：仅常规策略 ==================
intervention_scenario_case01 = [
    crosser_travel_ml,
    mask_work_school_a,
    test_crosser_legal,
]

# ================== 场景二：常规 + 升级（round2 起） ==================
# 口罩：round2 起扩展至社区、工作、家庭层（学校停课无需口罩）
mask_community_work_home_phase2 = MaskWearingLayerSpecific(
    layers=['community', 'work', 'home'],
    efficacy=0.5,
    start_day=_scenario_a_start_round2,
)
# 境内检测：round2 起 A 区 40% 有症状 / 1% 无症状（升级策略文档）
test_isolate_a_phase2 = cv.test_prob(
    symp_prob=0.4,
    asymp_prob=0.01,
    start_day=_scenario_a_start_round2,
    test_delay=2,
    subtarget=make_subtarget_position(),
)
# 接触者追踪：round2 起 A 区 40% 追踪
contact_tracing_phase2 = ContactTracingAOnly(
    trace_probs=0.4,
    trace_time=2,
    start_day=_scenario_a_start_round2,
)
# 学校停课：round2 起 A 区学校层全部移除
school_close_a_phase2 = SchoolCloseA(start_day=_scenario_a_start_round2)
# 社区层：round2 起保留 50% 边
community_restrict_a_phase2 = CommunityRestrictA(start_day=_scenario_a_start_round2, fraction=0.5)
# 工作层：round2 起保留 50% 边
work_from_home_a_phase2 = WorkFromHomeA(start_day=_scenario_a_start_round2, fraction=0.5)
# 疫苗：round2 起 A 区 10000 剂
vaccinate_a_10k_phase2 = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={_scenario_a_start_round2: 10000},
    sequence=sequence_random,
    subtarget=make_subtarget_position(),
)

# ================== 场景三：常规 + 升级 + 严控（round3 起） ==================
# 跨境：round3 起停止派出
crosser_travel_ml_case03 = CrosserTravelMultilayer(
    frac_cross_per_day=0.1,
    duration_min=1,
    duration_max=7,
    start_day=_scenario_a_start_round1,
    end_day_outbound=_scenario_a_start_round3,
)
# 边境检测：round3 前一天结束
test_crosser_legal_case03 = cv.test_prob(
    symp_prob=0.8,
    asymp_prob=0.1,
    test_delay=1,
    start_day=_scenario_a_start_round1,
    end_day=_scenario_a_start_round3 - 1,
    subtarget=make_subtarget_crosser_exclude_undocumented(crosser_prob=1.0),
)
# 境内检测：phase2 至 round3 前，phase3 严控阶段（概率与 phase2 一致）
test_isolate_a_phase2_case03 = cv.test_prob(
    symp_prob=0.4,
    asymp_prob=0.01,
    start_day=_scenario_a_start_round2,
    end_day=_scenario_a_start_round3 - 1,
    test_delay=2,
    subtarget=make_subtarget_position(),
)
test_isolate_a_phase3 = cv.test_prob(
    symp_prob=0.4,
    asymp_prob=0.01,
    start_day=_scenario_a_start_round3,
    test_delay=2,
    subtarget=make_subtarget_position(),
)
# 接触者追踪：phase2 至 round3 前，phase3 严控
contact_tracing_phase2_case03 = ContactTracingAOnly(
    trace_probs=0.4,
    trace_time=2,
    start_day=_scenario_a_start_round2,
    end_day=_scenario_a_start_round3 - 1,
)
contact_tracing_phase3 = ContactTracingAOnly(
    trace_probs=0.4,
    trace_time=2,
    start_day=_scenario_a_start_round3,
)
# 工作层：round3 起全面停工（移除剩余边）
work_from_home_a_phase3 = WorkFromHomeA(start_day=_scenario_a_start_round3, fraction=0)
# 社区层：round3 起增强限制（对剩余边再保留 40%，最终约 20%）
community_restrict_a_phase3 = CommunityRestrictA(start_day=_scenario_a_start_round3, fraction=0.4)
# 疫苗：round3 起 A 区第三批、B 区
vaccinate_a_10k_phase3 = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={_scenario_a_start_round3: 10000},
    sequence=sequence_random,
    subtarget=make_subtarget_position(),
)
vaccinate_b_5000 = cv.vaccinate_num(
    vaccine='pfizer',
    num_doses={_scenario_a_start_round3: 5000},
    sequence=sequence_random,
    subtarget=make_subtarget_position(region_name='B'),
)

intervention_scenario_case03 = [
    crosser_travel_ml_case03,
    mask_work_school_a,
    mask_community_work_home_phase2,
    test_crosser_legal_case03,
    test_isolate_a_phase2_case03,
    test_isolate_a_phase3,
    contact_tracing_phase2_case03,
    contact_tracing_phase3,
    school_close_a_phase2,
    community_restrict_a_phase2,
    community_restrict_a_phase3,
    work_from_home_a_phase2,
    work_from_home_a_phase3,
    vaccinate_a_10k_phase2,
    vaccinate_a_10k_phase3,
    vaccinate_b_5000,
]

intervention_scenario_case02 = [
    crosser_travel_ml,
    mask_work_school_a,
    mask_community_work_home_phase2,
    test_crosser_legal,
    test_isolate_a_phase2,
    contact_tracing_phase2,
    school_close_a_phase2,
    community_restrict_a_phase2,
    work_from_home_a_phase2,
    vaccinate_a_10k_phase2,
]

# ================== 场景四：常规 + 升级 → round4 起温和策略 ==================
# 口罩：round4 前结束
mask_work_school_a_case04 = MaskWearingLayerSpecific(
    layers=['work', 'school'],
    efficacy=0.5,
    start_day=_scenario_a_start_round1,
    end_day=_scenario_a_start_round4 - 1,
)
mask_community_work_home_case04 = MaskWearingLayerSpecific(
    layers=['community', 'work', 'home'],
    efficacy=0.5,
    start_day=_scenario_a_start_round2,
    end_day=_scenario_a_start_round4 - 1,
)
# 境内检测：round4 前结束
test_isolate_a_phase2_case04 = cv.test_prob(
    symp_prob=0.4,
    asymp_prob=0.01,
    start_day=_scenario_a_start_round2,
    end_day=_scenario_a_start_round4 - 1,
    test_delay=2,
    subtarget=make_subtarget_position(),
)
# 境内检测：round4 起低强度持续（温和策略保留，供接触者追踪触发）
test_isolate_a_phase4_case04 = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.005,
    start_day=_scenario_a_start_round4,
    test_delay=2,
    subtarget=make_subtarget_position(),
)
# 境内流动：round4 当日恢复边
school_close_a_phase2_case04 = SchoolCloseA(
    start_day=_scenario_a_start_round2,
    end_day=_scenario_a_start_round4,
)
community_restrict_a_phase2_case04 = CommunityRestrictA(
    start_day=_scenario_a_start_round2,
    fraction=0.5,
    end_day=_scenario_a_start_round4,
)
work_from_home_a_phase2_case04 = WorkFromHomeA(
    start_day=_scenario_a_start_round2,
    fraction=0.5,
    end_day=_scenario_a_start_round4,
)

intervention_scenario_case04 = [
    crosser_travel_ml,
    mask_work_school_a_case04,
    mask_community_work_home_case04,
    test_crosser_legal,
    test_isolate_a_phase2_case04,
    test_isolate_a_phase4_case04,
    contact_tracing_phase2,
    school_close_a_phase2_case04,
    community_restrict_a_phase2_case04,
    work_from_home_a_phase2_case04,
    vaccinate_a_10k_phase2,
]

# ================== 场景五：常规 + 升级 + 严控 → round4 起温和策略 ==================
# 跨境：round3 停止派出，round4 恢复
crosser_travel_ml_case05 = CrosserTravelMultilayer(
    frac_cross_per_day=0.1,
    duration_min=1,
    duration_max=7,
    start_day=_scenario_a_start_round1,
    end_day_outbound=_scenario_a_start_round3,
    resume_day_outbound=_scenario_a_start_round4,
)
# 边境检测：round3 取消，round4 恢复
test_crosser_legal_phase1_case05 = cv.test_prob(
    symp_prob=0.8,
    asymp_prob=0.1,
    test_delay=1,
    start_day=_scenario_a_start_round1,
    end_day=_scenario_a_start_round3 - 1,
    subtarget=make_subtarget_crosser_exclude_undocumented(crosser_prob=1.0),
)
test_crosser_legal_phase2_case05 = cv.test_prob(
    symp_prob=0.8,
    asymp_prob=0.1,
    test_delay=1,
    start_day=_scenario_a_start_round4,
    subtarget=make_subtarget_crosser_exclude_undocumented(crosser_prob=1.0),
)
# 口罩：round4 前结束
mask_work_school_a_case05 = MaskWearingLayerSpecific(
    layers=['work', 'school'],
    efficacy=0.5,
    start_day=_scenario_a_start_round1,
    end_day=_scenario_a_start_round4 - 1,
)
mask_community_work_home_case05 = MaskWearingLayerSpecific(
    layers=['community', 'work', 'home'],
    efficacy=0.5,
    start_day=_scenario_a_start_round2,
    end_day=_scenario_a_start_round4 - 1,
)
# 境内检测：round2 至 round3 前、round3 至 round4 前，round4 起低强度持续
test_isolate_a_phase2_case05 = cv.test_prob(
    symp_prob=0.4,
    asymp_prob=0.01,
    start_day=_scenario_a_start_round2,
    end_day=_scenario_a_start_round3 - 1,
    test_delay=2,
    subtarget=make_subtarget_position(),
)
test_isolate_a_phase3_case05 = cv.test_prob(
    symp_prob=0.4,
    asymp_prob=0.01,
    start_day=_scenario_a_start_round3,
    end_day=_scenario_a_start_round4 - 1,
    test_delay=2,
    subtarget=make_subtarget_position(),
)
# 境内检测：round4 起低强度持续（温和策略保留，供接触者追踪触发）
test_isolate_a_phase4_case05 = cv.test_prob(
    symp_prob=0.2,
    asymp_prob=0.005,
    start_day=_scenario_a_start_round4,
    test_delay=2,
    subtarget=make_subtarget_position(),
)
# 接触者追踪：round2 起持续（温和策略保留）
contact_tracing_phase2_case05 = ContactTracingAOnly(
    trace_probs=0.4,
    trace_time=2,
    start_day=_scenario_a_start_round2,
)
# 境内流动：round4 当日恢复边
school_close_a_phase2_case05 = SchoolCloseA(
    start_day=_scenario_a_start_round2,
    end_day=_scenario_a_start_round4,
)
community_restrict_a_phase2_case05 = CommunityRestrictA(
    start_day=_scenario_a_start_round2,
    fraction=0.5,
    end_day=_scenario_a_start_round4,
)
community_restrict_a_phase3_case05 = CommunityRestrictA(
    start_day=_scenario_a_start_round3,
    fraction=0.4,
    end_day=_scenario_a_start_round4,
)
work_from_home_a_phase2_case05 = WorkFromHomeA(
    start_day=_scenario_a_start_round2,
    fraction=0.5,
    end_day=_scenario_a_start_round4,
)
work_from_home_a_phase3_case05 = WorkFromHomeA(
    start_day=_scenario_a_start_round3,
    fraction=0,
    end_day=_scenario_a_start_round4,
)

intervention_scenario_case05 = [
    crosser_travel_ml_case05,
    mask_work_school_a_case05,
    mask_community_work_home_case05,
    test_crosser_legal_phase1_case05,
    test_crosser_legal_phase2_case05,
    test_isolate_a_phase2_case05,
    test_isolate_a_phase3_case05,
    test_isolate_a_phase4_case05,
    contact_tracing_phase2_case05,
    school_close_a_phase2_case05,
    community_restrict_a_phase2_case05,
    community_restrict_a_phase3_case05,
    work_from_home_a_phase2_case05,
    work_from_home_a_phase3_case05,
    vaccinate_a_10k_phase2,
    vaccinate_a_10k_phase3,
    vaccinate_b_5000,
]

# ================== 场景六：常规 + 升级 + 严控 → round4 温和（无低强度境内检测） ==================
# 与场景五相同，但 round4 起完全停止境内检测，接触者追踪无新确诊可追踪而失效，用于研究疫情复发
intervention_scenario_case06 = [
    crosser_travel_ml_case05,
    mask_work_school_a_case05,
    mask_community_work_home_case05,
    test_crosser_legal_phase1_case05,
    test_crosser_legal_phase2_case05,
    test_isolate_a_phase2_case05,
    test_isolate_a_phase3_case05,
    # 无 test_isolate_a_phase4，round4 起境内检测完全停止
    contact_tracing_phase2_case05,
    school_close_a_phase2_case05,
    community_restrict_a_phase2_case05,
    community_restrict_a_phase3_case05,
    work_from_home_a_phase2_case05,
    work_from_home_a_phase3_case05,
    vaccinate_a_10k_phase2,
    vaccinate_a_10k_phase3,
    vaccinate_b_5000,
]

# ================== 场景切换 ==================
interventions = []  # 无干预
interventions = intervention_scenario_case01  # 场景一：仅常规策略
# interventions = intervention_scenario_case02  # 场景二：常规 + 升级
# interventions = intervention_scenario_case03  # 场景三：常规 + 升级 + 严控
# interventions = intervention_scenario_case04  # 场景四：常规 + 升级 → round4 温和
# interventions = intervention_scenario_case05  # 场景五：常规 + 升级 + 严控 → round4 温和（含低强度检测）
# interventions = intervention_scenario_case06  # 场景六：常规 + 升级 + 严控 → round4 温和（无境内检测，研究复发）

# ================== 6. 运行模拟 ==================
sim = cv.Sim(
    pars=custom_pars,
    label='多层策略组合',
    interventions=interventions,
    analyzers=[MyPlot.CountryRegionAnalyzer(country_key='country', regions=('A', 'B'))],
)
sim.popdict = popdict_base
sim.reset_layer_pars(force=True)
sim.initialize()
sim.run()

# ================== 7. 保存与绘图 ==================
_scenario_name = (
    'case06' if interventions == intervention_scenario_case06
    else 'case05' if interventions == intervention_scenario_case05
    else 'case04' if interventions == intervention_scenario_case04
    else 'case03' if interventions == intervention_scenario_case03
    else 'case02' if interventions == intervention_scenario_case02
    else 'case01' if interventions == intervention_scenario_case01
    else 'baseline'
)
results_dir = os.path.join('myproject', 'results', '多层耦合网络图片', _scenario_name)
os.makedirs(results_dir, exist_ok=True)
sim_basename = _scenario_name
sim_path = os.path.join(results_dir, sim_basename + '.sim')
sim.save(filename=sim_path, keep_people=True)

MyPlot.plot_layer_region_infections(
    sim,
    country_key='country',
    regions=('A', 'B'),
    layers=['home', 'school', 'work', 'community'],
    show_regions=('A','B'),
    save_path=os.path.join(results_dir, sim_basename + '.png'),
)

print('多层策略模拟完成，结果已保存至', results_dir)
