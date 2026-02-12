"""
口罩干预措施示例
演示两种方法：1) 使用change_beta（简单） 2) 自定义mask干预类（灵活）
"""

import numpy as np
import covasim as cv


# ============================================================================
# 方法1：使用 change_beta（最简单，推荐）
# ============================================================================
def method1_change_beta():
    """
    使用 change_beta 干预措施模拟戴口罩
    文档说明：change_beta 可以用于 masks（口罩）、hand-washing（洗手）等行为改变
    
    参数说明：
    - days: 开始戴口罩的日期（可以是日期字符串或天数）
    - changes: beta的倍数（1.0=不变，0.7=降低30%，0.5=降低50%）
    - layers: 哪些接触层受影响（None=所有层，'h'=家庭，'s'=学校，'w'=工作，'c'=社区）
    """
    
    pars = dict(
        pop_size=1000,
        pop_infected=20,
        start_day='2021-07-04',
        n_days=60,
    )
    
    # 基线模拟（不戴口罩）
    baseline = cv.Sim(pars, label='Baseline (no mask)')
    
    # 方法1a: 从第10天开始，所有层降低30%传染率（相当于戴口罩效果）
    mask_intervention = cv.change_beta(
        days=10,           # 第10天开始
        changes=0.7,        # beta降低到70%（即降低30%）
        layers=None,       # 所有接触层
        label='Mask wearing (30% reduction)'
    )
    
    sim1 = cv.Sim(pars, interventions=mask_intervention, label='With mask (30% reduction)')
    
    # 方法1b: 分阶段戴口罩（逐步加强）
    # 第10天：降低20%，第20天：降低40%，第30天：降低50%
    progressive_mask = cv.change_beta(
        days=[10, 20, 30],
        changes=[0.8, 0.6, 0.5],  # 逐步加强
        label='Progressive mask wearing'
    )
    
    sim2 = cv.Sim(pars, interventions=progressive_mask, label='Progressive mask')
    
    # 方法1c: 只在特定层戴口罩（例如只在社区层）
    community_mask = cv.change_beta(
        days=10,
        changes=0.5,        # 社区层降低50%
        layers='c',        # 只影响社区层
        label='Mask in community only'
    )
    
    sim3 = cv.Sim(pars, interventions=community_mask, label='Mask in community')
    
    # 运行并对比
    msim = cv.parallel([baseline, sim1, sim2, sim3])
    msim.plot()
    
    return msim


# ============================================================================
# 方法2：自定义口罩干预类（更灵活，可以设置覆盖率、依从性等）
# ============================================================================
class mask_wearing(cv.Intervention):
    """
    自定义口罩干预类
    
    特点：
    - 可以设置口罩覆盖率（多少人戴口罩）
    - 可以设置口罩效果（降低多少传染率）
    - 可以设置依从性（随时间变化）
    - 可以针对特定人群（如只给易感者或感染者）
    
    参数：
    - start_day: 开始日期
    - end_day: 结束日期（可选）
    - coverage: 覆盖率（0-1），默认1.0（100%）
    - efficacy: 口罩效果（0-1），0.3表示降低30%传染率
    - target: 'all'（所有人）、'susceptible'（易感者）、'infectious'（感染者）
    """
    
    def __init__(self, start_day=None, end_day=None, coverage=1.0, efficacy=0.3, 
                 target='all', label='Mask wearing', **kwargs):
        super().__init__(**kwargs)
        self.start_day = start_day
        self.end_day = end_day
        self.coverage = coverage  # 覆盖率：0-1
        self.efficacy = efficacy   # 效果：降低的传染率比例（0.3 = 降低30%）
        self.target = target       # 目标人群
        self.label = label
        self.mask_wearers = None   # 存储哪些人戴口罩
        return
    
    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day) if self.start_day else 0
        self.end_day = sim.day(self.end_day) if self.end_day else sim.npts
        
        # 随机选择戴口罩的人（根据覆盖率）
        n_people = len(sim.people)
        n_mask_wearers = int(n_people * self.coverage)
        mask_indices = np.random.choice(n_people, size=n_mask_wearers, replace=False)
        self.mask_wearers = np.zeros(n_people, dtype=bool)
        self.mask_wearers[mask_indices] = True
        
        # 根据目标人群调整
        if self.target == 'susceptible':
            # 只给易感者戴口罩
            self.mask_wearers = self.mask_wearers & sim.people.susceptible
        elif self.target == 'infectious':
            # 只给感染者戴口罩
            self.mask_wearers = self.mask_wearers & sim.people.infectious
        
        print(f'Mask intervention initialized: {self.mask_wearers.sum()}/{n_people} people will wear masks')
        return
    
    def apply(self, sim):
        """
        应用干预：降低戴口罩人群的传染性
        通过修改 rel_trans（感染者）和 rel_sus（易感者）来实现
        """
        # 检查是否在干预时间范围内
        if self.start_day <= sim.t < self.end_day:
            # 对于感染者：降低传染性（rel_trans）
            infectious_mask_wearers = self.mask_wearers & sim.people.infectious
            if infectious_mask_wearers.any():
                # 降低传染性：原来的 rel_trans * (1 - efficacy)
                sim.people.rel_trans[infectious_mask_wearers] *= (1 - self.efficacy)
            
            # 对于易感者：降低易感性（rel_sus）
            susceptible_mask_wearers = self.mask_wearers & sim.people.susceptible
            if susceptible_mask_wearers.any():
                # 降低易感性：原来的 rel_sus * (1 - efficacy)
                sim.people.rel_sus[susceptible_mask_wearers] *= (1 - self.efficacy)
        
        # 干预结束后恢复
        elif sim.t >= self.end_day:
            # 恢复原始值（假设原始值为1.0）
            sim.people.rel_trans[self.mask_wearers] = 1.0
            sim.people.rel_sus[self.mask_wearers] = 1.0
        
        return


def method2_custom_mask():
    """
    使用自定义口罩干预类
    """
    
    pars = dict(
        pop_size=1000,
        pop_infected=20,
        start_day='2021-07-04',
        n_days=60,
    )
    
    # 基线模拟
    baseline = cv.Sim(pars, label='Baseline (no mask)')
    
    # 自定义口罩干预：100%覆盖率，降低30%传染率
    mask1 = mask_wearing(
        start_day=10,
        coverage=1.0,      # 100%覆盖率
        efficacy=0.3,     # 降低30%传染率
        target='all',     # 所有人
        label='Mask (100% coverage, 30% efficacy)'
    )
    
    sim1 = cv.Sim(pars, interventions=mask1, label='Mask 100% coverage')
    
    # 部分覆盖率：80%的人戴口罩，降低40%传染率
    mask2 = mask_wearing(
        start_day=10,
        coverage=0.8,     # 80%覆盖率
        efficacy=0.4,     # 降低40%传染率
        target='all',
        label='Mask (80% coverage, 40% efficacy)'
    )
    
    sim2 = cv.Sim(pars, interventions=mask2, label='Mask 80% coverage')
    
    # 只给易感者戴口罩（保护易感者）
    mask3 = mask_wearing(
        start_day=10,
        coverage=1.0,
        efficacy=0.3,
        target='susceptible',  # 只给易感者
        label='Mask for susceptible only'
    )
    
    sim3 = cv.Sim(pars, interventions=mask3, label='Mask (susceptible only)')
    
    # 运行并对比
    msim = cv.parallel([baseline, sim1, sim2, sim3])
    msim.plot()
    
    return msim


# ============================================================================
# 方法3：结合使用（change_beta + 自定义类）
# ============================================================================
def method3_combined():
    """
    结合使用多种干预措施
    例如：戴口罩 + 社交距离
    """
    
    pars = dict(
        pop_size=1000,
        pop_infected=20,
        start_day='2021-07-04',
        n_days=60,
    )
    
    # 基线
    baseline = cv.Sim(pars, label='Baseline')
    
    # 组合干预：戴口罩 + 社交距离（降低接触）
    mask = cv.change_beta(days=10, changes=0.7, label='Mask wearing')
    distancing = cv.change_beta(days=10, changes=0.8, label='Social distancing')
    
    # 可以传入多个干预措施
    sim = cv.Sim(pars, interventions=[mask, distancing], label='Mask + Distancing')
    
    msim = cv.parallel([baseline, sim])
    msim.plot()
    
    return msim


if __name__ == '__main__':
    print("=" * 60)
    print("方法1：使用 change_beta（最简单）")
    print("=" * 60)
    msim1 = method1_change_beta()
    
    print("\n" + "=" * 60)
    print("方法2：自定义口罩干预类（更灵活）")
    print("=" * 60)
    msim2 = method2_custom_mask()
    
    print("\n" + "=" * 60)
    print("方法3：组合干预措施")
    print("=" * 60)
    msim3 = method3_combined()
