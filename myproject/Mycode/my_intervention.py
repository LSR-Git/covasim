# 政策/干预类：从 compose_intervention 迁移，供组合情景复用
import numpy as np
import covasim as cv
import covasim.defaults as cvd


# 默认区域键与名称（与 compose_intervention 中 _region_key / _region_name_a|b 一致）
_region_key = 'position'
_region_name_a = 'A'
_region_name_b = 'B'


# ========== 1. 接触者追踪：仅追踪指定区域 ==========
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


# ========== 2. 境内流动限制：仅指定区域 base 层减边 ==========
class reduce_region_a_contacts(cv.Intervention):
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


# ========== 2b. 境内流动：按阶段对指定区域 base 层 beta 乘系数（可阶段 4 恢复为 1.0） ==========
class ScaleRegionBaseBetaByPhase(cv.Intervention):
    '''按日对涉及指定区域的 base 层边的 beta 设为当前阶段系数，用于境内流动 部分/增强/无限制 分阶段。
    day_scale_pairs: [(day_start, scale), ...] 升序，表示从 day_start 起使用 scale，直至下一段起始日。如 [(0,1.0), (17,0.5), (34,0.3), (42,1.0)]。'''
    def __init__(self, region_key=None, region_name=None, day_scale_pairs=None, **kwargs):
        super().__init__(**kwargs)
        self.region_key = region_key if region_key is not None else _region_key
        self.region_name = region_name if region_name is not None else _region_name_a
        self.day_scale_pairs = sorted(day_scale_pairs or [(0, 1.0)], key=lambda x: x[0])

    def _scale_for_day(self, t):
        s = 1.0
        for day_start, scale in self.day_scale_pairs:
            if t >= day_start:
                s = scale
        return cvd.default_float(s)

    def apply(self, sim):
        if 'base' not in sim.people.contacts:
            return
        people = sim.people
        position = getattr(people, self.region_key, None)
        country = getattr(people, 'country', None)
        if position is None:
            return
        t = sim.t
        scale = self._scale_for_day(t)
        in_a = (np.asarray(position) == self.region_name)
        is_abroad = (np.asarray(position) != np.asarray(country))
        layer = people.contacts['base']
        p1, p2 = layer['p1'], layer['p2']
        beta = layer['beta']
        edge_in_a = in_a[p1] | in_a[p2]
        edge_abroad = is_abroad[p1] | is_abroad[p2]
        domestic_in_a = edge_in_a & ~edge_abroad
        beta[domestic_in_a] = scale


# ========== 3. 候鸟动态跨境 ==========
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


# ========== 3b. 候鸟动态跨境（多层网络专用） ==========
class CrosserTravelMultilayer(cv.Intervention):
    '''多层网络专用：候鸟跨境时原属地各层（home/school/work/community）权重冻结，
    跨区层（cross_work/cross_community/cross_home）按 crosser_purpose 激活。
    务工：cross_work+cross_community；探亲：cross_home+cross_community；偷渡：仅 cross_community。

    使用前提：
      - popdict 须由 CrossNetwork.add_cross_layer_multilayer 生成，含 crosser、crosser_purpose、position、country
      - beta_layer 须包含 cross_work、cross_community、cross_home（如 0.6）

    参数：
      frac_cross_per_day: 每日出境候鸟占境内候鸟的比例（0~1），默认 0.1
      duration_min, duration_max: 境外停留天数范围（含端点），默认 1~7
      start_day: 开始出境的仿真日，默认 0
      end_day_outbound: 停止新出境的仿真日，None 表示不限制
      resume_day_outbound: 恢复新出境的仿真日，若设且 t>=resume 则忽略 end_day_outbound 恢复派出（用于严控→温和）
      region_key: 位置属性名，默认 'position'
      region_name_a, region_name_b: 两区名称，默认 'A'、'B'

    示例：
      from my_intervention import CrosserTravelMultilayer
      popdict = CrossNetwork.add_cross_layer_multilayer(popdict, ...)
      cv.Sim(pars={'beta_layer': {..., 'cross_work': 0.6, 'cross_community': 0.6, 'cross_home': 0.6}},
             interventions=[CrosserTravelMultilayer(frac_cross_per_day=0.1, duration_min=1, duration_max=7)])
    '''
    def __init__(
        self,
        frac_cross_per_day=0.1,
        duration_min=1,
        duration_max=7,
        start_day=0,
        end_day_outbound=None,
        resume_day_outbound=None,
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
        self.resume_day_outbound = resume_day_outbound
        self.region_key = region_key if region_key is not None else _region_key
        self.region_name_a = region_name_a if region_name_a is not None else _region_name_a
        self.region_name_b = region_name_b if region_name_b is not None else _region_name_b
        self._return_day = None
        self._cross_betas = {}

    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)
        if self.end_day_outbound is not None:
            self.end_day_outbound = sim.day(self.end_day_outbound)
        if self.resume_day_outbound is not None:
            self.resume_day_outbound = sim.day(self.resume_day_outbound)
        n = sim.n
        self._return_day = np.full(n, np.nan, dtype=float)
        for lkey in ['cross_work', 'cross_community', 'cross_home']:
            if lkey in sim['beta_layer']:
                self._cross_betas[lkey] = float(sim['beta_layer'][lkey])
            else:
                self._cross_betas[lkey] = 0.6
        # 确保区内层有 beta 数组
        for lkey in ['home', 'school', 'work', 'community']:
            if lkey in sim.people.contacts:
                layer = sim.people.contacts[lkey]
                if 'beta' not in layer or len(layer['beta']) != len(layer['p1']):
                    layer['beta'] = np.ones(len(layer['p1']), dtype=cvd.default_float)

    def apply(self, sim):
        t = sim.t
        people = sim.people
        position = getattr(people, self.region_key, None)
        country = getattr(people, 'country', None)
        crosser = getattr(people, 'crosser', None)
        crosser_purpose = getattr(people, 'crosser_purpose', None)
        if position is None or country is None or crosser is None:
            return
        return_day = self._return_day

        # 1) 到期者回国
        returning = crosser & (return_day == t) & ~people.quarantined & ~people.isolated
        if np.any(returning):
            position[returning] = country[returning]
            return_day[returning] = np.nan

        # 2) 从境内候鸟中按比例随机选人出境
        allow_outbound = (
            t >= self.start_day
            and (
                self.end_day_outbound is None
                or t < self.end_day_outbound
                or (self.resume_day_outbound is not None and t >= self.resume_day_outbound)
            )
        )
        if allow_outbound:
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
                    from_a = (np.asarray(country[go_inds]) == self.region_name_a)
                    position[go_inds] = np.where(from_a, self.region_name_b, self.region_name_a)

        # 3) 原属地各层权重冻结
        is_abroad = (np.asarray(position) != np.asarray(country))
        for lkey in ['home', 'school', 'work', 'community']:
            if lkey not in people.contacts:
                continue
            layer = people.contacts[lkey]
            p1, p2 = layer['p1'], layer['p2']
            beta = layer['beta']
            edge_abroad = is_abroad[p1] | is_abroad[p2]
            beta[edge_abroad] = cvd.default_float(0.0)
            beta[~edge_abroad] = cvd.default_float(1.0)

        # 4) 跨区层按 purpose 激活
        if crosser_purpose is None:
            crosser_purpose = np.empty(people.n, dtype=object)
            crosser_purpose[:] = ''
        purpose = np.asarray(crosser_purpose)

        for lkey in ['cross_work', 'cross_community', 'cross_home']:
            if lkey not in people.contacts:
                continue
            layer = people.contacts[lkey]
            p1, p2 = layer['p1'], layer['p2']
            beta = layer['beta']
            cb = self._cross_betas.get(lkey, 0.6)
            # 每条边一端为 crosser，判断该 crosser 是否 abroad 且符合 purpose
            active = np.zeros(len(p1), dtype=bool)
            for i in range(len(p1)):
                c_ind = p1[i] if crosser[p1[i]] else p2[i]
                if not is_abroad[c_ind]:
                    continue
                if lkey == 'cross_work' and purpose[c_ind] != 'work':
                    continue
                if lkey == 'cross_home' and purpose[c_ind] != 'visit':
                    continue
                active[i] = True
            beta[active] = cvd.default_float(cb)
            beta[~active] = cvd.default_float(0.0)


# ========== 3c. 多层级口罩佩戴（指定层、仅 A 区） ==========
class MaskWearingLayerSpecific(cv.Intervention):
    '''多层网络专用：在指定层（work、school）对涉及 A 区的 domestic 边，将 layer["beta"] 设为 efficacy。
    须放在 CrosserTravelMultilayer 之后，因其每日重写 layer["beta"]，本干预在其后覆盖 domestic 边的值。
    常规策略：工作层、学校层口罩佩戴，仅 A 区，100% 依从性。'''
    def __init__(
        self,
        layers=None,
        efficacy=0.5,
        start_day=0,
        end_day=None,
        region_key=None,
        region_name_a=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = layers if layers is not None else ['work', 'school']
        self.efficacy = cvd.default_float(efficacy)
        self.start_day = start_day
        self.end_day = end_day
        self.region_key = region_key if region_key is not None else _region_key
        self.region_name_a = region_name_a if region_name_a is not None else _region_name_a

    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)
        if self.end_day is not None:
            self.end_day = sim.day(self.end_day)

    def apply(self, sim):
        if sim.t < self.start_day:
            return
        if self.end_day is not None and sim.t > self.end_day:
            return
        people = sim.people
        position = getattr(people, self.region_key, None)
        country = getattr(people, 'country', None)
        if position is None or country is None:
            return
        in_a = (np.asarray(position) == self.region_name_a)
        is_abroad = (np.asarray(position) != np.asarray(country))
        for lkey in self.layers:
            if lkey not in people.contacts:
                continue
            layer = people.contacts[lkey]
            p1, p2 = layer['p1'], layer['p2']
            beta = layer['beta']
            edge_in_a = in_a[p1] | in_a[p2]
            edge_abroad = is_abroad[p1] | is_abroad[p2]
            domestic_in_a = edge_in_a & ~edge_abroad
            beta[domestic_in_a] = self.efficacy


# ========== 3e. A 区居家办公（工作层减边） ==========
class WorkFromHomeA(cv.Intervention):
    '''仅对 A 区（position=A）人员的工作层移除 70% 接触边（保留 30%）。通过 layer.pop_inds 实际移除边，非修改 beta。'''
    def __init__(self, start_day=0, end_day=None, region_key=None, region_name_a=None, fraction=0.3, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.start_day = start_day
        self.end_day = end_day
        self.region_key = region_key if region_key is not None else _region_key
        self.region_name_a = region_name_a if region_name_a is not None else _region_name_a
        self.fraction = fraction
        self.seed = seed
        self._stored_contacts = None
        self._applied = False

    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)
        if self.end_day is not None:
            self.end_day = sim.day(self.end_day)

    def apply(self, sim):
        lkey = 'work'
        if lkey not in sim.people.contacts:
            return
        layer = sim.people.contacts[lkey]
        region = getattr(sim.people, self.region_key, None)
        if region is None:
            return
        in_a = (np.asarray(region) == self.region_name_a)

        if sim.t == self.start_day and not self._applied:
            p1, p2 = layer['p1'][:], layer['p2'][:]
            edge_in_a = in_a[p1] | in_a[p2]
            n_total = edge_in_a.sum()
            if n_total == 0:
                return
            n_remove = int(n_total * (1 - self.fraction))
            if n_remove <= 0:
                return
            inds_all = np.where(edge_in_a)[0]
            rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
            rng.shuffle(inds_all)
            to_remove = inds_all[:n_remove]
            self._stored_contacts = layer.pop_inds(to_remove)
            self._applied = True
        elif self.end_day is not None and sim.t == self.end_day and self._applied:
            layer.append(self._stored_contacts)
            self._applied = False


# ========== 3g. A 区学校停课（学校层减边） ==========
class SchoolCloseA(cv.Intervention):
    '''仅对 A 区（position=A）人员的学校层移除全部接触边（全面停学）。通过 layer.pop_inds 实际移除边。'''
    def __init__(self, start_day=0, end_day=None, region_key=None, region_name_a=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.start_day = start_day
        self.end_day = end_day
        self.region_key = region_key if region_key is not None else _region_key
        self.region_name_a = region_name_a if region_name_a is not None else _region_name_a
        self.seed = seed
        self._stored_contacts = None
        self._applied = False

    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)
        if self.end_day is not None:
            self.end_day = sim.day(self.end_day)

    def apply(self, sim):
        lkey = 'school'
        if lkey not in sim.people.contacts:
            return
        layer = sim.people.contacts[lkey]
        region = getattr(sim.people, self.region_key, None)
        if region is None:
            return
        in_a = (np.asarray(region) == self.region_name_a)

        if sim.t == self.start_day and not self._applied:
            p1, p2 = layer['p1'][:], layer['p2'][:]
            edge_in_a = in_a[p1] | in_a[p2]
            n_total = edge_in_a.sum()
            if n_total == 0:
                return
            inds_all = np.where(edge_in_a)[0]
            rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
            rng.shuffle(inds_all)
            self._stored_contacts = layer.pop_inds(inds_all)
            self._applied = True
        elif self.end_day is not None and sim.t == self.end_day and self._applied:
            layer.append(self._stored_contacts)
            self._applied = False


# ========== 3h. A 区社区接触限制（社区层减边） ==========
class CommunityRestrictA(cv.Intervention):
    '''仅对 A 区（position=A）人员的社区层移除 50% 接触边（保留 50%）。通过 layer.pop_inds 实际移除边，非修改 beta。'''
    def __init__(self, start_day=0, end_day=None, region_key=None, region_name_a=None, fraction=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.start_day = start_day
        self.end_day = end_day
        self.region_key = region_key if region_key is not None else _region_key
        self.region_name_a = region_name_a if region_name_a is not None else _region_name_a
        self.fraction = fraction
        self.seed = seed
        self._stored_contacts = None
        self._applied = False

    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)
        if self.end_day is not None:
            self.end_day = sim.day(self.end_day)

    def apply(self, sim):
        lkey = 'community'
        if lkey not in sim.people.contacts:
            return
        layer = sim.people.contacts[lkey]
        region = getattr(sim.people, self.region_key, None)
        if region is None:
            return
        in_a = (np.asarray(region) == self.region_name_a)

        if sim.t == self.start_day and not self._applied:
            p1, p2 = layer['p1'][:], layer['p2'][:]
            edge_in_a = in_a[p1] | in_a[p2]
            n_total = edge_in_a.sum()
            if n_total == 0:
                return
            n_remove = int(n_total * (1 - self.fraction))
            if n_remove <= 0:
                return
            inds_all = np.where(edge_in_a)[0]
            rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
            rng.shuffle(inds_all)
            to_remove = inds_all[:n_remove]
            self._stored_contacts = layer.pop_inds(to_remove)
            self._applied = True
        elif self.end_day is not None and sim.t == self.end_day and self._applied:
            layer.append(self._stored_contacts)
            self._applied = False


# ========== 4. 口罩佩戴（单阶段） ==========
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


# ========== 4b. 口罩放松（撤销此前口罩对 rel_trans 的降低） ==========
class MaskRelax(cv.Intervention):
    '''在 start_day 当天对 subtarget 人群执行 rel_trans /= efficacy，用于“摘口罩”（撤销此前 MaskWearing/MaskWearingTwoPhase 的乘数）。'''
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
            relax_inds = inds
        else:
            n_relax = min(len(inds), int(len(inds) * self.fraction + 0.5))
            relax_inds = np.random.choice(inds, size=n_relax, replace=False) if n_relax > 0 else np.array([], dtype=int)
        if len(relax_inds) > 0 and self.efficacy != 0:
            sim.people.rel_trans[relax_inds] /= self.efficacy
        self._applied = True


# ========== 4c. 偷渡者注入：注入日向 A 区注入 n 个立即传染且不可检测的感染者 ==========
class InjectUndocumentedInfectious(cv.Intervention):
    '''在 inject_day 从 A 区易感者中选 n 人标记为偷渡者并感染，当日即传染；偷渡者不可被检测/隔离（需配合排除 undocumented 的 subtarget）。'''
    def __init__(self, inject_day, n, region_key=None, region_name_a=None, **kwargs):
        super().__init__(**kwargs)
        self.inject_day = inject_day
        self.n = int(n)
        self.region_key = region_key if region_key is not None else _region_key
        self.region_name_a = region_name_a if region_name_a is not None else _region_name_a
        self._applied = False

    def initialize(self, sim):
        super().initialize()
        self.inject_day = sim.day(self.inject_day)
        if not hasattr(sim.people, 'undocumented'):
            sim.people.undocumented = np.zeros(sim.n, dtype=bool)

    def apply(self, sim):
        if sim.t != self.inject_day or self._applied:
            return
        people = sim.people
        position = getattr(people, self.region_key, None)
        if position is None:
            return
        in_a = (np.asarray(position) == self.region_name_a)
        candidates = np.where(in_a & people.susceptible)[0]
        n_inject = min(self.n, len(candidates))
        if n_inject <= 0:
            self._applied = True
            return
        inds = np.random.choice(candidates, size=n_inject, replace=False)
        people.undocumented[inds] = True
        people.infect(inds, source=None, layer=None)
        people.dur_exp2inf[inds] = 0
        people.date_infectious[inds] = sim.t
        self._applied = True


# ========== 5. 两阶段口罩佩戴 ==========
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
