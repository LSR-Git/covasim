# Covasim 自定义干预措施完整指南

## 📖 目录
1. [什么是自定义干预措施](#1-什么是自定义干预措施)
2. [基本结构](#2-基本结构)
3. [核心方法详解](#3-核心方法详解)
4. [完整示例](#4-完整示例)
5. [高级技巧](#5-高级技巧)
6. [常见场景](#6-常见场景)
7. [最佳实践](#7-最佳实践)
8. [调试技巧](#8-调试技巧)

---

## 1. 什么是自定义干预措施

### 为什么需要自定义干预措施？

Covasim 内置了多种干预措施（如测试、接触者追踪、疫苗等），但在某些情况下，你可能需要：

- ✅ 实现特定的政策措施（如针对特定年龄组的保护措施）
- ✅ 根据复杂条件动态调整参数
- ✅ 收集和分析特定的数据
- ✅ 实现新的干预逻辑（如自定义的隔离策略）
- ✅ 组合多种干预效果

### 自定义干预措施的本质

自定义干预措施就是一个**继承自 `cv.Intervention` 的 Python 类**，它可以：
- 在模拟的每个时间步被调用
- 访问和修改模拟对象（`sim`）的任何属性
- 存储自己的状态和数据
- 在特定条件下执行特定操作

---

## 2. 基本结构

### 最小化模板

```python
import covasim as cv
import numpy as np

class MyIntervention(cv.Intervention):
    """
    我的自定义干预措施
    
    Args:
        start_day (int/str): 开始日期
        end_day (int/str): 结束日期
        **kwargs: 传递给父类的其他参数
    """
    
    def __init__(self, start_day=None, end_day=None, **kwargs):
        super().__init__(**kwargs)  # ⚠️ 必须调用父类的 __init__
        self.start_day = start_day
        self.end_day = end_day
        return
    
    def initialize(self, sim):
        """初始化干预措施（在模拟开始时调用一次）"""
        super().initialize()  # ⚠️ 必须调用父类的 initialize
        # 将日期字符串转换为整数
        self.start_day = sim.day(self.start_day)
        self.end_day = sim.day(self.end_day)
        # 其他初始化操作...
        return
    
    def apply(self, sim):
        """应用干预措施（每个时间步都会调用）"""
        # 在这里实现你的干预逻辑
        if self.start_day <= sim.t <= self.end_day:
            # 执行干预操作
            pass
        return
```

---

## 3. 核心方法详解

### 3.1 `__init__()` - 构造函数

**作用**: 初始化干预措施的参数

**必须做的事情**:
- ✅ 调用 `super().__init__(**kwargs)` 
- ✅ 存储所有需要的参数

```python
def __init__(self, param1, param2=10, **kwargs):
    super().__init__(**kwargs)  # 必须的！
    self.param1 = param1
    self.param2 = param2
    # 可以初始化一些属性
    self.my_data = []
    return
```

**kwargs 中有什么**:
- `label`: 干预措施的标签
- `show_label`: 是否在图例中显示标签
- `do_plot`: 是否绘制干预措施
- `line_args`: 绘图参数

---

### 3.2 `initialize(sim)` - 初始化方法

**作用**: 在模拟开始时进行初始化（只调用一次）

**适合做什么**:
- ✅ 转换日期格式（字符串 → 整数）
- ✅ 根据模拟参数初始化数据结构
- ✅ 预计算一些值
- ✅ 找到目标人群的索引

```python
def initialize(self, sim):
    super().initialize()  # 必须的！
    
    # 1. 转换日期
    self.start_day = sim.day(self.start_day)
    self.end_day = sim.day(self.end_day)
    self.days = [self.start_day, self.end_day]
    
    # 2. 找到目标人群
    self.target_inds = sim.people.age > 65  # 找到65岁以上的人
    
    # 3. 初始化数据存储
    self.results = np.zeros(sim.npts)  # 创建结果数组
    self.tvec = sim.tvec  # 保存时间向量
    
    return
```

**重要的 sim 属性**:
- `sim.t`: 当前时间步（整数）
- `sim.npts`: 模拟的总时间点数
- `sim.tvec`: 时间向量
- `sim.people`: 人群对象
- `sim.day(date)`: 将日期转换为整数
- `sim.date(day)`: 将整数转换为日期字符串

---

### 3.3 `apply(sim)` - 应用方法

**作用**: 在每个时间步应用干预措施（核心方法）

**适合做什么**:
- ✅ 检查当前时间是否在干预期间
- ✅ 修改 sim.people 的属性
- ✅ 收集数据
- ✅ 根据条件执行操作

```python
def apply(self, sim):
    """
    每个时间步都会调用这个方法
    sim.t 是当前的时间步（整数）
    """
    
    # 方式1: 在特定日期开始/结束
    if sim.t == self.start_day:
        # 开始干预
        sim.people.rel_sus[self.target_inds] = 0.5
    elif sim.t == self.end_day:
        # 结束干预
        sim.people.rel_sus[self.target_inds] = 1.0
    
    # 方式2: 在一段时间内持续生效
    if self.start_day <= sim.t <= self.end_day:
        # 每天都执行的操作
        pass
    
    # 方式3: 根据条件触发
    if sim.results['n_exposed'][sim.t] > 1000:
        # 当感染人数超过阈值时触发
        pass
    
    # 收集数据
    self.results[sim.t] = sim.people.exposed.sum()
    
    return
```

---

### 3.4 `finalize(sim)` - 结束方法（可选）

**作用**: 在模拟结束后进行清理或后处理

```python
def finalize(self, sim):
    super().finalize()
    # 进行一些后处理，如数据归一化
    self.total_protected = self.results.sum()
    return
```

---

### 3.5 `plot()` - 绘图方法（可选）

**作用**: 自定义绘图

```python
def plot(self):
    import pylab as pl
    pl.figure()
    pl.plot(self.tvec, self.results)
    pl.xlabel('Day')
    pl.ylabel('Number protected')
    pl.title('Impact of intervention')
    return
```

---

## 4. 完整示例

### 示例 1: 保护老年人

这是一个完整的示例，展示如何在特定时间段内降低老年人的易感性。

```python
import numpy as np
import pylab as pl
import covasim as cv

class protect_elderly(cv.Intervention):
    """
    保护老年人干预措施
    
    在指定时间段内降低老年人的易感性（例如，通过隔离措施）
    
    Args:
        start_day (int/str): 开始日期
        end_day (int/str): 结束日期
        age_cutoff (float): 年龄阈值（默认70岁）
        rel_sus (float): 相对易感性（0.0-1.0，越小保护越强）
        **kwargs: 其他参数
    """
    
    def __init__(self, start_day=None, end_day=None, age_cutoff=70, rel_sus=0.0, **kwargs):
        super().__init__(**kwargs)
        self.start_day = start_day
        self.end_day = end_day
        self.age_cutoff = age_cutoff
        self.rel_sus = rel_sus
        return
    
    def initialize(self, sim):
        super().initialize()
        
        # 转换日期
        self.start_day = sim.day(self.start_day)
        self.end_day = sim.day(self.end_day)
        self.days = [self.start_day, self.end_day]
        
        # 找到老年人
        self.elderly = sim.people.age > self.age_cutoff
        
        # 初始化结果存储
        self.exposed = np.zeros(sim.npts)
        self.tvec = sim.tvec
        
        print(f"找到 {self.elderly.sum()} 名老年人（年龄 > {self.age_cutoff}）")
        return
    
    def apply(self, sim):
        # 记录每天老年人中的暴露数量
        self.exposed[sim.t] = sim.people.exposed[self.elderly].sum()
        
        # 开始干预
        if sim.t == self.start_day:
            sim.people.rel_sus[self.elderly] = self.rel_sus
            print(f"第 {sim.t} 天: 开始保护老年人 (rel_sus={self.rel_sus})")
        
        # 结束干预
        elif sim.t == self.end_day:
            sim.people.rel_sus[self.elderly] = 1.0
            print(f"第 {sim.t} 天: 结束保护老年人")
        
        return
    
    def plot(self):
        pl.figure()
        pl.plot(self.tvec, self.exposed)
        pl.axvline(self.start_day, linestyle='--', color='g', label='开始干预')
        pl.axvline(self.end_day, linestyle='--', color='r', label='结束干预')
        pl.xlabel('Day')
        pl.ylabel('Number exposed')
        pl.title('Number of elderly people exposed to COVID')
        pl.legend()
        return

# 使用示例
if __name__ == '__main__':
    pars = dict(
        pop_size = 50000,
        pop_infected = 100,
        n_days = 90,
        verbose = 0.1,
    )
    
    # 创建干预措施
    protect = protect_elderly(
        start_day='2020-04-01', 
        end_day='2020-05-01', 
        age_cutoff=70,
        rel_sus=0.1,
        label='Protect elderly'
    )
    
    # 运行模拟
    sim = cv.Sim(pars, interventions=protect)
    sim.run()
    sim.plot()
    
    # 绘制干预措施的结果
    protect.plot()
```

---

### 示例 2: 动态触发的封锁措施

当感染人数超过阈值时自动触发封锁，低于阈值时解除。

```python
import covasim as cv
import numpy as np

class dynamic_lockdown(cv.Intervention):
    """
    动态封锁措施
    
    当活跃感染人数超过阈值时触发封锁，低于阈值时解除
    
    Args:
        threshold (int): 触发封锁的感染人数阈值
        beta_multiplier (float): 封锁期间的传播率乘数（<1 表示降低传播）
        min_duration (int): 最小封锁持续天数
        **kwargs: 其他参数
    """
    
    def __init__(self, threshold=1000, beta_multiplier=0.3, min_duration=14, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.beta_multiplier = beta_multiplier
        self.min_duration = min_duration
        return
    
    def initialize(self, sim):
        super().initialize()
        
        # 状态跟踪
        self.in_lockdown = False
        self.lockdown_start = None
        self.lockdown_history = []  # 记录封锁历史
        
        # 记录原始 beta 值
        self.original_beta = sim['beta']
        
        # 数据存储
        self.lockdown_status = np.zeros(sim.npts)  # 0=正常, 1=封锁
        
        return
    
    def apply(self, sim):
        current_infections = sim.people.infectious.sum()
        
        # 检查是否应该开始封锁
        if not self.in_lockdown and current_infections > self.threshold:
            self.in_lockdown = True
            self.lockdown_start = sim.t
            sim['beta'] = self.original_beta * self.beta_multiplier
            print(f"第 {sim.t} 天: 触发封锁 (感染人数: {current_infections})")
        
        # 检查是否应该解除封锁
        elif self.in_lockdown:
            days_in_lockdown = sim.t - self.lockdown_start
            if current_infections < self.threshold and days_in_lockdown >= self.min_duration:
                self.in_lockdown = False
                sim['beta'] = self.original_beta
                self.lockdown_history.append((self.lockdown_start, sim.t))
                print(f"第 {sim.t} 天: 解除封锁 (持续 {days_in_lockdown} 天)")
        
        # 记录当前状态
        self.lockdown_status[sim.t] = 1 if self.in_lockdown else 0
        
        return
    
    def finalize(self, sim):
        super().finalize()
        # 如果模拟结束时还在封锁中，记录下来
        if self.in_lockdown:
            self.lockdown_history.append((self.lockdown_start, sim.t))
        print(f"\n封锁历史: {len(self.lockdown_history)} 次封锁")
        for i, (start, end) in enumerate(self.lockdown_history):
            print(f"  第 {i+1} 次: 第 {start}-{end} 天 (持续 {end-start} 天)")
        return

# 使用示例
if __name__ == '__main__':
    pars = dict(
        pop_size = 20000,
        pop_infected = 50,
        n_days = 180,
        verbose = 0.1,
    )
    
    lockdown = dynamic_lockdown(
        threshold=500, 
        beta_multiplier=0.3,
        min_duration=21,
        label='Dynamic lockdown'
    )
    
    sim = cv.Sim(pars, interventions=lockdown)
    sim.run()
    sim.plot()
```

---

### 示例 3: 针对特定人群的测试策略

根据职业、年龄等属性对不同人群实施不同的测试策略。

```python
import covasim as cv
import numpy as np

class targeted_testing(cv.Intervention):
    """
    针对性测试策略
    
    对不同人群实施不同的测试概率
    
    Args:
        start_day (int/str): 开始日期
        high_risk_age (tuple): 高风险年龄范围 (min, max)
        high_risk_prob (float): 高风险人群测试概率
        normal_prob (float): 普通人群测试概率
        test_sensitivity (float): 测试灵敏度
        **kwargs: 其他参数
    """
    
    def __init__(self, start_day=0, high_risk_age=(60, 100), 
                 high_risk_prob=0.3, normal_prob=0.05, 
                 test_sensitivity=0.9, **kwargs):
        super().__init__(**kwargs)
        self.start_day = start_day
        self.high_risk_age = high_risk_age
        self.high_risk_prob = high_risk_prob
        self.normal_prob = normal_prob
        self.test_sensitivity = test_sensitivity
        return
    
    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)
        
        # 识别高风险人群
        ages = sim.people.age
        self.high_risk = (ages >= self.high_risk_age[0]) & (ages <= self.high_risk_age[1])
        self.normal_risk = ~self.high_risk
        
        # 数据统计
        self.n_tested = np.zeros(sim.npts)
        self.n_diagnosed = np.zeros(sim.npts)
        
        print(f"高风险人群: {self.high_risk.sum()} 人")
        print(f"普通人群: {self.normal_risk.sum()} 人")
        return
    
    def apply(self, sim):
        if sim.t < self.start_day:
            return
        
        # 确定要测试的人（症状性或随机）
        symptomatic = sim.people.symptomatic
        
        # 高风险人群测试
        high_risk_test_inds = cv.true(self.high_risk & symptomatic)
        if len(high_risk_test_inds) > 0:
            test_probs = np.full(len(high_risk_test_inds), self.high_risk_prob)
            to_test = cv.binomial_filter(test_probs, high_risk_test_inds)
            self._do_test(sim, to_test)
        
        # 普通人群测试
        normal_test_inds = cv.true(self.normal_risk & symptomatic)
        if len(normal_test_inds) > 0:
            test_probs = np.full(len(normal_test_inds), self.normal_prob)
            to_test = cv.binomial_filter(test_probs, normal_test_inds)
            self._do_test(sim, to_test)
        
        return
    
    def _do_test(self, sim, inds):
        """执行测试的辅助函数"""
        if len(inds) == 0:
            return
        
        # 标记为已测试
        sim.people.tested[inds] = True
        sim.people.date_tested[inds] = sim.t
        self.n_tested[sim.t] += len(inds)
        
        # 确定测试结果
        is_infectious = cv.true(sim.people.infectious[inds])
        if len(is_infectious) > 0:
            # 考虑测试灵敏度
            is_diagnosed = cv.n_binomial(self.test_sensitivity, len(is_infectious))
            diagnosed_inds = inds[is_infectious[is_diagnosed]]
            
            # 标记为已诊断
            sim.people.diagnosed[diagnosed_inds] = True
            sim.people.date_diagnosed[diagnosed_inds] = sim.t
            self.n_diagnosed[sim.t] += len(diagnosed_inds)
        
        return

# 使用示例
if __name__ == '__main__':
    pars = dict(
        pop_size = 10000,
        pop_infected = 20,
        n_days = 90,
        verbose = 0.1,
    )
    
    testing = targeted_testing(
        start_day=10,
        high_risk_age=(60, 100),
        high_risk_prob=0.5,
        normal_prob=0.1,
        label='Targeted testing'
    )
    
    sim = cv.Sim(pars, interventions=testing)
    sim.run()
    sim.plot()
```

---

## 5. 高级技巧

### 5.1 访问和修改人群属性

```python
def apply(self, sim):
    # 获取人群对象
    people = sim.people
    
    # 常用属性（布尔数组）
    susceptible = people.susceptible    # 易感者
    exposed = people.exposed            # 暴露者
    infectious = people.infectious      # 感染者
    symptomatic = people.symptomatic    # 有症状者
    diagnosed = people.diagnosed        # 已诊断者
    recovered = people.recovered        # 康复者
    dead = people.dead                  # 死亡者
    quarantined = people.quarantined    # 隔离者
    
    # 人口学属性
    age = people.age                    # 年龄
    sex = people.sex                    # 性别 (0=女, 1=男)
    
    # 修改易感性和症状概率
    people.rel_sus[inds] = 0.5         # 降低易感性
    people.rel_trans[inds] = 0.3       # 降低传播性
    people.symp_prob[inds] = 0.2       # 降低症状概率
    
    # 日期属性
    date_exposed = people.date_exposed
    date_symptomatic = people.date_symptomatic
    
    return
```

### 5.2 使用层（Layers）

```python
def initialize(self, sim):
    super().initialize()
    
    # 获取所有层的名称
    layer_keys = sim.people.contacts.keys()  # ['h', 's', 'w', 'c']
    
    # 访问特定层的接触网络
    household_contacts = sim.people.contacts['h']
    school_contacts = sim.people.contacts['s']
    work_contacts = sim.people.contacts['w']
    community_contacts = sim.people.contacts['c']
    
    return

def apply(self, sim):
    # 修改特定层的传播系数
    if sim.t == self.start_day:
        # 关闭学校（将学校层的 beta 设为 0）
        sim['beta_layer']['s'] = 0
        
    elif sim.t == self.end_day:
        # 重新开放学校
        sim['beta_layer']['s'] = 1.0
    
    return
```

### 5.3 使用辅助函数

```python
import covasim as cv

def apply(self, sim):
    # 找到满足条件的人的索引
    symptomatic_inds = cv.true(sim.people.symptomatic)  # 返回 True 的索引
    
    # 二项分布过滤（随机选择）
    # 从 symptomatic_inds 中随机选择 30% 的人
    to_test = cv.binomial_filter(0.3, symptomatic_inds)
    
    # 随机选择 N 个人
    # 从 symptomatic_inds 中随机选择 100 个人
    to_test = cv.choose(100, symptomatic_inds)
    
    # 多项式抽样
    n_samples = cv.n_binomial(prob=0.3, n=len(symptomatic_inds))
    
    return
```

### 5.4 存储和访问结果

```python
def initialize(self, sim):
    super().initialize()
    
    # 方式1: 使用 NumPy 数组
    self.daily_count = np.zeros(sim.npts)
    
    # 方式2: 使用列表
    self.events = []
    
    return

def apply(self, sim):
    # 存储数据
    self.daily_count[sim.t] = sim.people.infectious.sum()
    self.events.append({'day': sim.t, 'count': self.daily_count[sim.t]})
    return

def finalize(self, sim):
    super().finalize()
    
    # 将结果添加到 sim.results
    sim.results['custom_metric'] = self.daily_count
    
    return
```

---

## 6. 常见场景

### 场景 1: 学校关闭

```python
class school_closure(cv.Intervention):
    def __init__(self, start_day, end_day, **kwargs):
        super().__init__(**kwargs)
        self.start_day = start_day
        self.end_day = end_day
    
    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)
        self.end_day = sim.day(self.end_day)
        self.original_beta_s = sim['beta_layer']['s']
        return
    
    def apply(self, sim):
        if sim.t == self.start_day:
            sim['beta_layer']['s'] = 0  # 关闭学校
        elif sim.t == self.end_day:
            sim['beta_layer']['s'] = self.original_beta_s  # 重新开放
        return
```

### 场景 2: 分阶段放松限制

```python
class phased_reopening(cv.Intervention):
    def __init__(self, phases, **kwargs):
        """
        phases: 列表，每个元素是 (day, beta_multiplier) 元组
        例如: [(50, 0.5), (70, 0.7), (90, 1.0)]
        """
        super().__init__(**kwargs)
        self.phases = phases
    
    def initialize(self, sim):
        super().initialize()
        self.original_beta = sim['beta']
        # 转换日期
        self.phases = [(sim.day(day), mult) for day, mult in self.phases]
        return
    
    def apply(self, sim):
        for day, multiplier in self.phases:
            if sim.t == day:
                sim['beta'] = self.original_beta * multiplier
                print(f"第 {sim.t} 天: 调整 beta 为 {sim['beta']:.3f}")
        return
```

### 场景 3: 有限容量的隔离设施

```python
class quarantine_with_capacity(cv.Intervention):
    def __init__(self, start_day, capacity=100, **kwargs):
        super().__init__(**kwargs)
        self.start_day = start_day
        self.capacity = capacity
    
    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)
        self.currently_quarantined = 0
        return
    
    def apply(self, sim):
        if sim.t < self.start_day:
            return
        
        # 找到新诊断的人
        newly_diagnosed = cv.true(sim.people.diagnosed & 
                                   (sim.people.date_diagnosed == sim.t))
        
        # 根据容量限制隔离
        can_quarantine = min(len(newly_diagnosed), 
                            self.capacity - self.currently_quarantined)
        
        if can_quarantine > 0:
            to_quarantine = newly_diagnosed[:can_quarantine]
            sim.people.quarantined[to_quarantine] = True
            self.currently_quarantined += can_quarantine
        
        # 更新当前隔离人数（假设隔离14天）
        # 这是简化版本，实际应该跟踪每个人的隔离开始日期
        
        return
```

### 场景 4: 基于疫苗接种率的政策调整

```python
class vaccine_triggered_reopening(cv.Intervention):
    def __init__(self, vaccination_threshold=0.7, beta_after=1.0, **kwargs):
        super().__init__(**kwargs)
        self.vaccination_threshold = vaccination_threshold
        self.beta_after = beta_after
    
    def initialize(self, sim):
        super().initialize()
        self.original_beta = sim['beta']
        self.triggered = False
        return
    
    def apply(self, sim):
        if not self.triggered:
            # 计算疫苗接种率
            n_vaccinated = sim.people.vaccinated.sum()
            vaccination_rate = n_vaccinated / len(sim.people)
            
            if vaccination_rate >= self.vaccination_threshold:
                sim['beta'] = self.beta_after
                self.triggered = True
                print(f"第 {sim.t} 天: 疫苗接种率达到 {vaccination_rate:.1%}，调整政策")
        return
```

---

## 7. 最佳实践

### ✅ DO（应该做的）

1. **总是调用父类方法**
   ```python
   def __init__(self, **kwargs):
       super().__init__(**kwargs)  # 必须！
   
   def initialize(self, sim):
       super().initialize()  # 必须！
   ```

2. **在 `initialize()` 中转换日期**
   ```python
   def initialize(self, sim):
       super().initialize()
       self.start_day = sim.day(self.start_day)  # 字符串 → 整数
   ```

3. **使用描述性的变量名和文档字符串**
   ```python
   class my_intervention(cv.Intervention):
       """
       清晰的描述
       
       Args:
           param1 (type): 参数说明
       """
   ```

4. **存储原始值**
   ```python
   def initialize(self, sim):
       super().initialize()
       self.original_beta = sim['beta']  # 保存原始值以便恢复
   ```

5. **添加调试输出**
   ```python
   def apply(self, sim):
       if sim.t == self.start_day:
           print(f"干预开始: day {sim.t}")
   ```

### ❌ DON'T（不应该做的）

1. **不要忘记调用父类方法**
   ```python
   def __init__(self, **kwargs):
       # super().__init__(**kwargs)  # ❌ 忘记调用了！
       self.param = value
   ```

2. **不要在 `__init__()` 中使用 sim**
   ```python
   def __init__(self, sim, **kwargs):  # ❌ 不要这样做
       super().__init__(**kwargs)
       self.people = sim.people  # ❌ sim 还没有初始化
   ```

3. **不要直接修改而不保存原始值**
   ```python
   def apply(self, sim):
       sim['beta'] = 0.5  # ❌ 没有保存原始值
   ```

4. **不要使用全局变量**
   ```python
   global_counter = 0  # ❌ 不要使用全局变量
   
   class my_intervention(cv.Intervention):
       def apply(self, sim):
           global global_counter  # ❌
           global_counter += 1
   ```

5. **不要在 `apply()` 中进行耗时操作**
   ```python
   def apply(self, sim):
       # ❌ 这会在每个时间步都执行，非常慢！
       for i in range(1000000):
           heavy_computation()
   ```

---

## 8. 调试技巧

### 8.1 添加打印语句

```python
def initialize(self, sim):
    super().initialize()
    print(f"干预初始化:")
    print(f"  开始日期: {self.start_day}")
    print(f"  目标人数: {self.target_inds.sum()}")
    return

def apply(self, sim):
    if sim.t % 10 == 0:  # 每10天打印一次
        print(f"第 {sim.t} 天: 当前状态 = {self.status}")
    return
```

### 8.2 检查数据类型和形状

```python
def initialize(self, sim):
    super().initialize()
    print(f"sim.npts = {sim.npts}, type = {type(sim.npts)}")
    print(f"sim.people.age.shape = {sim.people.age.shape}")
    print(f"目标索引数量: {self.target_inds.sum()}")
    return
```

### 8.3 使用断言

```python
def apply(self, sim):
    assert self.start_day <= sim.t <= self.end_day, "时间超出范围"
    assert len(to_test) <= len(sim.people), "测试人数超过总人数"
    return
```

### 8.4 绘制中间结果

```python
def finalize(self, sim):
    super().finalize()
    
    import pylab as pl
    pl.figure()
    pl.subplot(2, 1, 1)
    pl.plot(self.daily_count)
    pl.title('Daily count')
    
    pl.subplot(2, 1, 2)
    pl.plot(self.cumulative)
    pl.title('Cumulative')
    pl.tight_layout()
    pl.savefig('debug_intervention.png')
    
    return
```

### 8.5 使用 try-except

```python
def apply(self, sim):
    try:
        # 你的代码
        result = some_operation()
    except Exception as e:
        print(f"错误发生在第 {sim.t} 天:")
        print(f"  异常: {type(e).__name__}")
        print(f"  信息: {str(e)}")
        raise  # 重新抛出异常
    return
```

---

## 9. 完整运行示例

下面是一个完整的示例，展示如何创建、运行和可视化自定义干预措施：

```python
import numpy as np
import pylab as pl
import covasim as cv

# 定义自定义干预措施
class my_custom_intervention(cv.Intervention):
    def __init__(self, start_day, threshold, **kwargs):
        super().__init__(**kwargs)
        self.start_day = start_day
        self.threshold = threshold
    
    def initialize(self, sim):
        super().initialize()
        self.start_day = sim.day(self.start_day)
        self.results = np.zeros(sim.npts)
        return
    
    def apply(self, sim):
        if sim.t >= self.start_day:
            n_inf = sim.people.infectious.sum()
            self.results[sim.t] = n_inf
            
            if n_inf > self.threshold:
                sim['beta'] *= 0.9  # 降低传播率
        return

# 创建模拟参数
pars = dict(
    pop_size = 10000,
    pop_infected = 20,
    n_days = 100,
    verbose = 0.1,
)

# 创建干预措施
my_interv = my_custom_intervention(
    start_day=20,
    threshold=100,
    label='My intervention'
)

# 创建并运行模拟
sim = cv.Sim(pars, interventions=my_interv)
sim.run()

# 可视化
fig, axes = pl.subplots(2, 1, figsize=(10, 8))

# 绘制标准结果
sim.plot(fig=fig, ax=axes[0])

# 绘制自定义结果
axes[1].plot(my_interv.results)
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Infectious count')
axes[1].set_title('Custom intervention results')
axes[1].axhline(my_interv.threshold, linestyle='--', color='r', label='Threshold')
axes[1].legend()

pl.tight_layout()
pl.show()
```

---

## 10. 总结

### 关键要点

1. **自定义干预措施 = 继承 `cv.Intervention` 的类**
2. **三个核心方法**:
   - `__init__()`: 存储参数
   - `initialize(sim)`: 初始化（调用一次）
   - `apply(sim)`: 应用干预（每个时间步）
3. **总是调用父类方法**: `super().__init__()` 和 `super().initialize()`
4. **在 `initialize()` 中转换日期**: `sim.day(date_string)`
5. **通过 `sim.people` 访问和修改人群属性**

### 下一步

- 查看 `covasim/interventions.py` 了解内置干预措施的实现
- 查看 `examples/t05_*.py` 了解更多示例
- 尝试组合多个自定义干预措施
- 实验不同的触发条件和效果

---

## 11. 参考资源

### 官方资源
- **源代码**: `covasim/interventions.py`
- **示例文件**: `examples/t05_custom_intervention.py`
- **教程**: `docs/tutorials/tut_interventions.ipynb`

### 重要属性速查

#### Sim 对象
```python
sim.t                 # 当前时间步（整数）
sim.npts             # 总时间点数
sim.tvec             # 时间向量
sim['beta']          # 传播率
sim['beta_layer']    # 各层的传播率
sim.people           # 人群对象
sim.results          # 结果字典
sim.day(date)        # 日期 → 整数
sim.date(day)        # 整数 → 日期
```

#### People 对象（布尔数组）
```python
people.susceptible   # 易感者
people.exposed       # 暴露者
people.infectious    # 感染者
people.symptomatic   # 有症状者
people.diagnosed     # 已诊断者
people.recovered     # 康复者
people.dead          # 死亡者
people.quarantined   # 隔离者
people.vaccinated    # 已接种疫苗者
```

#### People 对象（其他属性）
```python
people.age           # 年龄（浮点数数组）
people.sex           # 性别（0/1 数组）
people.rel_sus       # 相对易感性
people.rel_trans     # 相对传播性
people.symp_prob     # 症状概率
people.contacts      # 接触网络（字典）
```

### 辅助函数
```python
cv.true(arr)                      # 返回 True 的索引
cv.binomial_filter(prob, inds)   # 二项分布过滤
cv.choose(n, inds)               # 随机选择 n 个
cv.n_binomial(prob, n)           # 二项分布抽样数量
```

---

**祝你使用 Covasim 顺利！** 🎉

如有问题，请查看:
- GitHub: https://github.com/institutefordiseasemodeling/covasim
- 文档: 查看 `docs/` 目录
- 示例: 查看 `examples/` 目录
