# Covasim 免疫机制详解

## 📖 目录
1. [免疫系统概述](#1-免疫系统概述)
2. [核心概念](#2-核心概念)
3. [变种病毒（Variant）](#3-变种病毒variant)
4. [中和抗体（NAb）机制](#4-中和抗体nab机制)
5. [免疫计算](#5-免疫计算)
6. [免疫衰减机制](#6-免疫衰减机制)
7. [完整示例](#7-完整示例)
8. [参数详解](#8-参数详解)
9. [常见问题](#9-常见问题)

---

## 1. 免疫系统概述

### Covasim 的免疫建模方法

Covasim 使用**中和抗体（Neutralizing Antibodies, NAb）水平**作为免疫力的代理指标。这是基于真实世界研究的科学方法，核心思想是：

```
中和抗体水平 (NAb) → 免疫保护力 (VE) → 降低感染/症状/重症风险
```

### 免疫的两个来源

Covasim 中的免疫来自两个基本来源：

1. **自然免疫**（Natural Immunity）
   - 来源：从感染中康复
   - 影响因素：
     - 感染的变种类型
     - 症状严重程度（无症状、轻症、重症）
     - 康复后的时间

2. **疫苗免疫**（Vaccine Immunity）
   - 来源：接种疫苗
   - 影响因素：
     - 疫苗类型（辉瑞、莫德纳、阿斯利康等）
     - 接种次数（初次接种、加强针）
     - 接种后的时间

### 免疫的三个保护维度

免疫保护在三个不同的"轴"上起作用：

1. **易感性保护** (`sus`): 降低被感染的概率
2. **症状保护** (`symp`): 感染后降低出现症状的概率
3. **重症保护** (`sev`): 出现症状后降低发展为重症的概率

---

## 2. 核心概念

### 2.1 中和抗体（NAb）

**什么是中和抗体？**
- 能够阻止病毒感染细胞的抗体
- Covasim 中用数值表示其水平（0 表示无抗体）
- NAb 水平越高，免疫保护越强

**NAb 的生命周期：**
```
感染/接种 → NAb 增长 → 峰值 (peak_nab) → 衰减 (waning) → 基线
```

### 2.2 交叉免疫（Cross-Immunity）

不同变种之间的免疫保护可能不同：

```python
# 示例：对 Alpha 变种的感染可能对 Delta 变种提供 80% 的保护
immunity_matrix[Delta][Alpha] = 0.8
```

### 2.3 免疫增强（Boosting）

当个体已有抗体时，新的免疫事件会"增强"现有免疫：

```
新的 peak_nab = 旧的 peak_nab × boost_factor
```

---

## 3. 变种病毒（Variant）

### 3.1 Variant 类

变种病毒是 Covasim 免疫系统的核心组件，每个变种有不同的传播和免疫特性。

```python
class variant(sc.prettyobj):
    '''
    添加新变种到模拟中
    
    Args:
        variant (str/dict): 变种名称或参数字典
        days (int/list): 引入变种的日期
        label (str): 变种标签
        n_imports (int): 输入病例数量
        rescale (bool): 是否按人口规模缩放
    '''
```

### 3.2 预定义变种

Covasim 内置了多个真实世界的变种：

- **alpha** (B.1.1.7): 英国变种
- **beta** (B.1.351): 南非变种
- **gamma** (P.1): 巴西变种
- **delta** (B.1.617.2): 印度变种
- **omicron** (B.1.1.529): 奥密克戎

### 3.3 使用预定义变种

```python
import covasim as cv

# 方式1: 使用预定义变种
alpha = cv.variant('alpha', days=10)  # 第10天引入Alpha变种
delta = cv.variant('delta', days=50, n_imports=5)  # 第50天引入5例Delta

# 运行模拟
sim = cv.Sim(variants=[alpha, delta])
sim.run()
```

### 3.4 自定义变种

```python
# 方式2: 自定义变种参数
my_variant = cv.variant(
    variant={
        'rel_beta': 2.0,        # 相对传播率（是原始毒株的2倍）
        'rel_symp_prob': 1.2,   # 相对症状概率
        'rel_severe_prob': 1.5, # 相对重症概率
        'rel_crit_prob': 1.3,   # 相对危重概率
        'rel_death_prob': 1.4,  # 相对死亡概率
    },
    label='My Variant',
    days=30,
    n_imports=10
)

sim = cv.Sim(variants=my_variant)
sim.run()
```

### 3.5 变种参数详解

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `rel_beta` | 相对传播率 | 1.0 | >0 |
| `rel_symp_prob` | 相对症状概率 | 1.0 | 0-∞ |
| `rel_severe_prob` | 相对重症概率 | 1.0 | 0-∞ |
| `rel_crit_prob` | 相对危重概率 | 1.0 | 0-∞ |
| `rel_death_prob` | 相对死亡概率 | 1.0 | 0-∞ |

### 3.6 变种的应用机制

```python
def apply(self, sim):
    '''在指定日期引入变种感染'''
    for ind in find_day(self.days, sim.t):
        # 找到易感者
        susceptible_inds = cv.true(sim.people.susceptible)
        
        # 考虑缩放因子
        rescale_factor = sim.rescale_vec[sim.t] if self.rescale else 1.0
        scaled_imports = self.n_imports / rescale_factor
        n_imports = sc.randround(scaled_imports)
        
        # 随机选择人进行感染
        importation_inds = np.random.choice(susceptible_inds, n_imports, replace=False)
        
        # 用新变种感染这些人
        sim.people.infect(inds=importation_inds, layer='importation', variant=self.index)
        
        # 记录导入病例
        sim.results['n_imports'][sim.t] += n_imports
```

---

## 4. 中和抗体（NAb）机制

### 4.1 更新峰值 NAb：`update_peak_nab()`

这是免疫系统的核心函数，在发生免疫事件（感染或接种疫苗）时被调用。

```python
def update_peak_nab(people, inds, nab_pars, symp=None):
    '''
    更新峰值NAb水平
    
    Args:
        people: 人群对象
        inds: 要更新的人的索引
        nab_pars: NAb参数（来自sim或疫苗参数）
        symp: 症状字典（仅用于自然感染）
              {'asymp': [...], 'mild': [...], 'sev': [...]}
    '''
```

#### 两种情况处理：

**情况1：已有抗体的人（增强效应）**
```python
# 如果个体已有NAb，则增强现有水平
if len(prior_nab_inds):
    boost_factor = nab_pars['nab_boost']  # 通常为 1.5-2.0
    people.peak_nab[prior_nab_inds] *= boost_factor
```

**情况2：首次获得抗体的人（初始化）**
```python
# 从分布中抽取初始NAb水平
if len(no_prior_nab_inds):
    # 1. 从对数正态分布中抽样
    init_nab = cv.sample(**nab_pars['nab_init'], size=len(no_prior_nab_inds))
    no_prior_nab = 2 ** init_nab  # 转换为实际NAb水平
    
    # 2. 如果是自然感染，根据症状严重程度调整
    if symp is not None:
        # 无症状：较低的NAb
        # 轻症：中等NAb
        # 重症：较高的NAb
        prior_symp[symp['asymp']] = pars['rel_imm_symp']['asymp']  # 例如 0.5
        prior_symp[symp['mild']] = pars['rel_imm_symp']['mild']    # 例如 1.0
        prior_symp[symp['sev']] = pars['rel_imm_symp']['severe']   # 例如 1.5
        
        # 归一化因子
        norm_factor = 1 + nab_pars['nab_eff']['alpha_inf_diff']
        no_prior_nab = no_prior_nab * prior_symp * norm_factor
    
    # 3. 更新peak_nab
    people.peak_nab[no_prior_nab_inds] = no_prior_nab
    
    # 4. 记录NAb事件时间
    people.t_nab_event[inds] = people.t
```

### 4.2 更新 NAb 水平：`update_nab()`

在每个时间步，NAb 水平会根据预计算的衰减曲线更新。

```python
def update_nab(people, inds):
    '''
    随时间推进NAb水平（应用衰减）
    '''
    # 计算距离上次NAb事件的时间
    t_since_boost = people.t - people.t_nab_event[inds]
    
    # 应用预计算的衰减动力学
    people.nab[inds] += people.pars['nab_kin'][t_since_boost] * people.peak_nab[inds]
    
    # 确保NAb不低于0
    people.nab[inds] = np.where(people.nab[inds] < 0, 0, people.nab[inds])
    
    # 确保NAb不超过peak_nab
    people.nab[inds] = np.where(
        people.nab[inds] > people.peak_nab[inds], 
        people.peak_nab[inds], 
        people.nab[inds]
    )
```

**可视化NAb动态：**
```
NAb水平
  ^
  |     峰值
  |    /\
  |   /  \___
  |  /       \___
  | /            \___
  |/                 \___
  +----------------------> 时间
  感染/接种   衰减期    基线
```

### 4.3 计算疫苗效力：`calc_VE()`

将 NAb 水平转换为实际的免疫保护因子。

```python
def calc_VE(nab, ax, pars):
    '''
    将NAb水平转换为免疫保护因子
    
    基于论文：https://doi.org/10.1101/2021.03.09.21252641
    
    Args:
        nab (array): 有效NAb水平数组
        ax (str): 保护轴 - 'sus'(易感性), 'symp'(症状), 'sev'(重症)
        pars (dict): 疫苗效力参数
    
    Returns:
        array: 免疫保护因子（0-1，1表示完全保护）
    '''
    
    # 根据保护轴选择参数
    if ax == 'sus':          # 对感染的保护
        alpha = pars['alpha_inf']
        beta = pars['beta_inf']
    elif ax == 'symp':       # 对症状的保护
        alpha = pars['alpha_symp_inf']
        beta = pars['beta_symp_inf']
    else:                    # 对重症的保护 (ax == 'sev')
        alpha = pars['alpha_sev_symp']
        beta = pars['beta_sev_symp']
    
    # 逻辑函数（Inverse logit）
    exp_lo = np.exp(alpha) * nab ** beta
    output = exp_lo / (1 + exp_lo)
    
    return output
```

**数学形式：**
```
VE(nab) = exp(α) × nab^β / (1 + exp(α) × nab^β)
```

这是一个S型曲线：
```
VE
1.0 |           ________
    |         /
0.5 |       /
    |     /
0.0 |____/
    +-------------------> NAb水平
    低    中    高
```

### 4.4 计算症状性疾病的疫苗效力：`calc_VE_symp()`

```python
def calc_VE_symp(nab, pars):
    '''
    计算针对症状性疾病的边际疫苗效力
    
    这考虑了两层保护：
    1. 防止感染
    2. 感染后防止出现症状
    '''
    
    # 对感染的保护
    exp_lo_inf = np.exp(pars['alpha_inf']) * nab ** pars['beta_inf']
    inv_lo_inf = exp_lo_inf / (1 + exp_lo_inf)
    
    # 对症状的保护（给定感染）
    exp_lo_symp_inf = np.exp(pars['alpha_symp_inf']) * nab ** pars['beta_symp_inf']
    inv_lo_symp_inf = exp_lo_symp_inf / (1 + exp_lo_symp_inf)
    
    # 组合效果
    VE_symp = 1 - ((1 - inv_lo_inf) * (1 - inv_lo_symp_inf))
    
    return VE_symp
```

---

## 5. 免疫计算

### 5.1 初始化免疫：`init_immunity()`

在模拟开始时设置免疫矩阵和衰减动力学。

```python
def init_immunity(sim, create=False):
    '''初始化包含所有变种的免疫矩阵'''
    
    # 如果不使用衰减，跳过
    if not sim['use_waning']:
        return
    
    nv = sim['n_variants']  # 变种数量
    
    # 创建交叉免疫矩阵 (nv × nv)
    immunity = np.ones((nv, nv), dtype=float)
    
    # 填充已知的交叉免疫值
    default_cross_immunity = cvpar.get_cross_immunity()
    for i in range(nv):
        label_i = sim['variant_map'][i]
        for j in range(nv):
            label_j = sim['variant_map'][j]
            if label_i in default_cross_immunity and label_j in default_cross_immunity:
                immunity[j][i] = default_cross_immunity[label_j][label_i]
    
    sim['immunity'] = immunity
    
    # 预计算NAb衰减动力学
    sim['nab_kin'] = precompute_waning(length=sim.npts, pars=sim['nab_decay'])
```

**交叉免疫矩阵示例：**
```
           感染变种
         Wild  Alpha  Delta  Omicron
康  Wild  [1.0   0.9   0.8    0.5  ]
复  Alpha [0.85  1.0   0.85   0.6  ]
变  Delta [0.7   0.75  1.0    0.65 ]
种  Omi   [0.5   0.55  0.6    1.0  ]
```

### 5.2 检查免疫：`check_immunity()`

在每个时间步计算每个人对每个变种的免疫力。

```python
def check_immunity(people, variants=None):
    '''
    计算当前时间步的免疫力
    
    考虑两个来源：
    1. 自然免疫（既往感染）
    2. 疫苗免疫（接种疫苗）
    '''
    
    pars = people.pars
    nab_eff = pars['nab_eff']
    
    if variants is None:
        variants = range(pars['n_variants'])
    
    # 对每个变种更新免疫
    for variant in variants:
        natural_imm = np.zeros(len(people))
        vaccine_imm = np.zeros(len(people))
        
        # === 1. 自然免疫权重 ===
        # 找到已康复的人
        was_inf = cv.true(people.t >= people.date_recovered)
        
        if len(was_inf):
            # 获取他们感染的变种
            recovered_variant = people.recovered_variant[was_inf]
            
            # 从交叉免疫矩阵获取保护因子
            immunity = pars['immunity'][variant, :]
            natural_imm[was_inf] = immunity[recovered_variant.astype(int)]
        
        # === 2. 疫苗免疫权重 ===
        # 找到已接种的人
        is_vacc = cv.true(people.vaccinated)
        
        if len(is_vacc) and len(pars['vaccine_pars']):
            # 获取疫苗类型
            vacc_source = people.vaccine_source[is_vacc]
            
            # 从疫苗参数获取对当前变种的保护
            vx_pars = pars['vaccine_pars']
            vx_map = pars['vaccine_map']
            var_key = pars['variant_map'][variant]
            
            # 构建疫苗免疫数组
            imm_arr = np.zeros(max(vx_map.keys()) + 1)
            for num, key in vx_map.items():
                imm_arr[num] = vx_pars[key][var_key]
            
            vaccine_imm[is_vacc] = imm_arr[vacc_source]
        
        # === 3. 计算总体免疫 ===
        # 取自然免疫和疫苗免疫的较大值
        imm = np.maximum(natural_imm, vaccine_imm)
        
        # 计算有效NAb（考虑交叉免疫）
        effective_nabs = people.nab * imm
        
        # 将有效NAb转换为三个轴的保护力
        people.sus_imm[variant, :]  = calc_VE(effective_nabs, 'sus', nab_eff)
        people.symp_imm[variant, :] = calc_VE(effective_nabs, 'symp', nab_eff)
        people.sev_imm[variant, :]  = calc_VE(effective_nabs, 'sev', nab_eff)
```

**免疫计算流程图：**
```
┌─────────────┐        ┌──────────────┐
│ 自然免疫    │        │  疫苗免疫     │
│ (既往感染)  │        │  (接种疫苗)   │
└──────┬──────┘        └──────┬───────┘
       │                      │
       │                      │
       └──────────┬───────────┘
                  │ max()
                  ▼
           ┌─────────────┐
           │ 交叉免疫因子 │
           └──────┬──────┘
                  │ × NAb
                  ▼
          ┌──────────────┐
          │  有效 NAb     │
          └──────┬───────┘
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
    sus_imm  symp_imm  sev_imm
```

---

## 6. 免疫衰减机制

### 6.1 衰减函数概述

Covasim 提供多种衰减函数来模拟 NAb 随时间的变化。

```python
def precompute_waning(length, pars=None):
    '''
    预计算衰减曲线
    
    支持的函数形式：
    1. 'nab_growth_decay': NAb增长和衰减（默认）
    2. 'nab_decay': NAb衰减
    3. 'exp_decay': 指数衰减
    4. 自定义函数
    '''
    
    form = pars.pop('form')
    
    if form is None or form == 'nab_growth_decay':
        output = nab_growth_decay(length, **pars)
    elif form == 'nab_decay':
        output = nab_decay(length, **pars)
    elif form == 'exp_decay':
        output = exp_decay(length, **pars)
    elif callable(form):
        output = form(length, **pars)  # 自定义函数
    
    return output
```

### 6.2 NAb 增长和衰减：`nab_growth_decay()`

**最复杂和真实的模型**，基于 Khoury et al. 的研究。

```python
def nab_growth_decay(length, growth_time, decay_rate1, decay_time1, 
                     decay_rate2, decay_time2):
    '''
    NAb 增长/衰减函数
    
    使用：
    - 线性增长期
    - 指数衰减期（早期快速衰减）
    - 指数衰减期（后期慢速衰减，半衰期约10年）
    
    Args:
        length: 时间点数量
        growth_time: NAb增长时间（天）
        decay_rate1: 初始衰减率
        decay_time1: 第一衰减期持续时间
        decay_rate2: 后期衰减率
        decay_time2: 转换到后期衰减的时间
    '''
```

**可视化：**
```
NAb水平
  ^
  |      峰值
  |     /|
  |    / |
  |   /  |\
  |  /   | \___
  | /    |     \___
  |/     |         \___________
  +------+------+--------------> 时间
  0    growth decay1  decay2
       time   time1   time2
  
  阶段：
  1. 线性增长 (0 → growth_time)
  2. 快速指数衰减 (growth_time → decay_time1)
  3. 衰减率逐渐降低 (decay_time1 → decay_time2)
  4. 慢速指数衰减 (decay_time2 → ∞)
```

**示例参数：**
```python
nab_decay_params = {
    'form': 'nab_growth_decay',
    'growth_time': 21,      # 21天达到峰值
    'decay_rate1': 0.006,   # 初始快速衰减
    'decay_time1': 100,     # 100天后开始减慢
    'decay_rate2': 0.0005,  # 后期慢速衰减（~10年半衰期）
    'decay_time2': 300,     # 300天达到慢速衰减
}
```

### 6.3 NAb 衰减：`nab_decay()`

**简化版本**，仅包含衰减（不包含增长期）。

```python
def nab_decay(length, decay_rate1, decay_time1, decay_rate2):
    '''
    NAb衰减函数
    
    使用指数衰减，衰减率本身也会衰减
    
    Args:
        length: 时间点数量
        decay_rate1: 初始衰减率
        decay_time1: 第一衰减期持续时间（通常250天）
        decay_rate2: 衰减率的衰减率
    '''
    
    def f1(t, decay_rate1):
        '''简单指数衰减'''
        return np.exp(-t * decay_rate1)
    
    def f2(t, decay_rate1, decay_time1, decay_rate2):
        '''复杂指数衰减（衰减率也在衰减）'''
        return np.exp(-t * (decay_rate1 * np.exp(-(t - decay_time1) * decay_rate2)))
    
    # 前期：简单指数衰减
    # 后期：衰减率逐渐降低
```

**可视化：**
```
NAb水平
  ^
1.0|
   |\
   | \
   |  \___
   |      \___
   |          \________
   +--------------------> 时间
   0      decay_time1
```

### 6.4 指数衰减：`exp_decay()`

**最简单的模型**，使用标准的指数衰减。

```python
def exp_decay(length, init_val, half_life, delay=None):
    '''
    标准指数衰减
    
    Args:
        length: 时间点数量
        init_val: 初始值
        half_life: 半衰期（天）
        delay: 可选的延迟期（在此期间线性增长）
    '''
    
    decay_rate = np.log(2) / half_life if ~np.isnan(half_life) else 0.
    
    if delay is not None:
        # 先线性增长，后指数衰减
        growth = linear_growth(delay, init_val / delay)
        decay = init_val * np.exp(-decay_rate * t)
        result = np.concatenate([growth, decay])
    else:
        # 直接指数衰减
        result = init_val * np.exp(-decay_rate * t)
    
    return np.diff(result)
```

**数学形式：**
```
NAb(t) = init_val × e^(-λt)

其中: λ = ln(2) / half_life
```

**可视化：**
```
NAb水平
  ^
  |
  |\
  | \
  |  \
  |   \
  |    \
  |     \_____
  +------------> 时间
  0   t₁/₂  2t₁/₂
  
每经过一个半衰期，NAb水平减半
```

### 6.5 线性衰减：`linear_decay()`

```python
def linear_decay(length, init_val, slope):
    '''
    线性衰减
    
    Args:
        length: 时间点数量
        init_val: 初始值
        slope: 衰减斜率（负值）
    '''
    result = -slope * np.ones(length)
    result[0] = init_val
    return result
```

### 6.6 衰减函数对比

| 函数 | 复杂度 | 真实性 | 参数数量 | 适用场景 |
|------|--------|--------|----------|----------|
| `nab_growth_decay` | 高 | 最高 | 5 | 详细研究，真实世界场景 |
| `nab_decay` | 中 | 高 | 3 | 一般研究，合理准确性 |
| `exp_decay` | 低 | 中 | 2-3 | 快速测试，简单场景 |
| `linear_decay` | 低 | 低 | 2 | 特殊场景，教学用途 |

---

## 7. 完整示例

### 示例 1: 基本的免疫和变种使用

```python
import covasim as cv
import numpy as np
import matplotlib.pyplot as plt

# 基本参数
pars = {
    'pop_size': 50000,
    'pop_infected': 100,
    'n_days': 365,
    'use_waning': True,  # 启用免疫衰减
    'verbose': 0.1,
}

# 添加变种
variants = [
    cv.variant('wild', days=0, n_imports=0),      # 野生型（默认）
    cv.variant('alpha', days=100, n_imports=10),  # Alpha在第100天引入
    cv.variant('delta', days=200, n_imports=10),  # Delta在第200天引入
]

# 添加疫苗
vaccine = cv.vaccinate_prob(
    vaccine='pfizer',
    days=150,
    prob=0.005,  # 每天0.5%的人接种
)

# 创建并运行模拟
sim = cv.Sim(pars=pars, variants=variants, interventions=vaccine)
sim.run()

# 绘图
sim.plot()

# 查看变种分布
print(f"\n变种分布:")
for i, label in sim['variant_map'].items():
    print(f"  {label}: {sim.results['variant'][label][-1]:.0f} 例")
```

### 示例 2: 比较不同衰减函数

```python
import covasim as cv
import numpy as np
import matplotlib.pyplot as plt

def run_with_waning(waning_pars, label):
    '''使用特定衰减参数运行模拟'''
    pars = {
        'pop_size': 10000,
        'pop_infected': 50,
        'n_days': 365,
        'use_waning': True,
        'nab_decay': waning_pars,
        'verbose': 0,
    }
    
    # 添加疫苗（第50天开始）
    vaccine = cv.vaccinate_prob('pfizer', days=50, prob=0.01)
    
    sim = cv.Sim(pars=pars, interventions=vaccine, label=label)
    sim.run()
    return sim

# 定义不同的衰减参数
waning_scenarios = {
    'Fast decay': {
        'form': 'exp_decay',
        'init_val': 1.0,
        'half_life': 90,  # 90天半衰期（快速衰减）
    },
    'Medium decay': {
        'form': 'exp_decay',
        'init_val': 1.0,
        'half_life': 180,  # 180天半衰期（中等衰减）
    },
    'Slow decay': {
        'form': 'exp_decay',
        'init_val': 1.0,
        'half_life': 365,  # 365天半衰期（慢速衰减）
    },
    'Realistic (nab_growth_decay)': {
        'form': 'nab_growth_decay',
        'growth_time': 21,
        'decay_rate1': 0.006,
        'decay_time1': 100,
        'decay_rate2': 0.0005,
        'decay_time2': 300,
    }
}

# 运行所有场景
sims = []
for label, waning_pars in waning_scenarios.items():
    sim = run_with_waning(waning_pars, label)
    sims.append(sim)

# 绘制对比图
msim = cv.MultiSim(sims)
msim.plot(to_plot=['cum_infections', 'new_infections'])
plt.show()
```

### 示例 3: 探索交叉免疫

```python
import covasim as cv
import numpy as np

def test_cross_immunity(cross_imm_factor):
    '''
    测试不同交叉免疫水平的影响
    
    Args:
        cross_imm_factor: Alpha对Delta的交叉免疫因子 (0-1)
    '''
    
    pars = {
        'pop_size': 20000,
        'pop_infected': 100,
        'n_days': 300,
        'use_waning': True,
        'verbose': 0,
    }
    
    # 设置变种
    variants = [
        cv.variant('wild', days=0, n_imports=0),
        cv.variant('alpha', days=50, n_imports=50),  # Alpha在第50天引入
        cv.variant('delta', days=150, n_imports=20),  # Delta在第150天引入
    ]
    
    sim = cv.Sim(pars=pars, variants=variants)
    sim.initialize()
    
    # 修改交叉免疫矩阵
    # immunity[感染变种][康复变种] = 保护因子
    alpha_idx = list(sim['variant_map'].values()).index('alpha')
    delta_idx = list(sim['variant_map'].values()).index('delta')
    sim['immunity'][delta_idx][alpha_idx] = cross_imm_factor
    
    sim.run()
    return sim

# 测试不同的交叉免疫水平
cross_imm_levels = [0.3, 0.5, 0.7, 0.9]
sims = []

for level in cross_imm_levels:
    sim = test_cross_immunity(level)
    sim.label = f'Cross-immunity: {level:.1f}'
    sims.append(sim)

# 比较结果
msim = cv.MultiSim(sims)
msim.plot()

# 打印Delta波的大小
print("\nDelta波的感染总数:")
for sim in sims:
    delta_infections = sim.results['variant']['delta'][-1]
    print(f"  {sim.label}: {delta_infections:.0f}")
```

### 示例 4: 自定义免疫追踪分析器

```python
import covasim as cv
import numpy as np
import matplotlib.pyplot as plt

class immunity_tracker(cv.Analyzer):
    '''
    追踪人群中的NAb水平和免疫保护
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
    
    def initialize(self, sim):
        super().initialize()
        self.nab_levels = []
        self.sus_imm_levels = []
        self.n_vaccinated = []
        self.n_recovered = []
        return
    
    def apply(self, sim):
        # 记录NAb水平分布
        self.nab_levels.append(sim.people.nab.copy())
        
        # 记录易感性免疫（对变种0）
        self.sus_imm_levels.append(sim.people.sus_imm[0, :].copy())
        
        # 记录接种和康复人数
        self.n_vaccinated.append(sim.people.vaccinated.sum())
        self.n_recovered.append((sim.people.t >= sim.people.date_recovered).sum())
        
        return
    
    def plot(self):
        '''绘制免疫追踪结果'''
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 转换为数组
        nab_array = np.array(self.nab_levels)
        sus_imm_array = np.array(self.sus_imm_levels)
        
        # 1. NAb水平的百分位数
        ax = axes[0, 0]
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            values = np.percentile(nab_array, p, axis=1)
            ax.plot(values, label=f'{p}th percentile')
        ax.set_xlabel('Day')
        ax.set_ylabel('NAb level')
        ax.set_title('Distribution of NAb levels over time')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. 易感性免疫的百分位数
        ax = axes[0, 1]
        for p in percentiles:
            values = np.percentile(sus_imm_array, p, axis=1)
            ax.plot(values, label=f'{p}th percentile')
        ax.set_xlabel('Day')
        ax.set_ylabel('Susceptibility immunity')
        ax.set_title('Distribution of susceptibility immunity')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. 接种和康复人数
        ax = axes[1, 0]
        ax.plot(self.n_vaccinated, label='Vaccinated')
        ax.plot(self.n_recovered, label='Recovered')
        ax.set_xlabel('Day')
        ax.set_ylabel('Number of people')
        ax.set_title('Vaccinated and recovered population')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. NAb水平的热图
        ax = axes[1, 1]
        im = ax.imshow(nab_array.T[:1000, :], aspect='auto', cmap='viridis')
        ax.set_xlabel('Day')
        ax.set_ylabel('Individual (first 1000)')
        ax.set_title('NAb levels heatmap')
        plt.colorbar(im, ax=ax, label='NAb level')
        
        plt.tight_layout()
        return fig

# 使用自定义分析器
pars = {
    'pop_size': 10000,
    'pop_infected': 50,
    'n_days': 365,
    'use_waning': True,
}

vaccine = cv.vaccinate_prob('pfizer', days=100, prob=0.01)
tracker = immunity_tracker()

sim = cv.Sim(pars=pars, interventions=vaccine, analyzers=tracker)
sim.run()

# 绘制结果
sim.plot()
tracker.plot()
plt.show()
```

---

## 8. 参数详解

### 8.1 NAb 参数

#### `nab_init` - 初始NAb分布
```python
'nab_init': {
    'dist': 'normal',      # 分布类型：'normal', 'lognormal'
    'par1': 0,             # 均值（对数空间）
    'par2': 2,             # 标准差（对数空间）
}
```

#### `nab_boost` - NAb增强因子
```python
'nab_boost': 1.5  # 再次暴露时，NAb增加50%
```

#### `nab_eff` - NAb效力参数
```python
'nab_eff': {
    # 对感染的保护
    'alpha_inf': -3.0,
    'beta_inf': 2.5,
    'alpha_inf_diff': 1.0,  # 自然感染的归一化因子
    
    # 对症状的保护（给定感染）
    'alpha_symp_inf': -1.5,
    'beta_symp_inf': 2.0,
    
    # 对重症的保护（给定症状）
    'alpha_sev_symp': -2.0,
    'beta_sev_symp': 2.5,
}
```

#### `nab_decay` - NAb衰减参数
```python
# 选项1: nab_growth_decay（推荐）
'nab_decay': {
    'form': 'nab_growth_decay',
    'growth_time': 21,      # 增长期（天）
    'decay_rate1': 0.006,   # 初始衰减率
    'decay_time1': 100,     # 第一衰减期
    'decay_rate2': 0.0005,  # 后期衰减率
    'decay_time2': 300,     # 后期衰减开始
}

# 选项2: exp_decay（简单）
'nab_decay': {
    'form': 'exp_decay',
    'init_val': 1.0,        # 初始值
    'half_life': 180,       # 半衰期（天）
    'delay': 21,            # 可选：延迟期
}
```

### 8.2 症状严重程度的免疫缩放

```python
'rel_imm_symp': {
    'asymp': 0.2,    # 无症状者的NAb为有症状者的20%
    'mild': 0.5,     # 轻症者的NAb为重症者的50%
    'severe': 1.0,   # 重症者（基线）
}
```

### 8.3 变种参数

```python
variant_pars = {
    'rel_beta': 1.5,         # 传播率相对于野生型
    'rel_symp_prob': 1.2,    # 症状概率
    'rel_severe_prob': 1.3,  # 重症概率
    'rel_crit_prob': 1.2,    # 危重概率
    'rel_death_prob': 1.1,   # 死亡概率
}
```

### 8.4 疫苗参数

```python
vaccine_pars = {
    'pfizer': {
        'nab_eff': {...},         # NAb效力参数
        'nab_init': {...},        # 初始NAb分布
        'nab_boost': 2.0,         # 加强针增强因子
        'doses': 2,               # 剂数
        'interval': 21,           # 两剂间隔（天）
        
        # 对不同变种的交叉保护
        'wild': 1.0,     # 对野生型100%
        'alpha': 0.95,   # 对Alpha 95%
        'delta': 0.85,   # 对Delta 85%
        'omicron': 0.4,  # 对Omicron 40%
    }
}
```

---

## 9. 常见问题

### Q1: 如何启用/禁用免疫衰减？

```python
# 启用免疫衰减（推荐用于真实场景）
sim = cv.Sim(use_waning=True)

# 禁用免疫衰减（简化模型）
sim = cv.Sim(use_waning=False)
```

### Q2: 如何查看个体的NAb水平？

```python
# 运行模拟后
sim.run()

# 查看前10个人的NAb水平
print(sim.people.nab[:10])

# 查看峰值NAb
print(sim.people.peak_nab[:10])

# 查看上次NAb事件的时间
print(sim.people.t_nab_event[:10])
```

### Q3: 如何自定义衰减函数？

```python
def my_custom_decay(length, my_param1, my_param2):
    '''自定义衰减函数'''
    t = np.arange(length)
    # 实现你的衰减逻辑
    decay = my_param1 * np.exp(-my_param2 * t)
    return np.diff(decay)

# 使用自定义函数
pars = {
    'use_waning': True,
    'nab_decay': {
        'form': my_custom_decay,
        'my_param1': 1.0,
        'my_param2': 0.01,
    }
}

sim = cv.Sim(pars=pars)
```

### Q4: 如何设置变种之间的交叉免疫？

```python
sim = cv.Sim()
sim.initialize()

# 手动设置交叉免疫矩阵
# immunity[感染变种][康复变种] = 保护因子
variant_map = sim['variant_map']
wild_idx = list(variant_map.values()).index('wild')
alpha_idx = list(variant_map.values()).index('alpha')

# 设置Alpha感染对野生型的保护为90%
sim['immunity'][wild_idx][alpha_idx] = 0.9

# 设置野生型感染对Alpha的保护为80%
sim['immunity'][alpha_idx][wild_idx] = 0.8

sim.run()
```

### Q5: 自然免疫和疫苗免疫如何组合？

```python
# Covasim 使用 max() 函数组合两种免疫
# 即取较强的免疫保护

# 伪代码：
natural_imm = immunity_from_infection[variant]
vaccine_imm = immunity_from_vaccine[variant]
total_imm = max(natural_imm, vaccine_imm)
effective_nab = nab * total_imm
protection = calc_VE(effective_nab, axis)
```

### Q6: 如何模拟混合感染（既往感染+疫苗）？

```python
import covasim as cv

pars = {
    'pop_size': 10000,
    'pop_infected': 100,  # 初始感染
    'n_days': 300,
    'use_waning': True,
}

# 第100天开始疫苗接种
vaccine = cv.vaccinate_prob('pfizer', days=100, prob=0.01)

sim = cv.Sim(pars=pars, interventions=vaccine)
sim.run()

# 分析混合免疫
people = sim.people
has_natural = (people.t >= people.date_recovered)
has_vaccine = people.vaccinated
hybrid = has_natural & has_vaccine

print(f"仅自然免疫: {(has_natural & ~has_vaccine).sum()}")
print(f"仅疫苗免疫: {(~has_natural & has_vaccine).sum()}")
print(f"混合免疫: {hybrid.sum()}")
print(f"无免疫: {(~has_natural & ~has_vaccine).sum()}")
```

### Q7: 如何绘制NAb水平随时间的变化？

```python
import covasim as cv
import matplotlib.pyplot as plt
import numpy as np

# 使用追踪器记录NAb水平
class nab_tracker(cv.Analyzer):
    def initialize(self, sim):
        super().initialize()
        self.timepoints = []
        self.mean_nab = []
        self.median_nab = []
        
    def apply(self, sim):
        self.timepoints.append(sim.t)
        nab = sim.people.nab[sim.people.nab > 0]  # 只考虑有NAb的人
        if len(nab) > 0:
            self.mean_nab.append(nab.mean())
            self.median_nab.append(np.median(nab))
        else:
            self.mean_nab.append(0)
            self.median_nab.append(0)

tracker = nab_tracker()
sim = cv.Sim(analyzers=tracker)
sim.run()

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(tracker.timepoints, tracker.mean_nab, label='Mean NAb')
plt.plot(tracker.timepoints, tracker.median_nab, label='Median NAb')
plt.xlabel('Day')
plt.ylabel('NAb level')
plt.title('NAb levels over time')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## 10. 最佳实践

### ✅ 推荐做法

1. **使用 `use_waning=True` 进行真实场景模拟**
   ```python
   sim = cv.Sim(use_waning=True)  # 启用免疫衰减
   ```

2. **使用 `nab_growth_decay` 获得最真实的结果**
   ```python
   pars = {'nab_decay': {'form': 'nab_growth_decay', ...}}
   ```

3. **根据研究目标选择合适的衰减函数**
   - 详细研究 → `nab_growth_decay`
   - 快速测试 → `exp_decay`
   - 敏感性分析 → 比较多种函数

4. **记录和分析NAb动态**
   ```python
   # 使用分析器追踪NAb水平
   tracker = immunity_tracker()
   sim = cv.Sim(analyzers=tracker)
   ```

5. **考虑变种和交叉免疫**
   ```python
   variants = [
       cv.variant('wild', days=0),
       cv.variant('delta', days=100),
   ]
   ```

### ❌ 避免的做法

1. **不要在不需要时使用免疫衰减**
   ```python
   # 如果只是简单测试，可以关闭
   sim = cv.Sim(use_waning=False)
   ```

2. **不要忽略疫苗特异性保护**
   ```python
   # ❌ 错误：所有变种使用相同的疫苗效力
   # ✅ 正确：为每个变种指定不同的保护水平
   ```

3. **不要假设线性免疫**
   - 免疫保护是非线性的（S型曲线）
   - 使用 `calc_VE()` 正确转换NAb到保护力

---

## 11. 总结

### 核心要点

1. **NAb 作为免疫代理**
   - Covasim 使用中和抗体水平来量化免疫
   - NAb → 免疫保护 通过逻辑函数转换

2. **两个免疫来源**
   - 自然免疫：从感染中获得
   - 疫苗免疫：从接种中获得
   - 取两者中较强的

3. **三个保护维度**
   - 易感性保护
   - 症状保护
   - 重症保护

4. **免疫衰减**
   - NAb 水平随时间衰减
   - 多种衰减模型可选
   - 可自定义衰减函数

5. **变种和交叉免疫**
   - 不同变种有不同特性
   - 交叉免疫矩阵控制变种间保护
   - 可模拟免疫逃逸

### 关键函数

| 函数 | 用途 |
|------|------|
| `update_peak_nab()` | 更新峰值NAb（感染/接种时） |
| `update_nab()` | 应用衰减，更新当前NAb |
| `calc_VE()` | NAb → 免疫保护因子 |
| `check_immunity()` | 计算总体免疫（每个时间步） |
| `init_immunity()` | 初始化免疫矩阵和衰减 |
| `precompute_waning()` | 预计算衰减曲线 |

### 下一步

- 阅读源代码 `covasim/immunity.py`
- 查看示例 `examples/t08_*.py`
- 实验不同的衰减参数
- 尝试创建自定义变种和疫苗

---

## 12. 参考资源

### 科学文献

1. **NAb与免疫保护关系**:
   - Khoury et al. (2021): https://www.nature.com/articles/s41591-021-01377-8
   - "Neutralizing antibody levels are highly predictive of immune protection"

2. **NAb动力学**:
   - https://doi.org/10.1101/2021.03.09.21252641
   - 描述NAb如何随时间增长和衰减

### Covasim 资源

- **源代码**: `covasim/immunity.py`
- **示例**:
  - `examples/t08_waning_immunity.py`
  - `examples/t08_variants.py`
  - `examples/t08_boosters.py`
- **文档**: `docs/tutorials/`

### 关键参数位置

- NAb 参数: `covasim/parameters.py` 中的 `get_nab_pars()`
- 变种参数: `covasim/parameters.py` 中的 `get_variant_pars()`
- 交叉免疫: `covasim/parameters.py` 中的 `get_cross_immunity()`

---

**祝你使用 Covasim 的免疫系统顺利！** 🧬💉

如有疑问，欢迎查阅源代码或官方文档。
