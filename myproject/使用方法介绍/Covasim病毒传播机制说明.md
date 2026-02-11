# Covasim 病毒传播机制详解

## 📋 概述

Covasim 使用**基于接触网络的概率传播模型**。传播发生在接触网络的边上，通过计算每条接触边的传播概率来决定是否发生感染。

---

## 🔄 传播流程（每日执行）

### 1. 模拟步骤 (`sim.py` 的 `step()` 方法)

每天模拟执行以下步骤：

```python
# 1. 更新人员状态
people.update_states_pre(t=t)

# 2. 更新接触网络（如果是动态网络）
contacts = people.update_contacts()

# 3. 计算病毒载量（随时间变化的传播能力）
viral_load = compute_viral_load(t, date_inf, date_rec, date_dead, ...)

# 4. 对每个病毒变体，遍历每个接触层
for variant in range(n_variants):
    for layer in contacts:
        # 计算传播概率
        # 执行传播
```

---

## 🧮 核心传播公式

### 传播概率计算公式

对于每条接触边（感染者 → 易感者），传播概率为：

```
P(传播) = β × β_layer × layer_beta × rel_trans × rel_sus × viral_load
```

其中：

- **`β`** (beta): 基础传播率（全局参数，默认 0.016）
- **`β_layer`**: 该接触层的传播权重（例如：家庭层=3.0，学校层=0.6）
- **`layer_beta`**: 每条接触边的权重（通常为 1.0，但可以自定义）
- **`rel_trans`**: 感染者的相对传播能力（考虑症状、隔离等因素）
- **`rel_sus`**: 易感者的相对易感性（考虑免疫力、隔离等因素）
- **`viral_load`**: 病毒载量（随时间变化的传播能力）

### 实际传播判断

```python
# 对于每条接触边，生成随机数判断是否传播
if random() < P(传播):
    发生感染
```

---

## 📊 详细传播计算过程

### 步骤 1: 计算相对传播能力 (`rel_trans`)

在 `utils.py` 的 `compute_trans_sus()` 函数中：

```python
# 无症状因子
f_asymp = symp + ~symp * asymp_factor
# 例如：有症状=1.0，无症状=0.8（如果 asymp_factor=0.8）

# 隔离因子
f_iso = ~iso + iso * iso_factor
# 例如：未隔离=1.0，隔离=0.2（如果 iso_factor=0.2）

# 检疫因子
f_quar = ~quar + quar * quar_factor
# 例如：未检疫=1.0，检疫=0.3（如果 quar_factor=0.3）

# 最终相对传播能力
rel_trans = rel_trans_base × inf × f_quar × f_asymp × f_iso × beta_layer × viral_load
```

**影响因素：**
- ✅ **有症状 vs 无症状**: 有症状者传播能力更强（默认无症状因子=1.0，即相同）
- ✅ **隔离状态**: 隔离者传播能力降低（`iso_factor`，默认约 0.2-0.3）
- ✅ **检疫状态**: 检疫者传播能力降低（`quar_factor`，默认约 0.2-0.3）
- ✅ **病毒载量**: 随时间变化（感染初期和高峰期载量高）
- ✅ **接触层类型**: 不同层有不同的传播权重（`beta_layer`）

### 步骤 2: 计算相对易感性 (`rel_sus`)

```python
# 最终相对易感性
rel_sus = rel_sus_base × sus × f_quar × (1 - immunity_factors)
```

**影响因素：**
- ✅ **易感状态**: 只有易感者才能被感染
- ✅ **检疫状态**: 检疫者易感性降低
- ✅ **免疫力**: 有免疫力的人易感性降低（`1 - immunity_factors`）
  - 自然感染后的免疫力
  - 疫苗接种后的免疫力
  - 交叉免疫力（对不同变体的保护）

### 步骤 3: 计算每条接触边的传播概率

在 `utils.py` 的 `compute_infections()` 函数中：

```python
# 对于每条接触边 (p1, p2)
betas = β × layer_betas × rel_trans[p1] × rel_sus[p2]

# 判断是否传播
if random() < betas:
    发生感染
```

**关键点：**
- 遍历所有接触边（`p1`, `p2`）
- 只考虑感染者（`rel_trans > 0`）和易感者（`rel_sus > 0`）之间的接触
- 每条边独立判断是否传播

### 步骤 4: 执行感染 (`people.infect()`)

如果传播发生，调用 `people.infect()` 方法：

```python
# 1. 更新状态
person.susceptible = False
person.exposed = True
person.date_exposed = t

# 2. 计算潜伏期（从暴露到具有传染性）
dur_exp2inf = sample(分布参数)
person.date_infectious = t + dur_exp2inf

# 3. 记录感染信息
infection_log.append({
    'source': 感染者ID,
    'target': 被感染者ID,
    'date': 感染日期,
    'layer': 接触层,
    'variant': 病毒变体
})
```

---

## 🔑 关键参数说明

### 1. Beta (β) - 基础传播率

- **位置**: `pars['beta']`
- **默认值**: 0.016
- **含义**: 每次接触的基础传播概率
- **调整**: 通过 `change_beta` 干预措施可以动态修改

### 2. Beta Layer (β_layer) - 层传播权重

- **位置**: `pars['beta_layer']`
- **默认值**（hybrid 类型）:
  - 家庭层 (h): 3.0
  - 学校层 (s): 0.6
  - 工作层 (w): 0.6
  - 社区层 (c): 0.3
- **含义**: 不同接触层的传播能力差异

### 3. Contacts - 接触网络

- **结构**: 每条边包含 `(p1, p2, beta)`
  - `p1`, `p2`: 接触的两个人
  - `beta`: 该边的权重（通常为 1.0）
- **类型**: 
  - 静态网络：接触关系固定
  - 动态网络：每天更新接触关系

### 4. Viral Load - 病毒载量

- **计算**: `compute_viral_load()` 函数
- **特点**: 随时间变化
  - 感染初期：载量低
  - 高峰期：载量高（约感染后 30% 时间点）
  - 恢复期：载量下降
- **参数**: `viral_dist` 字典
  - `frac_time`: 高峰期时间点（默认 0.3）
  - `load_ratio`: 高峰期载量倍数（默认 2.0）
  - `high_cap`: 最高载量上限（默认 4.0）

### 5. Asymptomatic Factor - 无症状因子

- **位置**: `pars['asymp_factor']`
- **默认值**: 1.0（无症状和有症状传播能力相同）
- **含义**: 无症状者的传播能力相对于有症状者的比例

### 6. Isolation/Quarantine Factors - 隔离/检疫因子

- **`iso_factor`**: 隔离者的传播能力降低（默认约 0.2-0.3）
- **`quar_factor`**: 检疫者的传播和易感性降低（默认约 0.2-0.3）

---

## 🌐 多接触层传播

Covasim 支持多个接触层（例如：家庭、学校、工作、社区），传播过程：

1. **遍历每个变体**（如果有多个病毒变体）
2. **遍历每个接触层**
3. **对每层独立计算传播**
4. **合并所有层的感染结果**

```python
for variant in variants:
    for layer_key, layer in contacts.items():
        # 计算该层的传播
        source_inds, target_inds = compute_infections(...)
        # 执行感染
        people.infect(target_inds, ...)
```

---

## 🛡️ 免疫力对传播的影响

### 自然感染免疫力

- 感染后产生中和抗体（NAbs）
- 免疫力随时间衰减（waning）
- 通过 `sus_imm` 数组计算易感性降低

### 疫苗接种免疫力

- 疫苗接种后产生免疫力
- 不同疫苗有不同的保护效果
- 可能发生突破性感染（breakthrough infection）

### 交叉免疫力

- 感染一种变体后，对其他变体也有部分保护
- 通过 `cross_immunity` 参数控制（默认 50%）

---

## 📈 传播示例

### 示例 1: 简单传播

假设：
- β = 0.016
- β_layer = 1.0（单一层）
- 感染者：有症状，未隔离
- 易感者：无免疫力，未检疫
- viral_load = 1.0

```
P(传播) = 0.016 × 1.0 × 1.0 × 1.0 × 1.0 × 1.0 = 0.016 (1.6%)
```

### 示例 2: 隔离降低传播

假设：
- 感染者被隔离（iso_factor = 0.2）

```
P(传播) = 0.016 × 1.0 × 0.2 × 1.0 × 1.0 × 1.0 = 0.0032 (0.32%)
```

传播概率降低 80%！

### 示例 3: 免疫力保护

假设：
- 易感者有 70% 的免疫力（immunity_factors = 0.7）

```
rel_sus = 1.0 × 1.0 × (1 - 0.7) = 0.3
P(传播) = 0.016 × 1.0 × 1.0 × 1.0 × 0.3 × 1.0 = 0.0048 (0.48%)
```

传播概率降低 70%！

---

## 🔍 代码位置

### 主要文件

1. **`covasim/sim.py`** (第 558-649 行)
   - `step()` 方法：主循环，调用传播计算

2. **`covasim/utils.py`** (第 82-128 行)
   - `compute_trans_sus()`: 计算相对传播能力和易感性
   - `compute_infections()`: 计算实际传播
   - `compute_viral_load()`: 计算病毒载量

3. **`covasim/people.py`** (第 435-516 行)
   - `infect()` 方法：执行感染，更新状态

### 关键函数调用链

```
sim.step()
  ├─ compute_viral_load()          # 计算病毒载量
  ├─ compute_trans_sus()           # 计算传播能力和易感性
  ├─ compute_infections()          # 计算传播概率并判断
  └─ people.infect()               # 执行感染
```

---

## 💡 关键理解点

1. **传播是概率性的**: 不是确定性的，每次接触都有一定概率传播
2. **基于接触网络**: 传播只发生在有接触关系的两个人之间
3. **多因素影响**: 传播概率受多种因素影响（症状、隔离、免疫力等）
4. **分层传播**: 不同接触层有不同的传播权重
5. **时间动态**: 病毒载量随时间变化，影响传播能力
6. **双向传播**: 接触边是双向的，两个人都可能感染对方

---

## 🎯 如何调整传播

### 1. 调整基础传播率

```python
pars = {'beta': 0.02}  # 增加传播率
```

### 2. 调整层传播权重

```python
pars = {
    'beta_layer': {
        'country': 0.5  # 降低该层的传播权重
    }
}
```

### 3. 使用干预措施

```python
# 降低传播率（例如：社交距离）
cb = cv.change_beta(days=30, changes=0.5)  # 降低 50%

# 隔离感染者
# 通过测试和接触者追踪自动实现
```

### 4. 调整无症状因子

```python
pars = {'asymp_factor': 0.8}  # 无症状者传播能力为有症状者的 80%
```

---

## 📚 相关参数总结

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `beta` | `pars['beta']` | 0.016 | 基础传播率 |
| `beta_layer` | `pars['beta_layer']` | 见上 | 层传播权重 |
| `asymp_factor` | `pars['asymp_factor']` | 1.0 | 无症状因子 |
| `iso_factor` | `pars['iso_factor']` | 0.2-0.3 | 隔离因子 |
| `quar_factor` | `pars['quar_factor']` | 0.2-0.3 | 检疫因子 |
| `viral_dist` | `pars['viral_dist']` | 见上 | 病毒载量分布 |
| `use_waning` | `pars['use_waning']` | True | 是否使用免疫力衰减 |

---

## 🔗 参考资料

- **源代码**: `covasim/sim.py`, `covasim/utils.py`, `covasim/people.py`
- **参数定义**: `covasim/parameters.py`
- **示例**: `examples/` 目录下的各种示例
