# Covasim 模型接口和参数说明

## 主要接口

Covasim 的主要接口是 **`cv.Sim()`** 类，定义在 `covasim/sim.py` 文件中。

### 基本用法

```python
import covasim as cv

# 最简单的用法（使用默认参数）
sim = cv.Sim()
sim.run()
sim.plot()

# 自定义参数
sim = cv.Sim(pop_size=10000, n_days=90, pop_type='hybrid')
sim.run()
sim.plot()
```

### Sim 类的构造函数参数

```python
cv.Sim(
    pars=None,      # 参数字典，用于修改默认值
    datafile=None,  # 数据文件路径（Excel, CSV）或 pandas DataFrame
    label=None,     # 模拟的名称（用于区分批量运行）
    simfile=None,   # 保存模拟的文件名
    popfile=None,   # 如果提供，从此文件加载人口
    people=None,    # 如果提供，使用预生成的人口对象
    version=None,   # 如果提供，使用指定版本的默认参数
    **kwargs        # 额外的参数，会传递给 cv.make_pars()
)
```

## 主要参数说明

所有参数定义在 `covasim/parameters.py` 的 `make_pars()` 函数中（第15-151行）。

### 1. 人口参数（Population parameters）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pop_size` | 20000 | 代理数量（即易感人群数量） |
| `pop_infected` | 20 | 初始感染人数 |
| `pop_type` | 'random' | 人口类型：'random'（最快）、'synthpops'（最真实）、'hybrid'（折中） |
| `location` | None | 加载数据的位置（默认西雅图） |

### 2. 模拟参数（Simulation parameters）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `start_day` | '2020-03-01' | 模拟开始日期 |
| `end_day` | None | 模拟结束日期 |
| `n_days` | 60 | 运行天数（如果未指定 end_day） |
| `rand_seed` | 1 | 随机种子（None 表示不重置） |
| `verbose` | 0.1 | 显示信息级别：0（静默）、0.1（部分）、1（更多）、2（全部） |

### 3. 缩放参数（Rescaling parameters）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pop_scale` | 1 | 人口缩放因子 |
| `scaled_pop` | None | 总缩放人口数 |
| `rescale` | True | 是否启用动态缩放 |
| `rescale_threshold` | 0.05 | 触发缩放的易感人群比例 |
| `rescale_factor` | 1.2 | 每次缩放的因子 |
| `frac_susceptible` | 1.0 | 易感人群比例 |

### 4. 网络参数（Network parameters）

这些参数在人口创建后初始化，由 `reset_layer_pars()` 函数设置。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `contacts` | None | 每层的接触数（字典，按层设置） |
| `dynam_layer` | None | 哪些层是动态的（字典） |
| `beta_layer` | None | 每层的传播率（字典） |

**对于 'random' 类型**：
- `contacts = {'a': 20}` （所有接触）
- `beta_layer = {'a': 1.0}`

**对于 'hybrid' 类型**：
- `contacts = {'h': 2.0, 's': 20, 'w': 16, 'c': 20}` 
  - h = 家庭（household）
  - s = 学校（school）
  - w = 工作场所（workplace）
  - c = 社区（community）
- `beta_layer = {'h': 3.0, 's': 0.6, 'w': 0.6, 'c': 0.3}`

### 5. 疾病传播参数（Disease transmission parameters）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beta` | 0.016 | 每个有症状接触的传播率 |
| `beta_dist` | dict(...) | 个体传播率分布 |
| `viral_dist` | dict(...) | 随时间变化的病毒载量 |
| `asymp_factor` | 1.0 | 无症状病例的传播率倍数 |
| `n_imports` | 0 | 平均每日输入病例数（泊松分布） |
| `n_variants` | 1 | 循环变体数量 |

### 6. 疾病持续时间参数（Duration parameters）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dur['exp2inf']` | lognormal(4.5, 1.5) | 暴露到感染的时间 |
| `dur['inf2sym']` | lognormal(1.1, 0.9) | 感染到有症状的时间 |
| `dur['sym2sev']` | lognormal(6.6, 4.9) | 有症状到严重的时间 |
| `dur['sev2crit']` | lognormal(1.5, 2.0) | 严重到危重的时间 |
| `dur['asym2rec']` | lognormal(8.0, 2.0) | 无症状恢复时间 |
| `dur['mild2rec']` | lognormal(8.0, 2.0) | 轻症恢复时间 |
| `dur['sev2rec']` | lognormal(18.1, 6.3) | 重症恢复时间 |
| `dur['crit2rec']` | lognormal(18.1, 6.3) | 危重恢复时间 |
| `dur['crit2die']` | lognormal(10.7, 4.8) | 危重到死亡的时间 |

### 7. 严重程度参数（Severity parameters）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rel_symp_prob` | 1.0 | 有症状病例比例的缩放因子 |
| `rel_severe_prob` | 1.0 | 严重病例比例的缩放因子 |
| `rel_crit_prob` | 1.0 | 危重病例比例的缩放因子 |
| `rel_death_prob` | 1.0 | 死亡比例的缩放因子 |
| `prog_by_age` | True | 是否基于年龄设置疾病进展 |
| `prognoses` | None | 按年龄的预后数组（初始化时填充） |

### 8. 保护措施参数（Protection measures）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iso_factor` | None | 隔离病例的传播率倍数（按层设置） |
| `quar_factor` | None | 隔离的传播率和易感性倍数（按层设置） |
| `quar_period` | 14 | 隔离天数 |

### 9. 免疫参数（Immunity parameters）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_waning` | True | 是否使用动态计算的免疫 |
| `nab_init` | dict(...) | 初始中和抗体水平分布 |
| `nab_decay` | dict(...) | 中和抗体衰减动力学 |
| `nab_boost` | 1.5 | 再感染时的抗体增强因子 |
| `nab_eff` | dict(...) | 将抗体映射到有效性的参数 |
| `rel_imm_symp` | dict(...) | 自然感染的相对免疫（按症状） |
| `immunity` | None | 免疫和交叉免疫因子矩阵 |
| `trans_redux` | 0.59 | 突破性感染的传播减少 |

### 10. 变体参数（Variant parameters）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rel_beta` | 1.0 | 按变体的相对传播率 |
| `variants` | [] | 额外的病毒变体（用户填充） |
| `variant_pars` | dict(...) | 变体特定参数 |

### 11. 疫苗参数（Vaccine parameters）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vaccine_pars` | {} | 使用的疫苗（初始化时填充） |
| `vaccine_map` | {} | 从数字到疫苗键的反向映射 |

### 12. 卫生系统参数（Health system parameters）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_beds_hosp` | None | 医院床位数量（默认无限制） |
| `n_beds_icu` | None | ICU 床位数量（默认无限制） |
| `no_hosp_factor` | 2.0 | 无床位时严重转危重的倍数 |
| `no_icu_factor` | 2.0 | 无 ICU 床位时危重转死亡的倍数 |

### 13. 干预和分析（Interventions and analyzers）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `interventions` | [] | 模拟中的干预措施（用户填充） |
| `analyzers` | [] | 自定义分析函数（用户填充） |
| `timelimit` | None | 模拟时间限制（秒） |
| `stopping_func` | None | 中途停止模拟的函数 |

## 如何使用参数

### 方法1：直接传递参数

```python
import covasim as cv

# 直接传递参数作为关键字参数
sim = cv.Sim(
    pop_size=10000,
    n_days=90,
    pop_type='hybrid',
    beta=0.02,
    pop_infected=50
)
sim.run()
```

### 方法2：使用参数字典

```python
import covasim as cv

# 使用字典传递参数
pars = {
    'pop_size': 10000,
    'n_days': 90,
    'pop_type': 'hybrid',
    'beta': 0.02,
    'pop_infected': 50
}
sim = cv.Sim(pars=pars)
sim.run()
```

### 方法3：创建后修改参数

```python
import covasim as cv

# 创建模拟后修改参数
sim = cv.Sim()
sim['pop_size'] = 10000
sim['n_days'] = 90
sim['beta'] = 0.02
sim.run()
```

### 方法4：查看所有参数

```python
import covasim as cv

sim = cv.Sim()
# 查看所有参数
print(sim.pars)

# 查看特定参数
print(sim['pop_size'])
print(sim['beta'])
```

## 重要提示

1. **参数位置**：所有参数定义在 `covasim/parameters.py` 的 `make_pars()` 函数中（第15-151行）

2. **网络参数**：网络相关参数（`contacts`、`beta_layer` 等）在人口初始化后由 `reset_layer_pars()` 自动设置，但可以在创建模拟前手动设置

3. **初始化顺序**：
   ```python
   sim = cv.Sim(...)  # 创建模拟对象
   sim.initialize()   # 初始化人口和参数
   sim.run()          # 运行模拟
   sim.plot()         # 绘制结果
   ```

4. **查看参数文档**：
   - 代码文档：`covasim/parameters.py`
   - 文本文档：`covasim/README.rst`
   - 示例代码：`examples/` 目录

## 快速参考

最常用的参数组合：

```python
import covasim as cv

# 基本模拟
sim = cv.Sim(pop_size=10000, n_days=90)

# 使用混合人口
sim = cv.Sim(pop_size=10000, n_days=90, pop_type='hybrid')

# 自定义传播率
sim = cv.Sim(pop_size=10000, n_days=90, beta=0.02)

# 加载数据
sim = cv.Sim(pop_size=10000, datafile='data.xlsx')

# 设置初始感染数
sim = cv.Sim(pop_size=10000, n_days=90, pop_infected=100)
```
