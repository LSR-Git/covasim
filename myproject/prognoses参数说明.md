# `get_prognoses` 函数说明

## 函数作用

`get_prognoses()` 函数用于**生成疾病预后的概率参数**，这些参数控制不同年龄段的人在感染 COVID-19 后的疾病进展概率。

## 返回的参数结构

函数返回一个字典，包含以下键值对：

```python
prognoses = {
    'age_cutoffs':    # 年龄分组的边界值（如 [0, 10, 20, 30, ...]）
    'sus_ORs':        # 易感性比值（不同年龄段的相对易感性）
    'trans_ORs':      # 传播性比值（不同年龄段的相对传播能力）
    'symp_probs':     # 出现症状的概率（条件概率：感染后出现症状）
    'severe_probs':   # 发展为重症的概率（条件概率：有症状后发展为重症）
    'crit_probs':     # 发展为危重症的概率（条件概率：重症后发展为危重症）
    'death_probs':    # 死亡概率（条件概率：危重症后死亡）
    'comorbidities':  # 合并症因子（影响重症概率的倍数）
}
```

## 这些参数如何影响模拟

1. **易感性 (sus_ORs)**: 控制不同年龄段的人被感染的概率
2. **传播性 (trans_ORs)**: 控制不同年龄段的人感染他人时的传播能力
3. **症状概率 (symp_probs)**: 控制感染后出现症状的概率
4. **重症概率 (severe_probs)**: 控制有症状后发展为重症的概率
5. **危重症概率 (crit_probs)**: 控制重症后发展为危重症的概率
6. **死亡概率 (death_probs)**: 控制危重症后死亡的概率

## 如何在 `pars` 中自定义

### 方法1：创建模拟后修改（推荐）

```python
import covasim as cv

# 创建模拟
sim = cv.Sim(pop_size=100, n_days=90)

# 在初始化之前修改预后参数
# 例如：将最老年龄组（80+）的死亡概率翻倍
sim['prognoses']['death_probs'][-1] *= 2.0

# 或者修改特定年龄组的症状概率
# sim['prognoses']['symp_probs'][5] = 0.8  # 50-60岁年龄组

sim.initialize()
sim.run()
```

### 方法2：先获取默认值，修改后再传入

```python
import covasim as cv
import numpy as np

# 获取默认的预后参数
prognoses = cv.get_prognoses()

# 自定义修改
# 例如：降低所有年龄段的死亡概率
prognoses['death_probs'] *= 0.5

# 或者创建完全自定义的预后参数
custom_prognoses = {
    'age_cutoffs': np.array([0, 20, 40, 60, 80]),
    'sus_ORs': np.array([0.5, 1.0, 1.0, 1.2, 1.5]),
    'trans_ORs': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    'symp_probs': np.array([0.4, 0.6, 0.7, 0.8, 0.9]),
    'severe_probs': np.array([0.001, 0.01, 0.05, 0.15, 0.25]),
    'crit_probs': np.array([0.0001, 0.001, 0.005, 0.04, 0.17]),
    'death_probs': np.array([0.00001, 0.0001, 0.001, 0.008, 0.08]),
    'comorbidities': np.array([1.0, 1.0, 1.0, 1.0, 1.0])
}

# 转换为条件概率（必须调用）
custom_prognoses = cv.parameters.relative_prognoses(custom_prognoses)

# 创建模拟时传入
sim = cv.Sim(pop_size=100, n_days=90, prognoses=custom_prognoses)
sim.initialize()
sim.run()
```

### 方法3：在你的代码中使用

```python
import numpy as np
import covasim as cv
import ContactNetwork

# ... 你的其他代码 ...

# 创建自定义人口
custom_popdict, custom_keys = ContactNetwork.create_custom_population(100, custom_config_test, countries_config)

# 创建自定义预后参数
custom_prognoses = cv.get_prognoses(by_age=True)  # 获取默认值

# 自定义修改（例如：降低死亡风险）
custom_prognoses['death_probs'] *= 0.5  # 所有年龄段的死亡概率减半

# 创建模拟
sim = cv.Sim(
    pop_size=100, 
    n_days=90,
    prognoses=custom_prognoses  # 传入自定义的预后参数
)
sim.popdict = custom_popdict
sim.reset_layer_pars() 
sim.initialize()
sim.run()
```

## 重要注意事项

1. **必须在 `sim.initialize()` 之前修改**：预后参数在初始化时被用来设置每个人的属性，所以必须在初始化前设置好。

2. **使用条件概率**：Covasim 内部使用条件概率（conditional probabilities），即：
   - `severe_probs` = P(重症 | 有症状)
   - `crit_probs` = P(危重症 | 重症)
   - `death_probs` = P(死亡 | 危重症)

3. **数组长度必须匹配**：所有数组的长度必须等于 `age_cutoffs` 的长度。

4. **如果修改了预后参数，需要重新初始化**：
   ```python
   sim['prognoses']['death_probs'][-1] *= 2.0
   sim.init_people()  # 重新初始化人员属性
   ```

## 示例：针对特定年龄组调整

```python
import covasim as cv

sim = cv.Sim(pop_size=1000, n_days=90)

# 查看默认的年龄分组
print("年龄分组:", sim['prognoses']['age_cutoffs'])
# 输出: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# 修改 60-70 岁年龄组（索引为 6）的死亡概率
sim['prognoses']['death_probs'][6] *= 1.5  # 增加50%

# 修改 80+ 岁年龄组（索引为 -1，即最后一个）的症状概率
sim['prognoses']['symp_probs'][-1] = 0.95  # 设置为95%

sim.initialize()
sim.run()
```

## 总结

- `get_prognoses()` 返回疾病预后的概率参数
- 这些参数存储在 `pars['prognoses']` 中
- **可以在 `pars` 中自定义**，但必须在 `sim.initialize()` 之前设置
- 这些参数控制不同年龄段的人在感染后的疾病进展概率
