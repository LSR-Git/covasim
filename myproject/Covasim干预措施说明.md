# Covasim 可用干预措施（Interventions）说明

## 📚 在哪里查看干预措施？

### 1. **源代码文件**
- **主要文件**: `covasim/interventions.py` - 包含所有干预措施的完整实现和文档
- **查看方法**: 
  ```python
  import covasim as cv
  help(cv.test_prob)  # 查看某个干预措施的详细文档
  ```

### 2. **示例文件**
- **位置**: `examples/` 目录
- **相关示例**:
  - `t05_testing.py` - 测试相关干预
  - `t05_contact_tracing.py` - 接触者追踪
  - `t05_change_beta.py` - 改变传播率
  - `t08_vaccinate_prob.py` - 概率型疫苗接种
  - `t08_vaccinate_num.py` - 数量型疫苗接种

### 3. **教程文档**
- **位置**: `docs/tutorials/tut_interventions.ipynb` - 完整的干预措施教程

### 4. **在线查看**
```python
# 在 Python 中查看所有可用的干预措施
import covasim as cv
print(dir(cv))  # 查看所有可用的函数和类

# 或者查看 interventions 模块
from covasim import interventions
print([x for x in dir(interventions) if not x.startswith('_')])
```

---

## 🎯 所有可用的干预措施列表

### 一、基础干预类
1. **`Intervention`** - 所有干预措施的基类（用于创建自定义干预）

### 二、通用干预
2. **`dynamic_pars`** - 动态修改模拟参数
   ```python
   cv.dynamic_pars({'beta': {'days': [40, 50], 'vals': [0.005, 0.015]}})
   ```

3. **`sequence`** - 按顺序应用多个干预措施
   ```python
   cv.sequence(days=[15, 30, 45], interventions=[...])
   ```

### 三、传播率（Beta）干预
4. **`change_beta`** - 改变传播率（最常用的干预）
   ```python
   cv.change_beta(days=25, changes=0.3)  # 在第25天将传播率降低到0.3
   cv.change_beta([14, 28], [0.7, 1], layers='s')  # 针对特定层
   ```

5. **`clip_edges`** - 减少接触数量（而不是传播概率）
   ```python
   cv.clip_edges(days=25, changes=0.3)  # 减少70%的接触
   ```

### 四、测试相关干预
6. **`test_prob`** - 基于概率的测试（推荐）
   ```python
   cv.test_prob(symp_prob=0.2, asymp_prob=0.01, start_day='2020-03-01')
   ```
   - `symp_prob`: 有症状者被测试的概率
   - `asymp_prob`: 无症状者被测试的概率
   - `test_delay`: 测试结果延迟天数
   - `sensitivity`: 测试灵敏度

7. **`test_num`** - 基于数量的测试（用于历史数据）
   ```python
   cv.test_num(daily_tests=500, start_day='2020-03-01')
   cv.test_num(daily_tests='data')  # 从数据文件读取
   ```

### 五、接触者追踪
8. **`contact_tracing`** - 接触者追踪和隔离
   ```python
   cv.contact_tracing(trace_probs=0.3, start_day=50)
   cv.contact_tracing(trace_probs={'h': 1.0, 's': 0.5, 'w': 0.5, 'c': 0.3})
   ```
   - `trace_probs`: 追踪概率（可以是数字或按层的字典）
   - `trace_time`: 追踪延迟天数

### 六、疫苗相关干预
9. **`simple_vaccine`** - 简单疫苗（用于 use_waning=False）
   ```python
   cv.simple_vaccine(days=50, prob=0.3, rel_sus=0.5, rel_symp=0.1)
   ```

10. **`vaccinate_prob`** - 基于概率的疫苗接种（推荐）
    ```python
    cv.vaccinate_prob(vaccine='pfizer', days=30, prob=0.7)
    ```
    - `vaccine`: 疫苗类型（'pfizer', 'moderna', 'astrazeneca', 'johnson' 等）
    - `days`: 开始接种的日期
    - `prob`: 每日接种概率
    - `booster`: 是否为加强针

11. **`vaccinate_num`** - 基于数量的疫苗接种
    ```python
    cv.vaccinate_num(vaccine='pfizer', num_doses=100, sequence='age')
    ```
    - `num_doses`: 每日接种数量
    - `sequence`: 接种优先级（'age' 或自定义函数）

12. **`vaccinate`** - 疫苗干预的包装函数（自动选择 prob 或 num）
    ```python
    cv.vaccinate(vaccine='pfizer', days=30, prob=0.7)  # 自动调用 vaccinate_prob
    cv.vaccinate(vaccine='pfizer', num_doses=100)      # 自动调用 vaccinate_num
    ```

---

## 📖 使用示例

### 完整示例：包含多种干预措施
```python
import covasim as cv

# 1. 测试干预
tp = cv.test_prob(symp_prob=0.2, asymp_prob=0.01, start_day='2022-02-14', test_delay=2)

# 2. 接触者追踪
ct = cv.contact_tracing(trace_probs=0.3, start_day='2022-02-14')

# 3. 改变传播率（例如：社交距离）
cb = cv.change_beta(days='2022-03-01', changes=0.5)  # 降低50%传播率

# 4. 疫苗接种
vx = cv.vaccinate_prob('pfizer', days=5, prob=0.1, start_day='2022-02-20')

# 创建模拟并应用所有干预
sim = cv.Sim(
    pop_size=1000,
    pop_infected=10,
    start_day='2022-02-14',
    end_day='2022-03-29',
    interventions=[tp, ct, cb, vx]  # 添加所有干预措施
)
sim.run()
sim.plot()
```

---

## 🔍 如何查看某个干预措施的详细文档？

### 方法1：使用 help()
```python
import covasim as cv
help(cv.test_prob)
help(cv.vaccinate_prob)
```

### 方法2：查看源代码
```python
# 在 Python 中查看源代码
import inspect
import covasim as cv
print(inspect.getsource(cv.test_prob))
```

### 方法3：查看文档字符串
```python
print(cv.test_prob.__doc__)
```

### 方法4：在 IDE 中查看
- 在 VSCode/PyCharm 等 IDE 中，将鼠标悬停在函数名上
- 或使用 `Ctrl+Click` (Windows) / `Cmd+Click` (Mac) 跳转到定义

---

## 💡 常用干预措施组合

### 基础组合（测试 + 追踪）
```python
tp = cv.test_prob(symp_prob=0.2, asymp_prob=0.01)
ct = cv.contact_tracing(trace_probs=0.3)
sim = cv.Sim(interventions=[tp, ct])
```

### 完整组合（测试 + 追踪 + 疫苗）
```python
tp = cv.test_prob(symp_prob=0.2, asymp_prob=0.01)
ct = cv.contact_tracing(trace_probs=0.3)
vx = cv.vaccinate_prob('pfizer', days=30, prob=0.1)
sim = cv.Sim(interventions=[tp, ct, vx])
```

### 社交距离组合（改变传播率）
```python
cb = cv.change_beta(days=['2022-03-01', '2022-04-01'], changes=[0.5, 1.0])
sim = cv.Sim(interventions=cb)
```

---

## 📝 注意事项

1. **干预措施必须添加到 `interventions` 参数中**：
   ```python
   sim = cv.Sim(interventions=[tp, ct])  # ✅ 正确
   # 或者
   sim = cv.Sim(pars={'interventions': [tp, ct]})  # ✅ 也正确
   ```

2. **日期格式**：可以使用字符串、整数或日期对象
   ```python
   cv.test_prob(start_day='2022-02-14')  # 字符串
   cv.test_prob(start_day=10)             # 整数（从模拟开始的天数）
   ```

3. **多个干预措施**：可以同时使用多个干预措施
   ```python
   sim = cv.Sim(interventions=[tp, ct, cb, vx])
   ```

4. **查看结果**：运行模拟后，这些干预措施会产生相应的结果：
   - `test_prob` → `new_tests`, `new_diagnoses`
   - `contact_tracing` → `new_quarantined`
   - `vaccinate_prob` → `new_doses`, `new_vaccinated`

---

## 🔗 相关资源

- **官方文档**: 查看 `docs/` 目录
- **示例代码**: 查看 `examples/` 目录
- **源代码**: `covasim/interventions.py`
- **测试文件**: `tests/test_interventions.py` - 包含所有干预措施的使用示例
