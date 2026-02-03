# 如何确认导入的是本地还是 pip 安装的 covasim

## Python 导入机制

当执行 `import covasim as cv` 时，Python 会按以下顺序搜索：

1. **当前目录**（运行脚本的目录）
2. **PYTHONPATH** 环境变量中的目录
3. **sys.path** 中的目录（包括 site-packages，即 pip 安装的位置）

## 如何确认当前导入的是哪个版本

### 方法1：查看模块路径（最直接）

在代码中添加以下代码来查看导入的模块路径：

```python
import covasim as cv
import os

# 查看 covasim 模块的文件路径
print("Covasim 模块路径:", cv.__file__)
print("Covasim 模块目录:", os.path.dirname(cv.__file__))

# 判断是本地还是 pip 安装的
module_path = os.path.dirname(cv.__file__)
if 'site-packages' in module_path or 'dist-packages' in module_path:
    print("✅ 导入的是 pip 安装的版本")
else:
    print("✅ 导入的是本地开发版本")
```

### 方法2：查看版本信息

```python
import covasim as cv

# 查看版本
print("版本:", cv.__version__)
print("版本日期:", cv.__versiondate__)

# 查看模块路径
import os
print("模块路径:", os.path.dirname(cv.__file__))
```

### 方法3：在脚本开头添加检查代码

在你的 `t11_custom_population_type.py` 文件开头添加：

```python
import numpy as np
import covasim as cv
import sciris as sc
import os

# 检查导入的版本
print("="*50)
print("Covasim 导入信息:")
print(f"模块路径: {os.path.dirname(cv.__file__)}")
print(f"版本: {cv.__version__}")
if 'site-packages' in os.path.dirname(cv.__file__):
    print("来源: pip 安装的版本")
else:
    print("来源: 本地开发版本")
print("="*50)
```

## 不同情况下的导入行为

### 情况1：从项目根目录运行（未安装）

```bash
# 在 E:\大论文相关\covasim 目录下运行
cd E:\大论文相关\covasim
python examples/t11_custom_population_type.py
```

**结果：** 会导入本地的 `covasim` 目录（因为当前目录在 sys.path 中）

### 情况2：以可编辑模式安装（推荐用于开发）

```bash
# 在项目根目录执行
cd E:\大论文相关\covasim
pip install -e .
```

**结果：** 会导入本地版本（`-e` 表示可编辑模式，会链接到本地目录）

### 情况3：普通 pip 安装

```bash
pip install covasim
```

**结果：** 会导入 pip 安装的版本（在 site-packages 中）

### 情况4：从其他目录运行

```bash
# 从其他目录运行
cd C:\Users\YourName
python E:\大论文相关\covasim\examples\t11_custom_population_type.py
```

**结果：** 
- 如果已 `pip install -e .`：导入本地版本
- 如果只 `pip install covasim`：导入 pip 版本
- 如果都没安装：可能导入失败或导入本地（如果项目目录在 PYTHONPATH 中）

## 如何强制使用本地版本

### 方法1：使用可编辑安装（推荐）

```bash
cd E:\大论文相关\covasim
pip install -e .
```

这样无论从哪里运行，都会使用本地版本。

### 方法2：修改 sys.path

在脚本开头添加：

```python
import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import covasim as cv
```

### 方法3：使用相对导入（不推荐，复杂）

```python
# 需要修改导入方式，比较复杂
```

## 如何强制使用 pip 安装的版本

### 方法1：从其他目录运行

```bash
# 从项目外的目录运行
cd C:\Users\YourName
python E:\大论文相关\covasim\examples\t11_custom_population_type.py
```

### 方法2：临时移除本地路径

```python
import sys
import os

# 移除项目目录（如果存在）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root in sys.path:
    sys.path.remove(project_root)

import covasim as cv
```

## 推荐做法

### 对于开发/修改代码：

```bash
# 1. 进入项目根目录
cd E:\大论文相关\covasim

# 2. 以可编辑模式安装
pip install -e .

# 3. 这样无论从哪里运行，都会使用本地版本
python examples/t11_custom_population_type.py
```

### 对于只是使用（不修改代码）：

```bash
# 直接使用 pip 安装的版本
pip install covasim
```

## 快速检查脚本

创建一个 `check_import.py` 文件：

```python
"""快速检查导入的 covasim 版本"""
import sys
import os

try:
    import covasim as cv
    module_path = os.path.dirname(cv.__file__)
    
    print("="*60)
    print("Covasim 导入检查")
    print("="*60)
    print(f"模块路径: {module_path}")
    print(f"版本: {cv.__version__}")
    print(f"版本日期: {cv.__versiondate__}")
    
    if 'site-packages' in module_path or 'dist-packages' in module_path:
        print("\n✅ 来源: pip 安装的版本")
        print("   位置: Python 的 site-packages 目录")
    else:
        print("\n✅ 来源: 本地开发版本")
        print("   位置: 项目目录")
    
    print("\n当前工作目录:", os.getcwd())
    print("Python 路径 (sys.path):")
    for i, path in enumerate(sys.path[:5], 1):  # 只显示前5个
        print(f"  {i}. {path}")
    if len(sys.path) > 5:
        print(f"  ... 还有 {len(sys.path)-5} 个路径")
    print("="*60)
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已安装 covasim: pip install covasim")
```

运行它：
```bash
python check_import.py
```

## 总结

- **默认行为**：Python 会优先导入当前目录或 sys.path 前面的路径中的模块
- **推荐做法**：开发时使用 `pip install -e .`，这样会始终使用本地版本
- **检查方法**：使用 `cv.__file__` 查看模块路径，判断是本地还是 pip 安装的
