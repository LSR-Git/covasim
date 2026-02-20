import os
import sys
import matplotlib
import matplotlib.pyplot as plt
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import covasim as cv
import MyPlot
# 设置 matplotlib 显示中文（Windows 常用 SimHei / 微软雅黑）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框
# 保存模拟结果与图片到指定目录（传完整路径，避免 sc.makefilepath 拼接时中文名被截成只剩 .sim）
results_dir = r'myproject\results\多层耦合网络图片\居家办公干预'
os.makedirs(results_dir, exist_ok=True)
sim_basename = 'case01'
sim_path = os.path.join(results_dir, sim_basename + '.sim')

sim = cv.load(sim_path)

# # 按 A/B 两区域分别绘制：左上/右上为 A 区 SEIR+病程，左下/右下为 B 区，并保存图片
# MyPlot.plot_two_country_epidemic_curves(
#     sim, country_key='country', regions=('A', 'B'),
#     save_path=os.path.join(results_dir, sim_basename + '.png'),
#     figsize=(12, 10),
#     show_severity=False,
#     curves = ['n_exposed', 'n_infectious','n_quarantined','n_isolated'],  # 新增参数：只画 I 和 R
#     # curves = ['cum_infections'],
#     show_regions=('A')
# )

# 各层每日新感染人数（按区域、按传播层）
MyPlot.plot_layer_region_infections(
    sim,
    country_key='country',
    regions=('A', 'B'),
    layers=['home', 'school', 'work', 'community'],
    show_regions=('A'),
    save_path=os.path.join(results_dir, sim_basename + '.png'),
)
