import numpy as np
import covasim as cv
import Enums
import sciris as sc
import os
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import ContactNetwork
import CrossNetwork
import MyPlot

# 设置 matplotlib 显示中文（Windows 常用 SimHei / 微软雅黑）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框

msim = cv.MultiSim.load('E:/大论文相关/covasim/myproject/results/双耦合网络图片/跨境传播敏感性/.msim')

# 绘制三个 sim 的累计感染人数
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for sim in msim.sims:
    ax.plot(sim.results['t'], sim.results['cum_infections'].values, label=sim.label)
ax.set_xlabel('天数')
ax.set_ylabel('累计感染人数')
ax.set_title('不同流动人口比例下累计感染人数对比')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
result_path = r'E:\大论文相关\covasim\myproject\results\双耦合网络图片\跨境传播敏感性\result.png'
os.makedirs(os.path.dirname(result_path), exist_ok=True)
plt.savefig(result_path, dpi=150)
plt.show()
