"""
图4-6-1：无跨境 vs 有跨境时 A/B 两区人数对比（四子图）
纵轴为人员数量，与 no_intervention 左侧图一致（S/E/I 累计/R/D）
左上：无跨境 A区  左下：无跨境 B区  右上：有跨境 A区  右下：有跨境 B区
"""
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 保证可导入 MyPlot（CountryRegionAnalyzer 在加载 .sim 时需要）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import covasim as cv
import MyPlot

# 中文与负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 结果目录与 sim 文件
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results', '双耦合网络图片', '无干预模拟'
)
SIM_NO_CROSS = os.path.join(RESULTS_DIR, 'no_cross.sim')
SIM_CROSS = os.path.join(RESULTS_DIR, 'cross_1%.sim')

# 与 MyPlot 中一致的 SEIR 颜色
COLORS_SEIR = dict(S='#808080', E='#ffcc00', I='#e45226', R='#4dac26', D='#000000')


def get_region_data(sim, analyzer_label='CountryRegionAnalyzer'):
    """从已运行的 sim 中取出按区域记录的数据（需含 CountryRegionAnalyzer）。"""
    try:
        analyzer = sim.get_analyzer(analyzer_label)
    except Exception:
        raise ValueError(f'Sim 中未找到 {analyzer_label}，请确保运行 sim 时加入了该分析器。') from None
    data = getattr(analyzer, 'region_data', None)
    if not data:
        raise ValueError('Analyzer 中无 region_data。')
    return data


def region_population(d):
    """用首日各状态人数之和作为该区人口（常数）。"""
    return (
        d['n_susceptible'][0] + d['n_exposed'][0] + d['n_infectious'][0]
        + d['n_recovered'][0] + d['n_dead'][0]
    )


def plot_seir_count(ax, t, d, title, fontsize=10):
    """在 ax 上绘制 S/E/I(cum)/R/D 的人数，风格同 no_intervention 左侧图。"""
    ax.plot(t, d['n_susceptible'], color=COLORS_SEIR['S'], label='S 易感者', marker='o', markersize=2)
    ax.plot(t, d['n_exposed'], color=COLORS_SEIR['E'], label='E 暴露者', marker='o', markersize=2)
    ax.plot(t, d['cum_infections'], color=COLORS_SEIR['I'], label='I 累计感染者', marker='o', markersize=2)
    ax.plot(t, d['n_recovered'], color=COLORS_SEIR['R'], label='R 恢复者', marker='o', markersize=2)
    ax.plot(t, d['n_dead'], color=COLORS_SEIR['D'], label='D 死亡者', marker='o', markersize=2)
    ax.set_xlabel('时间 (天)', fontsize=fontsize)
    ax.set_ylabel('人员数量', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', fontsize=fontsize - 1)
    ax.grid(True, alpha=0.3)


def main():
    if not os.path.isfile(SIM_NO_CROSS):
        raise FileNotFoundError(f'未找到无跨境 sim 文件: {SIM_NO_CROSS}')
    if not os.path.isfile(SIM_CROSS):
        raise FileNotFoundError(f'未找到有跨境 sim 文件: {SIM_CROSS}')

    sim_no = cv.Sim.load(SIM_NO_CROSS)
    sim_cross = cv.Sim.load(SIM_CROSS)

    data_no = get_region_data(sim_no)
    data_cross = get_region_data(sim_cross)

    A, B = 'A', 'B'
    t_no = data_no[A]['t']
    t_cross = data_cross[A]['t']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # 左上/左下：无跨境 A区、B区；右上/右下：有跨境 A区、B区（纵轴均为人员数量）
    plot_seir_count(axes[0, 0], t_no, data_no[A], '无跨境 — A区')
    plot_seir_count(axes[0, 1], t_cross, data_cross[A], '有跨境 — A区')
    plot_seir_count(axes[1, 0], t_no, data_no[B], '无跨境 — B区')
    plot_seir_count(axes[1, 1], t_cross, data_cross[B], '有跨境 — B区')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, '图4-6-1_无跨境与有跨境AB区.png')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'已保存: {out_path}')


if __name__ == '__main__':
    main()
