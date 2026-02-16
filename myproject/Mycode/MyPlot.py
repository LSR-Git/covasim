"""
Plotting utilities for Covasim contact networks.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

try:
    import covasim as cv
    _has_covasim = True
except ImportError:
    _has_covasim = False


def plot_contact_network(G, layers=None, size=None, figsize=(10, 8), layout='spring', seed=None, **draw_kwargs):
    """
    Draw a contact network (from sim.people.to_graph()).

    Args:
        G: networkx.MultiDiGraph, contact graph (e.g. people.to_graph()).
        layers: None | str | list[str]. Which layer(s) to draw. None = all layers;
            str = single layer (e.g. 'base', 'cross'); list = multiple layers.
        size: None | int. If int, draw only a random subgraph of this many nodes to avoid clutter; None = draw all.
        figsize: tuple. Matplotlib figure size, default (10, 8).
        layout: str. 'spring' (default, nodes spread evenly in the plane), 'kamada_kawai', or 'circular'.
        seed: int | None. Random seed for node sampling when size is set.
        **draw_kwargs: Passed to nx.draw (e.g. node_size, alpha, font_size).
    """
    if not G.number_of_nodes():
        return
    rng = np.random.RandomState(seed)

    # 1) Filter edges by layer
    if layers is not None:
        layer_set = {layers} if isinstance(layers, str) else set(layers)
        H = nx.MultiDiGraph()
        nodes_in_H = set()
        for u, v, k in G.edges(keys=True):
            edata = dict(G[u][v][k])
            edge_layer = edata.get('layer', None)
            if edge_layer in layer_set:
                H.add_edge(u, v, key=k, **edata)
                nodes_in_H.add(u)
                nodes_in_H.add(v)
        for n in nodes_in_H:
            for key, val in G.nodes[n].items():
                H.nodes[n][key] = val
        G_plot = H
    else:
        G_plot = G.copy()

    if G_plot.number_of_edges() == 0:
        if layers is not None:
            print("plot_contact_network: no edges in selected layer(s), skipping draw.")
        return

    # 2) Optional subsample by size
    nodes = list(G_plot.nodes())
    if size is not None and isinstance(size, int) and len(nodes) > size:
        chosen = rng.choice(len(nodes), size=size, replace=False)
        selected = [nodes[i] for i in chosen]
        G_plot = G_plot.subgraph(selected).copy()

    if G_plot.number_of_nodes() == 0:
        print("plot_contact_network: no nodes after subsampling, skipping draw.")
        return

    # 3) Layout and draw (no arrows; nodes spread evenly in the plane)
    fig, ax = plt.subplots(figsize=figsize)
    if layout == 'circular':
        pos = nx.circular_layout(G_plot)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G_plot)
    else:
        # spring: larger k + more iterations so nodes distribute evenly in the network
        pos = nx.spring_layout(G_plot, seed=seed, k=1.2, iterations=100)
    default_draw = dict(node_size=30, alpha=0.7, with_labels=False, arrows=False)
    default_draw.update(draw_kwargs)
    nx.draw(G_plot, pos, ax=ax, **default_draw)
    plt.show()
    return fig


def _get_node_country(G, n):
    """Get country label for node n: support G.nodes[n]['country'] or G.nodes[n]['person'].country."""
    data = G.nodes[n]
    if 'country' in data:
        return data['country']
    if 'person' in data:
        return getattr(data['person'], 'country', None)
    return None


def _node_colors_by_state(G, nodelist):
    """Return list of colors: infected/exposed=red, susceptible=orange, else=lightgray."""
    colors = []
    for n in nodelist:
        data = G.nodes[n]
        is_inf = data.get('infectious', False) or data.get('exposed', False)
        if is_inf:
            colors.append('red')
        else:
            colors.append('lightgray')
    return colors


def plot_two_country_networks(G, sample_size=50, seed=1, figsize=(14, 7), offset=1.0):
    """
    Draw two country networks side by side (A on the left, B on the right).
    Countries are taken from node attribute 'country' (or person.country); values 'A' and 'B' define the two regions.
    Nodes from country A are drawn as circles, from country B as squares.
    Sampling: sample_size nodes per country subgraph, then add all their neighbors, then take subgraph.

    Args:
        G: single networkx Graph/DiGraph/MultiDiGraph; nodes must have 'country' (e.g. 'A' or 'B').
        sample_size: number of nodes to randomly sample from each country before adding neighbors.
        seed: random seed for sampling and layout.
        figsize: figure size (width, height).
        offset: horizontal offset of the two layouts (A shifted left by offset, B right by offset).
    """
    nodes_A = [n for n in G.nodes() if _get_node_country(G, n) == 'A']
    nodes_B = [n for n in G.nodes() if _get_node_country(G, n) == 'B']
    if not nodes_A or not nodes_B:
        print("plot_two_country_networks: need nodes with country 'A' and 'B' in G.")
        return

    G_A = G.subgraph(nodes_A).copy()
    G_B = G.subgraph(nodes_B).copy()

    np.random.seed(seed)
    rng_A = np.random.default_rng(seed)
    rng_B = np.random.default_rng(seed * seed)

    n_A, n_B = len(nodes_A), len(nodes_B)
    sampled_A = rng_A.choice(nodes_A, size=min(sample_size, n_A), replace=False)
    sampled_B = rng_B.choice(nodes_B, size=min(max(1, sample_size - 5), n_B), replace=False)

    expand_A = set(sampled_A)
    for n in sampled_A:
        expand_A.update(G_A.neighbors(n))
    expand_B = set(sampled_B)
    for n in sampled_B:
        expand_B.update(G_B.neighbors(n))

    sub_A = G_A.subgraph(expand_A).copy()
    sub_B = G_B.subgraph(expand_B).copy()

    # Layout: use edge 'weight' if present, else 'beta'
    def set_layout_weight(G):
        if G.is_multigraph():
            for u, v, k in list(G.edges(keys=True)):
                ed = G.edges[u, v, k]
                if 'weight' not in ed and 'beta' in ed:
                    ed['weight'] = float(ed['beta'])
        else:
            for u, v in G.edges():
                ed = G.edges[u, v]
                if 'weight' not in ed and 'beta' in ed:
                    ed['weight'] = float(ed['beta'])

    set_layout_weight(sub_A)
    set_layout_weight(sub_B)

    pos_A = nx.spring_layout(sub_A, weight='weight', seed=seed)
    pos_B = nx.spring_layout(sub_B, weight='weight', seed=seed)
    for n in pos_A:
        pos_A[n][0] -= offset
    for n in pos_B:
        pos_B[n][0] += offset

    plt.figure(figsize=figsize)

    # A network: nodes by country (circle = A, square = B), color by state (infected=red, susceptible=orange)
    nodes_A_circle = [n for n in sub_A.nodes() if _get_node_country(sub_A, n) == 'A']
    nodes_A_square = [n for n in sub_A.nodes() if _get_node_country(sub_A, n) == 'B']
    nx.draw_networkx_nodes(sub_A, pos=pos_A, nodelist=nodes_A_circle,
                          node_color=_node_colors_by_state(sub_A, nodes_A_circle), edgecolors='black',
                          node_shape='o', node_size=60, linewidths=1.0)
    nx.draw_networkx_nodes(sub_A, pos=pos_A, nodelist=nodes_A_square,
                          node_color=_node_colors_by_state(sub_A, nodes_A_square), edgecolors='black',
                          node_shape='s', node_size=60, linewidths=1.0)
    edgelist_A = list(sub_A.edges())
    nx.draw_networkx_edges(sub_A, pos=pos_A, edgelist=edgelist_A, edge_color='gray', width=0.5, arrows=False)

    # B network: nodes by country, color by state
    nodes_B_circle = [n for n in sub_B.nodes() if _get_node_country(sub_B, n) == 'A']
    nodes_B_square = [n for n in sub_B.nodes() if _get_node_country(sub_B, n) == 'B']
    nx.draw_networkx_nodes(sub_B, pos=pos_B, nodelist=nodes_B_circle,
                          node_color=_node_colors_by_state(sub_B, nodes_B_circle), edgecolors='black',
                          node_shape='o', node_size=60, linewidths=1.0)
    nx.draw_networkx_nodes(sub_B, pos=pos_B, nodelist=nodes_B_square,
                          node_color=_node_colors_by_state(sub_B, nodes_B_square), edgecolors='black',
                          node_shape='s', node_size=60, linewidths=1.0)
    edgelist_B = list(sub_B.edges())
    nx.draw_networkx_edges(sub_B, pos=pos_B, edgelist=edgelist_B, edge_color='gray', width=0.5, arrows=False)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markersize=10, label='Country A'),
        plt.Line2D([0], [0], marker='s', color='w', markeredgecolor='black', markersize=10, label='Country B'),
        plt.Line2D([0], [0], marker='o', color='red', markeredgecolor='black', markersize=10, label='Infected'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('off')
    plt.show()

    print(f"A sampled nodes: {len(sampled_A)}")
    print(f"A nodes after adding neighbors: {len(expand_A)}")
    print(f"B sampled nodes: {len(sampled_B)}")
    print(f"B nodes after adding neighbors: {len(expand_B)}")


# ---------------------------------------------------------------------------
# 按 Country 属性记录两区域疫情，并绘制 2x2 四图（A 左/右、B 左/右）
# ---------------------------------------------------------------------------

if _has_covasim:
    class CountryRegionAnalyzer(cv.Analyzer):
        '''
        按人员的 country 属性每日记录各区域的 S/E/I/R/D 及病程状态，供 plot_two_country_epidemic_curves 使用。
        需在 Sim 的 analyzers 中加入本分析器并运行后，再调用绘图函数。
        '''
        def __init__(self, country_key='country', regions=('A', 'B'), label='CountryRegionAnalyzer'):
            super().__init__(label=label)
            self.country_key = country_key
            self.regions = list(regions)
            self.region_data = None

        def initialize(self, sim=None):
            super().initialize(sim)
            if sim is None:
                return
            n_pts = int(sim.npts)
            self.region_data = {}
            keys_stock = ['n_susceptible', 'n_exposed', 'n_infectious', 'n_recovered', 'n_dead', 'cum_infections']
            keys_severity = ['n_asymptomatic', 'n_presymptomatic', 'n_mild', 'n_severe', 'n_critical', 'n_recovered', 'n_dead']
            for r in self.regions:
                self.region_data[r] = {
                    't': np.arange(n_pts, dtype=float),
                    **{k: np.zeros(n_pts, dtype=float) for k in keys_stock},
                    **{k: np.zeros(n_pts, dtype=float) for k in keys_severity},
                }
            return

        def apply(self, sim):
            if self.region_data is None:
                return
            people = sim.people
            if not hasattr(people, self.country_key) and self.country_key not in people.keys():
                return
            try:
                country_arr = people[self.country_key]
            except Exception:
                return
            t = int(sim.t)
            if t < 0 or t >= len(self.region_data[self.regions[0]]['t']):
                return
            for region in self.regions:
                inds = (country_arr == region)
                if not np.any(inds):
                    continue
                p = people
                # 库存量
                self.region_data[region]['n_susceptible'][t] = np.count_nonzero(p.susceptible[inds])
                self.region_data[region]['n_exposed'][t] = np.count_nonzero(p.exposed[inds])
                self.region_data[region]['n_infectious'][t] = np.count_nonzero(p.infectious[inds])
                self.region_data[region]['n_recovered'][t] = np.count_nonzero(p.recovered[inds])
                self.region_data[region]['n_dead'][t] = np.count_nonzero(p.dead[inds])
                self.region_data[region]['cum_infections'][t] = (
                    self.region_data[region]['n_exposed'][t]
                    + self.region_data[region]['n_infectious'][t]
                    + self.region_data[region]['n_recovered'][t]
                    + self.region_data[region]['n_dead'][t]
                )
                # 病程：无症状=传染期且未症状，症状前=暴露且未到传染日，轻症=症状且非重/危，重/危
                exp_inds = inds.copy()
                inf_inds = inds & p.infectious
                sym_inds = inds & p.symptomatic
                sev_inds = inds & p.severe
                crit_inds = inds & p.critical
                rec_inds = inds & p.recovered
                dead_inds = inds & p.dead
                # 传染但未症状（含无症状与症状前中尚未发病）
                asym_inds = inf_inds & ~p.symptomatic
                # 暴露但尚未传染（症状前）
                pre_inds = inds & p.exposed & ~p.infectious
                # 轻症：有症状且非重、非危
                mild_inds = sym_inds & ~p.severe & ~p.critical
                self.region_data[region]['n_asymptomatic'][t] = np.count_nonzero(asym_inds)
                self.region_data[region]['n_presymptomatic'][t] = np.count_nonzero(pre_inds)
                self.region_data[region]['n_mild'][t] = np.count_nonzero(mild_inds)
                self.region_data[region]['n_severe'][t] = np.count_nonzero(sev_inds)
                self.region_data[region]['n_critical'][t] = np.count_nonzero(crit_inds)
                self.region_data[region]['n_recovered'][t] = np.count_nonzero(rec_inds)  # 与上重复键，覆盖
                self.region_data[region]['n_dead'][t] = np.count_nonzero(dead_inds)
            return


def plot_two_country_epidemic_curves(
    sim,
    country_key='country',
    regions=('A', 'B'),
    analyzer_label='CountryRegionAnalyzer',
    figsize=(12, 10),
    fontsize=10,
    save_path=None,
    show_severity=True,
    show_regions=None,
):
    '''
    按区域画 A/B 两区疫情曲线。
    - show_severity=True（默认）：含右列各病程图；False 则仅左列 S/E/I(cum)/R/D。
    - show_regions=None（默认）：两区都画。'A' 或 ('A',) 只画 A 区；'B' 或 ('B',) 只画 B 区。
      只画 A 区且 show_severity=False 时，即仅显示 A 区左上角那一幅 SEIR 图（1×1）。
    需要 sim 在运行前加入 CountryRegionAnalyzer 并已 run。
    save_path: 可选，若提供则先将图片保存到该路径再 show。
    '''
    if not _has_covasim:
        raise RuntimeError('plot_two_country_epidemic_curves requires covasim')
    plt.rcParams.setdefault('font.sans-serif', ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif'])
    plt.rcParams.setdefault('axes.unicode_minus', False)
    try:
        analyzer = sim.get_analyzer(analyzer_label)
    except Exception:
        raise ValueError(
            'Sim 未找到 CountryRegionAnalyzer。请先加入 analyzers=[CountryRegionAnalyzer()] 并运行 sim.run()。'
        ) from None
    data = getattr(analyzer, 'region_data', None)
    if not data or list(regions) != list(analyzer.regions):
        raise ValueError('Analyzer 中无 region_data 或 regions 与绘图不一致')
    A, B = regions[0], regions[1]
    # 要显示的区：None/'both'→两区，'A'→仅A，'B'→仅B
    if show_regions is None or show_regions == 'both' or show_regions == ('A', 'B'):
        to_show = [A, B]
    elif show_regions == 'A' or show_regions == (A,):
        to_show = [A]
    elif show_regions == 'B' or show_regions == (B,):
        to_show = [B]
    else:
        to_show = list(show_regions)

    n_rows = len(to_show)
    n_cols = 2 if (show_severity and n_rows > 0) else 1
    w, h = (figsize[0], figsize[1]) if isinstance(figsize, (tuple, list)) and len(figsize) >= 2 else (12, 10)
    if n_rows == 1 and n_cols == 1:
        fig, axes = plt.subplots(1, 1, figsize=(w * 0.5, h * 0.5))
        axes = np.array([[axes]])
    elif n_rows == 1 and n_cols == 2:
        fig, axes = plt.subplots(1, 2, figsize=(w, h * 0.5))
        axes = axes.reshape(1, -1)
    elif n_rows == 2 and n_cols == 1:
        # 两区、不画病程：左右排列（左 A 区 SEIR，右 B 区 SEIR）
        fig, axes = plt.subplots(1, 2, figsize=(w, h * 0.5))
        axes = axes.reshape(1, -1)
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize)

    def _get_ax(row, col):
        if n_rows == 1 and n_cols == 1:
            return axes[0, 0]
        # 两区、不画病程时是 1×2 左右排列，SEIR 用 (0, i)
        if n_rows == 2 and n_cols == 1:
            return axes[0, row]
        return axes[row, col]

    colors_seir = dict(S='#808080', E='#ffcc00', I='#e45226', R='#4dac26', D='#000000')
    colors_sev = dict(
        Asymptomatic='#e996c6', Presymptomatic='#87ceeb', Mild='#e67e50', Severe='#8b0000',
        Critical='#4169e1', R='#4dac26', D='#000000'
    )

    def _plot_seir(ax, t, d, title, region_label):
        ax.plot(t, d['n_susceptible'], color=colors_seir['S'], label='S 易感者', marker='o', markersize=2)
        ax.plot(t, d['n_exposed'], color=colors_seir['E'], label='E 暴露者', marker='o', markersize=2)
        ax.plot(t, d['cum_infections'], color=colors_seir['I'], label='I 累计感染者', marker='o', markersize=2)
        ax.plot(t, d['n_recovered'], color=colors_seir['R'], label='R 恢复者', marker='o', markersize=2)
        ax.plot(t, d['n_dead'], color=colors_seir['D'], label='D 死亡者', marker='o', markersize=2)
        ax.set_xlabel('时间 (天)', fontsize=fontsize)
        ax.set_ylabel('人员数量', fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left', fontsize=fontsize - 1)
        ax.grid(True, alpha=0.3)

    def _plot_severity(ax, t, d, title):
        ax.plot(t, d['n_asymptomatic'], color=colors_sev['Asymptomatic'], label='无症状', marker='o', markersize=2)
        ax.plot(t, d['n_presymptomatic'], color=colors_sev['Presymptomatic'], label='症状前', marker='o', markersize=2)
        ax.plot(t, d['n_mild'], color=colors_sev['Mild'], label='轻症', marker='o', markersize=2)
        ax.plot(t, d['n_severe'], color=colors_sev['Severe'], label='重症', marker='o', markersize=2)
        ax.plot(t, d['n_critical'], color=colors_sev['Critical'], label='危重症', marker='o', markersize=2)
        ax.plot(t, d['n_recovered'], color=colors_sev['R'], label='R 恢复者', marker='o', markersize=2)
        ax.plot(t, d['n_dead'], color=colors_sev['D'], label='D 死亡者', marker='o', markersize=2)
        ax.set_xlabel('时间 (天)', fontsize=fontsize)
        ax.set_ylabel('人员数量', fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left', fontsize=fontsize - 1)
        ax.grid(True, alpha=0.3)

    for i, reg in enumerate(to_show):
        t = data[reg]['t']
        _plot_seir(_get_ax(i, 0), t, data[reg], f'({reg}) 每日累计感染者情况', f'区域 {reg}')
        if show_severity and n_cols >= 2:
            _plot_severity(_get_ax(i, 1), t, data[reg], f'({reg}) 每日累计各病程感染者情况')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


if __name__ == '__main__':
    # Minimal example: small graph with layer attribute (no dependency on cross_network)
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, key=0, layer='base', weight=1.0)
    G.add_edge(1, 2, key=0, layer='base', weight=1.0)
    G.add_edge(2, 0, key=0, layer='cross', weight=0.6)
    for n in [0, 1, 2]:
        G.nodes[n]['age'] = 30
    plot_contact_network(G, layers='base', size=3, figsize=(6, 4), seed=42)
