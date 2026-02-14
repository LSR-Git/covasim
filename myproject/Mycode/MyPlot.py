"""
Plotting utilities for Covasim contact networks.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    # Minimal example: small graph with layer attribute (no dependency on cross_network)
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, key=0, layer='base', weight=1.0)
    G.add_edge(1, 2, key=0, layer='base', weight=1.0)
    G.add_edge(2, 0, key=0, layer='cross', weight=0.6)
    for n in [0, 1, 2]:
        G.nodes[n]['age'] = 30
    plot_contact_network(G, layers='base', size=3, figsize=(6, 4), seed=42)
