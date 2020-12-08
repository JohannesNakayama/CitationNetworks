#%%

import bibl_io
import os
import importlib
import cProfile
import matplotlib.pyplot as plt
import networkx as nx

#%%

# get path(s)
wos_paths = [
    os.path.join('data', 'wos_sample_a.txt'),
    os.path.join('data', 'wos_sample_b.txt'),
    os.path.join('data', 'wos_sample_c.txt')
]

# instantiate Bibliography object
B = bibl_io.Bibliography(wos_paths, 'wos')

#%%

# instantiate CitationNetwork object
C = bibl_io.CitationNetwork(B, ['network science'], 'nppc')

#%%

M = bibl_io.MainPathNetwork(C, mc=2, iterations=1)
M.create_network()

#%%

nx.draw_shell(
    M.main_path_net, node_size=70, alpha=0.3, edge_cmap="Greens" 
)

#%%

# nx.write_gexf(M.main_path_net, os.path.join('graphs', 'net2.gexf'))


#%%

wos_paths_2 = [
    os.path.join('data', 'electric-vehicle', 'ev_1.txt'),
    os.path.join('data', 'electric-vehicle', 'ev_2.txt')
]

B2 = bibl_io.Bibliography(wos_paths_2, 'wos')

#%%

C2 = bibl_io.CitationNetwork(B2, ['electric vehicle'], 'nppc')

#%%

M2 = bibl_io.MainPathNetwork(C2)

#%%

# reload modules in case of changes
importlib.reload(bibl_io)

#%% 

# --- EXPLORATORY SECION --- #

# find longest path
l = []
for i in range(len(M.main_paths)):
    l.append(len(M.main_paths[i]))
idx = l.index(max(l))
lp = M.main_paths[idx]
lp_network = nx.DiGraph()
lp_network.add_weighted_edges_from(lp)
nx.draw(lp_network)

#%%

# graph sketch board
p = M.main_paths[25]
n = nx.DiGraph()
n.add_weighted_edges_from(p)
l = nx.planar_layout(n)
nx.draw(n, pos=l)

#%%

nx.draw_shell(n)

#%% 

# plot out-degree distribution
C.out_deg_dist.plot.line('out_degree', 'freq')

#%%

# profile code
cProfile.run('bibl_io.CitationNetwork(B, ["network science"], "nppc")')

#%%

def out_degrees_from(net, node_list):
    out_deg_list = [net.out_degree(node) for node in node_list]
    freq = [0] * (max(out_deg_list) + 1)
    for i in out_deg_list:
        freq[i] += 1
    data = {
        'out_degree': list(range(max(out_deg_list) + 1)),
        'freq': freq
    }
    return(pd.DataFrame(data))

#%%

def prune_net(net, min_out=1, min_edge_w=1):
    drop_nodes = []
    for node in net.nodes:
        if (net.out_degree(node) > 0 and net.out_degree(node) <= min_out):
            drop_nodes.append(node)
    drop_edges = []
    for edge in net.edges:
        if (net[edge[0]][edge[1]]['weight'] <= min_edge_w):
            drop_edges.append(edge)
    for node in net.nodes:
        if (net.in_degree(node) == 0 and net.out_degree(node) == 0):
            drop_nodes.append(node)
    net.remove_nodes_from(drop_nodes)
    net.remove_edges_from(drop_edges)
    return(net)

#%%

prune_net(C.cit_net, 4, 5)

nx.dag_longest_path(C.cit_net)
nx.dag_longest_path_length(C.cit_net)

#%%
