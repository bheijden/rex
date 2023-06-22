from functools import lru_cache
import grandiso
import networkx as nx

# @lru_cache()
# def _is_node_attr_match(motif_node_id: str, host_node_id: str, motif: nx.Graph, host: nx.Graph) -> bool:
# 	"""
# 	Check if a node in the host graph matches the attributes in the motif.
#
# 	Arguments:
# 		motif_node_id (str): The motif node ID
# 		host_node_id (str): The host graph ID
# 		motif (nx.Graph): The motif graph
# 		host (nx.Graph): The host graph
#
# 	Returns:
# 		bool: True if the host node matches the attributes in the motif
#
# 	"""
# 	return host.nodes[host_node_id]["kind"] == motif.nodes[motif_node_id]["kind"]


G = nx.DiGraph()

G.add_node("A_0", kind="A")
G.add_node("B_0", kind="B")
G.add_node("B_1", kind="B")

G.add_edge("A_0", "B_0")
G.add_edge("B_0", "B_1")

P_edges = [('0_0', '1_0'), ('1_0', '1_1')]
P_nodes = ('0_0', '1_1', '1_0')
P = nx.DiGraph()
for n in P_nodes:
	P.add_node(n, kind=n.split('_')[0])
for e in P_edges:
	P.add_edge(*e)
P_interestingness = {k: 1 if "1_" in k else 0 for k in P.nodes}

MCS_edges = [('0_s0', '1_s0'), ('1_s0', '1_s1')]
MCS_nodes = ('0_s0', '1_s0', '1_s1')
MCS = nx.DiGraph()
for n in MCS_nodes:
	MCS.add_node(n, kind=n.split('_')[0])
for e in MCS_edges:
	MCS.add_edge(*e)
MCS_interestingness = {k: 1 if "1_" in k else 0 for k in MCS.nodes}

_is_node_attr_match = lambda motif_node_id, host_node_id, motif, host: host.nodes[host_node_id]["kind"] == motif.nodes[motif_node_id]["kind"]
_is_node_attr_match = lru_cache()(_is_node_attr_match)
try:
	x = next(grandiso.find_motifs_iter(P, MCS, directed=True, is_node_attr_match=_is_node_attr_match, interestingness=P_interestingness))
	gr_match = True
except StopIteration:
	gr_match = False
print(gr_match)

# try:
# 	x = next(grandiso.find_motifs_iter(G, G, directed=True, is_node_attr_match=_is_node_attr_match))
# 	gr_match = True
# except StopIteration:
# 	gr_match = False
# print(gr_match)