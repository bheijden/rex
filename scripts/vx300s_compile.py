import networkx as nx
from typing import Dict, List, Tuple, Union, Optional, Callable, Any, Iterable, Set
import os
import time
import tempfile
import datetime
import tqdm
import yaml
import jumpy.numpy as jp
import jumpy.random
import jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
gpu_device = jax.devices('gpu')[0]
cpu_device = jax.devices('cpu')[0]

# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.initialize_cache("./cache")
# import logging
# logging.getLogger("jax").setLevel(logging.INFO)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib
matplotlib.use("TkAgg")

import rex
import rex.open_colors as oc
from rex.utils import timer
import rex.utils as utils
from rex.multiprocessing import new_process
from rex.base import StepState
from rex.plot import plot_graph, plot_computation_graph
from rex.supergraph import create_graph
import supergraph as sg
from rex.constants import LATEST, BUFFER, FAST_AS_POSSIBLE, SIMULATED, SYNC, PHASE, FREQUENCY, WARN, REAL_TIME, \
	ASYNC, WALL_CLOCK, SCHEDULING_MODES, JITTER_MODES, CLOCK_MODES, INFO, DEBUG
from rex.wrappers import GymWrapper, AutoResetWrapper, VecGymWrapper
from rex.proto import log_pb2

import experiments as exp
import envs.vx300s as vx300s
import envs.vx300s.planner.rex

# Determine default color scheme
default_cscheme = {
    "selected": "red",
    "mpc": "blue",
    "est": "green",
}
ecolor, fcolor = oc.cscheme_fn(default_cscheme)


def constrained_topological_sort(graph, constraint_fn=None, edge_constraint_fn=None, start=None, in_degree=None,
                                 return_delta=False):
    """
    Perform a constrained topological sort on a DAG.
    This function filters nodes based on the provided constraint function while performing the sort.
    """
    if constraint_fn is None:
        constraint_fn = lambda *args, **kwargs: True

    if edge_constraint_fn is None:
        edge_constraint_fn = lambda *args, **kwargs: True

    # Initialize the list for the sorted order
    order = []

    # Determine the in-degree for each node
    if in_degree is None:
        in_degree = {u: 0 for u in graph.nodes()}
        for u, v in graph.edges():
            in_degree[v] += 1

    # Determine the initial set of nodes to start traversals from
    if start is None:
        # Use a set to manage the nodes with zero in-degree
        zero_in_degree = {u for u in graph.nodes() if in_degree[u] == 0}
    else:
        zero_in_degree = set(start)
    start = zero_in_degree.copy() if start is None else start

    delta_in_degree = {}
    while zero_in_degree:
        # Pop a node from the zero in-degree set
        u = zero_in_degree.pop()

        # Check if the node satisfies the constraint
        if constraint_fn(graph, u):
            order.append(u)

            # Decrease the in-degree of successors and add to zero in-degree set if needed
            for v in graph.successors(u):
                if edge_constraint_fn(graph, u, v) and v not in start:
                    delta_in_degree[v] = delta_in_degree.get(v, 0) - 1
                    # in_degree[v] -= 1
                    if in_degree[v] + delta_in_degree[v] == 0:
                        zero_in_degree.add(v)

    if return_delta:
        return order, delta_in_degree
    else:
        return order


data_npruned = {"pruned": True, "alpha": 0.5, "edgecolor": oc.ecolor.pruned, "facecolor": oc.fcolor.pruned}
data_epruned = {"pruned": True, "alpha": 0.5, "color": oc.ecolor.pruned, "linestyle": "--"}


def prune(_G, _n, node: bool = True, in_edges: bool = True, out_edges: bool = True):
    if node:
        _G.nodes[_n].update(data_npruned.copy())
    if in_edges:
        for v, _, data in _G.in_edges(_n, data=True):
            data.update(data_epruned.copy())
    if out_edges:
        for _, u, data in _G.out_edges(_n, data=True):
            data.update(data_epruned.copy())


def calibrate_graph(G: nx.DiGraph, ts_offset: float = 0., relabel_nodes: bool = False) -> nx.DiGraph:
    """
    Calibrates a directed graph by adjusting sequence numbers and time steps of its nodes
    and optionally relabels the nodes.

    Args:
        G (nx.DiGraph): The directed graph to be calibrated.
        ts_offset (float): The time step offset to be applied.
        relabel_nodes (bool, optional): If True, relabels the nodes of the graph. Defaults to False.

    Returns:
        nx.DiGraph: The calibrated (and optionally relabeled) graph.
    """
    seq_offset = {}
    topo = nx.topological_sort(G)
    G_relabel = G.copy(as_view=False)
    relabel_map = {}

    for n in topo:
        kind = G_relabel.nodes[n]["kind"]
        if kind not in seq_offset:
            seq_offset[kind] = G_relabel.nodes[n]["seq"]

        G_relabel.nodes[n]["seq"] -= seq_offset[kind]
        G_relabel.nodes[n]["ts_step"] -= ts_offset
        G_relabel.nodes[n]["ts_sent"] -= ts_offset
        G_relabel.nodes[n]["position"] = (G_relabel.nodes[n]["position"][0] - ts_offset, G_relabel.nodes[n]["position"][1])

        if relabel_nodes:
            new_label = f"{kind}_{G_relabel.nodes[n]['seq']}"
            relabel_map[n] = new_label

        for u, _, edata in G_relabel.out_edges(n, data=True):
            edata["ts_recv"] -= ts_offset
            edata["ts_sent"] -= ts_offset
            edata["seq"] -= seq_offset[kind]

    if relabel_nodes:
        G_relabel = nx.relabel_nodes(G_relabel, relabel_map, copy=False)

    return G_relabel


def make_partition_constraint_fn(ts_end):
    def _constraint_fn(_G, _n):
        if _G.nodes[_n]["pruned"]:
            return False
        elif _G.nodes[_n]["kind"] not in ["world", "armactuator", "controller", "planner"]:
            return False
        elif _G.nodes[_n]["kind"] in ["planner", "controller"] and _G.nodes[_n]["ts_step"] > ts_end:
            return False
        else:
            return True

    return _constraint_fn


def edge_constraint_fn(_G, _u, _v):
    if _G.edges[_u, _v]["pruned"]:
        return False
    else:
        return True


def make_prune_constraint_fn(_n):
    def _constraint_fn(_G, _u):
        if _u == _n:
            return False
        else:
            return True

    return _constraint_fn


def make_mpc_partitions(G: nx.DiGraph, dt_horizon: float, until_idx: int):
    # Base graph has all pruned graphs removed
    G = rex.supergraph.prune_graph(G, copy=False)

    # Split nodes by kind and sort by sequence
    ndict: Dict[str, Dict[str, Dict]] = {}
    for n, data in G.nodes(data=True):
        if data["kind"] not in ndict:
            ndict[data["kind"]] = {}
        ndict[data["kind"]][n] = data

    # Sort nodes by sequence in list
    nlist_sorted: Dict[str, List[Tuple[str, Dict]]] = {}
    for kind, nodes in ndict.items():
        nlist_sorted[kind] = [(k, v) for k, v in sorted(nodes.items(), key=lambda item: item[1]["seq"])]

    # Start pruning from nodes with unpruned indeg==0
    start_prune = []
    for n, data in G.nodes(data=True):
        if data["kind"] != "planner" and G.in_degree(n) == 0:
            start_prune.append(n)

    # Remove all incoming edges of planner nodes (except for stateful ones)
    for (n, data) in nlist_sorted["planner"]:
        for u, v, edata in G.in_edges(n, data=True):
            if not edata["stateful"]:
                edata.update(data_epruned.copy())

    # Determine the in-degree for each node
    in_degree = {u: 0 for u in G.nodes()}
    for u, v, edata in G.edges(data=True):
        if not edata["pruned"]:
            in_degree[v] += 1

    # Then, iteratively unprune a planner node and get an MPC partition
    Gs_mpc = {}
    for (n, data) in nlist_sorted["planner"][:until_idx]:
        # Prune all nodes that are not descendants of the planner node
        constraint_fn = make_prune_constraint_fn(n)
        order_prune, delta_in_degree = constrained_topological_sort(G, start=start_prune, in_degree=in_degree,
                                                                    return_delta=True, constraint_fn=constraint_fn,
                                                                    edge_constraint_fn=edge_constraint_fn)
        for n_prune in order_prune:
            prune(G, n_prune, node=True, in_edges=False, out_edges=True)
        for n_delta, acc in delta_in_degree.items():
            in_degree[n_delta] += acc  # Acc is negative

        # Find MPC partition
        ts_start = G.nodes[n]["ts_step"]
        ts_controller = min([G.nodes[v]["ts_step"] for u, v, edata in G.edges(n, data=True) if
                             G.nodes[v]["kind"] == "controller"])
        ts_end = ts_controller + dt_horizon
        constraint_fn = make_partition_constraint_fn(ts_end)
        start_partition = [n]
        order_partition = constrained_topological_sort(G, start=start_partition, in_degree=in_degree,
                                                       constraint_fn=constraint_fn, return_delta=False,
                                                       edge_constraint_fn=edge_constraint_fn)

        # Partition graph
        G_mpc = G.subgraph(order_partition).copy()
        Gs_mpc[n] = G_mpc

        # todo: redirect edges to first planner node
        # todo: plumb through estimator nodes -->
        # todo: what to do with all ndata["ts_step"], ndata["ts_sent"], edata["seq"], edata["ts_recv"]
        # todo: what to do with ndata["seq"] and edata["seq"]?
        # todo: Recalibrate all seq numbers to start from 0
        # todo: Pruned incoming edges should have seq numbers of -1, -2, etc... --> can buffers handle this?

        # Get the first node of every kind # todo: use this to offset seq numbers
        first_nodes = {}
        topo = nx.topological_sort(G_mpc)
        for n in topo:
            if G_mpc.nodes[n]["kind"] not in first_nodes:
                first_nodes[G_mpc.nodes[n]["kind"]] = n

        # Prune all nodes that are only connected in MPC partition
        start_prune = start_partition

        # if False:
        #     _ = vx300s.show_computation_graph(G, root=None, xmax=2.0, draw_pruned=True)
        #     _ = vx300s.show_computation_graph(G_mpc, root=None, xmax=None, draw_pruned=True)

    return Gs_mpc


def make_partitions(G: nx.DiGraph, dt_horizon: float, verbose: bool = False, progress_bar: bool = False):
    """
    This function partitions a directed graph (DiGraph) into estimator and model predictive control (MPC) partitions
    based on the nodes' roles and interactions within a robotic system's computational graph.
    It prunes the graph, sorts nodes by kind and sequence, and creates partitions for MPC planning nodes
    considering their connections to sensors, actuators, and controllers.

    Arguments:
    G (nx.DiGraph): The directed graph representing the computational network of the robotic system.
        It is modified in-place.
    dt_horizon (float): The time horizon for the MPC planning. It defines the time frame within which the
        planner operates.
    verbose (bool, optional): If True, enables verbose output during the partitioning process. Default is False.
    progress_bar (bool, optional): If True, displays a progress bar for the partitioning process. Default is False.

    Returns:
    dict: A dictionary where keys are planner node names and values are dictionaries containing the
    subgraphs for estimator ('est') and MPC ('mpc') partitions, as well as the original ('partition') and calibrated ('calibrated') subgraphs.
    """

    # Base graph has all pruned graphs removed
    G = rex.supergraph.prune_graph(G, copy=False)

    # Split nodes by kind and sort by sequence
    ndict: Dict[str, Dict[str, Dict]] = {}
    for n, data in G.nodes(data=True):
        if data["kind"] not in ndict:
            ndict[data["kind"]] = {}
        ndict[data["kind"]][n] = data

    # Sort nodes by sequence in list
    # nlist_sorted: Dict[str, List[Tuple[str, Dict]]] = {}
    nlist_sorted: Dict[str, List[str]] = {}
    for kind, nodes in ndict.items():
        # nlist_sorted[kind] = [(k, v) for k, v in sorted(nodes.items(), key=lambda item: item[1]["seq"])]
        nlist_sorted[kind] = [k for k, v in sorted(nodes.items(), key=lambda item: item[1]["seq"])]

    # Determine the in-degree for each node
    in_degree = {u: 0 for u in G.nodes()}
    for u, v, edata in G.edges(data=True):
        if not edata["pruned"]:
            in_degree[v] += 1

    # Step-by-step process for partitioning each MPC planner node:
    # 1. Iterate through each MPC planner node in the sorted list.
    # 2. For each planner node, follow the graph edges to identify key sensor, actuator, and controller nodes connected to it.
    # 3. Starting from the planner node, trace back to find the nearest boxsensor node, then to the world node connected to this boxsensor.
    # 4. Identify the nearest world node connected to an armsensor node.
    # 5. Determine the actuator and controller nodes associated with the world nodes.
    # 6. For the controller nodes, establish the planning horizon based on the time steps and the given dt_horizon.
    # 7. Collect all controller nodes within this planning horizon.
    # 8. Apply node constraints and perform a constrained topological sort to obtain the initial partition.
    # 9. Split this partition into two: one before the horizon (pre-horizon) and one within the horizon.
    # 10. In the partition, redirect edges as necessary to maintain graph consistency.
    # 11. Store the partitions in the Gs dictionary, keyed by the planner node name.
    # 12. If verbose, print the details of the partitioned nodes and edges for each planner node.
    # 13. Optionally, visualize the computational graph for debugging or presentation purposes.
    Gs = {}
    desc = "Finding MPC partitions"
    pbar = tqdm.tqdm(total=len(nlist_sorted["planner"]), desc=desc, disable=not progress_bar)
    for j, n_plan in enumerate(nlist_sorted["planner"]):
        # Update progressbar
        skipped = j + 1 - len(Gs)
        pbar.set_postfix_str(f"skipped {skipped}/{j+1}")
        pbar.update(1)

        n_box, n_arm, n_world_box, n_world_arm, n_world, n_controller, n_first_controller, n_actuator, n_prev_plan = None, None, None, None, None, None, None, None, None
        # Find the previous boxsensor that provided its measurement to the planner
        for u, _ in G.in_edges(n_plan):
            if G.nodes[u]["kind"] == "boxsensor":
                n_box = u
                break
        if n_box is None:
            print(f"Skipping planner node {n_plan} because no boxsensor node found") if verbose else None
            continue

        # Find the previous world node that provided its state to the boxsensor
        for u, _ in G.in_edges(n_box):
            if G.nodes[u]["kind"] == "world":
                n_world_box = u
                break
        if n_world_box is None:
            print(f"Skipping planner node {n_plan} because no world node for the boxsensor found") if verbose else None
            continue

        # Find the nearest world node that provided a measurement to the armsensor
        n_world_arm = n_world_box
        while True:
            n_prev_world = None
            for _, v, edata in G.out_edges(n_world_arm, data=True):
                if G.nodes[v]["kind"] == "armsensor":
                    n_arm = v
                    break
                elif edata["stateful"]:
                    n_prev_world = v
            if n_arm is not None or n_prev_world is None:
                break
            else:
                n_world_arm = n_prev_world
        if n_arm is None:
            print(f"Skipping planner node {n_plan} because no armsensor node found") if verbose else None
            continue

        # Select the world node that comes right after the world_arm node, because that's the node we need to initialize with the arm and box state measurements
        for _, v, edata in G.out_edges(n_world_arm, data=True):
            if edata["stateful"]:
                n_world = v
                break
        if n_world is None:
            print(f"Skipping planner node {n_plan} because no world node found") if verbose else None
            continue

        # Find the previous actuator, controller, and prev_plan nodes
        for u, _ in G.in_edges(n_world):
            if G.nodes[u]["kind"] == "armactuator":
                n_actuator = u
                break
        if n_actuator is None:
            print(f"Skipping planner node {n_plan} because no armactuator node found") if verbose else None
            continue

        for u, _ in G.in_edges(n_actuator):
            if G.nodes[u]["kind"] == "controller":
                n_controller = u
                break
        if n_controller is None:
            print(f"Skipping planner node {n_plan} because no controller node found") if verbose else None
            continue

        for u, _ in G.in_edges(n_controller):
            if G.nodes[u]["kind"] == "planner":
                n_prev_plan = u
                break
        if n_prev_plan is None:
            print(f"Skipping planner node {n_plan} because no prev_plan node found") if verbose else None
            continue

        # Find n_controller plan horizon
        n_horizon = []
        for _, v in G.out_edges(n_plan):
            if G.nodes[v]["kind"] == "controller":
                n_horizon.append(v)
        if len(n_horizon) == 0:
            print(f"Skipping planner node {n_plan} because no downstream controller node found") if verbose else None
            continue

        # Find the controller in n_horizon based on the minimum ts_step (don't use numpy)
        n_first_controller = n_horizon[0]
        for v in n_horizon:
            if G.nodes[v]["ts_step"] < G.nodes[n_first_controller]["ts_step"]:
                n_first_controller = v

        # Determine the start and end time steps for the planner
        ts_start = G.nodes[n_plan]["ts_step"]
        ts_controller = G.nodes[n_first_controller]["ts_step"]
        # ts_future = ts_controller - ts_start
        ts_end = ts_controller + dt_horizon

        # Find all controller nodes that are in the horizon of this planner
        n_all_controllers = [n_controller]
        while True:
            next_controller = None
            for _, v, edata in G.out_edges(n_all_controllers[-1], data=True):
                if edata["stateful"]:
                    next_controller = v
                    break

            # If no next controller found, then we're at the end of the graph
            if next_controller is None:
                break

            # If the next controller is outside the horizon, then we're at the end of the horizon
            if G.nodes[next_controller]["ts_step"] <= ts_end:
                n_all_controllers.append(next_controller)
            else:
                break
        if next_controller is None:
            print(f"Skipping planner node {n_plan} because no controller outside horizon found") if verbose else None
            continue

        def _node_constraint_fn(_G, _u):
            if _G.nodes[_u]["kind"] in ["armactuator", "planner", "world"]:
                return True
            elif _G.nodes[_u]["kind"] == "controller" and _G.nodes[_u]["ts_step"] <= ts_end:
                return True
            else:
                return False

        start_partition = n_all_controllers + [n_plan, n_actuator, n_world]
        # delta_in_degree = {}
        # for n in start_partition:
        #     for _, v in G.out_edges(n):
        #         if v in start_partition:
        #             delta_in_degree[v] = delta_in_degree.get(v, 0) + 1  # Add one so that it does not get added twice to zero_in_degree

        # Correct for the delta_in_degree
        # for k, dd in delta_in_degree.items():
        #     in_degree[k] = in_degree.get(k, 0) + dd
        partition = constrained_topological_sort(G, start=start_partition, in_degree=in_degree,
                                                 constraint_fn=_node_constraint_fn, return_delta=False,
                                                 edge_constraint_fn=None)
        # Correct for the delta_in_degree
        # for k, dd in delta_in_degree.items():
        #     in_degree[k] = in_degree.get(k, 0) - dd

        # Split partition into two parts: the part before the horizon, and the part in the horizon
        G_part = G.subgraph(partition).copy()
        n_pre_horizon, n_horizon = [], []
        for n in n_all_controllers:
            if G.nodes[n]["ts_step"] < ts_controller:
                n_pre_horizon.append(n)
            else:
                n_horizon.append(n)

        # Redirect edges to first planner node (may not always be connected if window=1)
        edata = G_part.edges[n_plan, n_first_controller]
        for n in n_horizon:
            G_part.add_edge(n_plan, n, **edata)
        assert len(G_part.out_edges(n_plan)) == len(n_horizon), "Not all edges redirected"

        # Get all world nodes to which we are connecting a cost node
        # n_world_iter = n_world
        # assert n_world_iter in G_part.nodes, f"World node {_n_world} not in partition"
        # n_world_cost = []
        # while True:
        #     n_world_next = None
        #     for _, v, edata in G_part.out_edges(n_world_iter, data=True):
        #         if edata["stateful"]:
        #             n_world_next = v
        #             break
        #     # Update the iterative world node
        #     if n_world_next is None:
        #         n_world_cost.append(n_world_iter)
        #         break
        #
        #     _n_actuator_iter, _n_actuator_next = None, None
        #     for u, _, edata in G_part.in_edges(n_world_iter, data=True):
        #         if G_part.nodes[u]["kind"] == "armactuator":
        #             _n_actuator_iter = u
        #             break
        #     for u, _, edata in G_part.in_edges(n_world_next, data=True):
        #         if G_part.nodes[u]["kind"] == "armactuator":
        #             _n_actuator_next = u
        #             break
        #     assert _n_actuator_iter is not None and _n_actuator_next is not None, "No actuator found"
        #     if _n_actuator_next != _n_actuator_iter:
        #         n_world_cost.append(n_world_iter)
        #
        #     # Update the iterative world node
        #     n_world_iter = n_world_next
        # assert len(n_world_cost) > 0, "No world nodes found to be used as starting points for cost nodes"
        #
        # def cost_data(ts: float, seq: int, seq_world: int, order=1):
        #     ndata = {
        #         "seq": seq,
        #         "ts_step": ts,
        #         "ts_sent": ts,
        #         "pruned": False,
        #         "super": False,
        #         "stateful": True,
        #         "inputs": {"world": {"input_name": "world", 'window': 1}},
        #         "order": order,  # could be wrong
        #         "position": (ts, order),
        #         "color": "yellow",
        #         "kind": "cost",
        #         "edgecolor": oc.ewheel["yellow"],
        #         "facecolor": oc.fwheel["yellow"],
        #         "alpha": 1.0,
        #     }
        #     edata = {'kind': 'world',
        #              'output': 'world',
        #              'window': 1,
        #              'seq': seq_world,
        #              'ts_sent': ts,
        #              'ts_recv': ts,
        #              'stateful': False,
        #              'pruned': False,
        #              'color': '#212529',
        #              'linestyle': '-',
        #              'alpha': 1.0
        #              }
        #     return ndata, edata
        #
        # # Plumb through the cost nodes
        # for i, n_wc in enumerate(n_world_cost):
        #     ts, seq = G_part.nodes[n_wc]["ts_step"], G_part.nodes[n_wc]["seq"]
        #     ndata, edata = cost_data(ts=ts, seq=i, seq_world=seq)
        #     G_part.add_node(f"cost_{i}", **ndata)
        #     G_part.add_edge(n_wc, f"cost_{i}", **edata)
        #     if i > 0:
        #         edata_state = edata.copy()
        #         edata_state.update({'kind': 'cost', "stateful": True, "seq": i-1})
        #         G_part.add_edge(f"cost_{i-1}", f"cost_{i}", **edata_state)

        # Get the first node of every kind
        # G_relabel = calibrate_graph(G_part.copy(), ts_offset=ts_start, relabel_nodes=False)

        # Create graphs
        # relabel = {n: f"{data['kind']}_{data['seq']}" for n, data in G_relabel.nodes(data=True)}
        # G_relabel = nx.relabel_nodes(G_relabel, relabel, copy=False)
        # n_mpc = nx.descendants(G_relabel, relabel[n_plan])
        # n_mpc = set(n_mpc).union({relabel[n_plan]})
        # n_est = set(G_relabel.nodes).difference(n_mpc)
        # G_mpc = calibrate_graph(G_relabel.subgraph(n_mpc).copy(), relabel_nodes=True)
        # G_est = calibrate_graph(G_relabel.subgraph(n_est).copy(), relabel_nodes=True)
        # Gs[n_plan] = {"partition": G_part, "calibrated": G_relabel, "mpc": G_mpc, "est": G_est}
        Gs[n_plan] = G_part

        print(f"Partitioned planner node {n_plan}: {G_part.number_of_nodes()} nodes, {G_part.number_of_edges()} edges") if verbose else None
        if False:
            # for n in Gs[n_plan]["mpc"].nodes():
            #     Gs[n_plan]["part"].nodes[n]["pruned"] = True
            #     Gs[n_plan]["part"].nodes[n]["edgecolor"] = ecolor.mpc
            #     Gs[n_plan]["part"].nodes[n]["facecolor"] = fcolor.mpc
            # for n in Gs[n_plan]["est"].nodes():
            #     Gs[n_plan]["part"].nodes[n]["pruned"] = True
            #     Gs[n_plan]["part"].nodes[n]["edgecolor"] = ecolor.est
            #     Gs[n_plan]["part"].nodes[n]["facecolor"] = fcolor.est
            # _ = vx300s.show_computation_graph(G_relabel, root=None, xmax=None, draw_pruned=True)

            # Key nodes used to identify the partition
            nodes = [n_plan, n_box, n_arm, n_world_box, n_world_arm, n_world, n_actuator, n_controller, n_prev_plan]
            for n in nodes:
                G.nodes[n]["pruned"] = True
                G.nodes[n]["edgecolor"] = ecolor.selected
                G.nodes[n]["facecolor"] = fcolor.selected
            _ = vx300s.show_computation_graph(G_part, root=None, xmax=2.0, draw_pruned=True)
    return Gs


def cost_data(ts: float, seq: int, seq_world: int, order=1):
    ndata = {
        "seq": seq,
        "ts_step": ts,
        "ts_sent": ts,
        "pruned": False,
        "super": False,
        "stateful": True,
        "inputs": {"world": {"input_name": "world", 'window': 1}},
        "order": order,  # could be wrong
        "position": (ts, order),
        "color": "yellow",
        "kind": "cost",
        "edgecolor": oc.ewheel["yellow"],
        "facecolor": oc.fwheel["yellow"],
        "alpha": 1.0,
    }
    edata = {'kind': 'world',
             'output': 'world',
             'window': 1,
             'seq': seq_world,
             'ts_sent': ts,
             'ts_recv': ts,
             'stateful': False,
             'pruned': False,
             'color': '#212529',
             'linestyle': '-',
             'alpha': 1.0
             }
    return ndata, edata


def connect_cost(G, n_world, seq):
    ts_world, seq_world = G.nodes[n_world]["ts_step"], G.nodes[n_world]["seq"]
    ndata, edata = cost_data(ts=ts_world, seq=seq, seq_world=seq_world)
    G.add_node(f"cost_{seq}", **ndata)
    G.add_edge(n_world, f"cost_{seq}", **edata)
    if seq > 0:
        edata_state = edata.copy()
        edata_state.update({'kind': 'cost', "stateful": True, "seq": seq - 1})
        G.add_edge(f"cost_{seq - 1}", f"cost_{seq}", **edata_state)


def partition_list(base_list, T):
    """
    Partitions a list into T equal consecutive parts.
    If len(base_list)=N is not perfectly divisible by T, distributes the deficit equally
    over the partitions, preferring to add them to the first ones.

    Args:
    base_list (int): The  list.
    T (int): The number of partitions.

    Returns:
    List[List[int]]: A list of partitions.
    """
    N = len(base_list)

    # Calculate the base size of each partition and the number of leftovers
    partition_size, leftovers = divmod(N, T)

    # Initialize partitions
    partitions = []
    start_index = 0

    for i in range(T):
        # Determine the end index of the current partition
        # Add an extra element to the later partitions if there are leftovers
        end_index = start_index + partition_size + (1 if i < leftovers else 0)

        # Slice the partition and add it to the partitions list
        partitions.append(base_list[start_index:end_index])

        # Update the start index for the next partition
        start_index = end_index

    return partitions


def split_partitions_into_estimator_mpc_partitions(Gs: Dict[str, nx.DiGraph], num_cost_est=9, num_cost_mpc=11):
    Gs_est = {}
    Gs_mpc = {}
    # 1. Separate the partitioned graph into estimator and MPC subgraphs.
    # 2. Recalibrate sequence numbers and time steps for all nodes in the partition relative to the current planner node.
    # 3. Relabel nodes in the partition for clarity and consistency.
    # 4. Identify and process world nodes for connecting cost nodes.
    # 5. Integrate cost nodes into the graph, with appropriate node and edge data.
    for n_plan, G in Gs.items():
        # Split into estimator and MPC partitions
        n_mpc = nx.descendants(G, n_plan)
        n_mpc = set(n_mpc).union({n_plan})
        n_est = set(G.nodes).difference(n_mpc)
        G_mpc = G.subgraph(n_mpc).copy()
        G_est = G.subgraph(n_est).copy()

        # Calibrate the estimator and MPC partitions
        ts_start = G.nodes[n_plan]["ts_step"]
        G_mpc = calibrate_graph(G_mpc, ts_offset=ts_start, relabel_nodes=True)
        G_est = calibrate_graph(G_est, ts_offset=ts_start, relabel_nodes=True)
        Gs_est[n_plan] = G_est
        Gs_mpc[n_plan] = G_mpc

        # Add cost nodes
        for (num_cost, G_split) in zip((num_cost_mpc, num_cost_est), (G_mpc, G_est)):
            # sort world_nodes by seq
            n_worlds = [n for n in G_split.nodes if G_split.nodes[n]["kind"] == "world"]
            n_worlds = sorted(n_worlds, key=lambda n: G_split.nodes[n]["seq"])

            worlds_partitions = partition_list(list(reversed(n_worlds)), num_cost)
            n_connect_to_cost = [l[0] for l in reversed(worlds_partitions)]

            # Add cost nodes
            for i, n_world in enumerate(n_connect_to_cost):
                connect_cost(G_split, n_world, i)

    return Gs_est, Gs_mpc


def make_supergraph(Gs_raw: List[nx.DiGraph], root: str, supergraph_mode: str = "MCS", backtrack: int = 30, progress_bar: bool = False, validate: bool = False, S_init: nx.DiGraph = None):
    # Get all graphs
    Gs = []
    for i, G in enumerate(Gs_raw):
        # Trace root node (not pruned yet)
        G_traced = rex.supergraph.trace_root(G, root=root, seq=-1)

        # Prune unused nodes (not in computation graph of traced root)
        G_traced_pruned = rex.supergraph.prune_graph(G_traced)
        Gs.append(G_traced_pruned)

    if supergraph_mode == "MCS":
        # Define initial supergraph
        if S_init is None:
            S_init, _ = sg.as_supergraph(Gs[0], leaf_kind=root, sort=[f"{root}_0"])

        # Run evaluation
        S, S_init_to_S, Gs_monomorphism = sg.grow_supergraph(
            Gs,
            S_init,
            root,
            combination_mode="linear",
            backtrack=backtrack,
            progress_fn=None,
            progress_bar=progress_bar,
            validate=validate,
        )
    elif supergraph_mode == "topological":
        from supergraph.evaluate import baselines_S

        S, _ = baselines_S(Gs, root)
        S_init_to_S = {n: n for n in S.nodes()}
        Gs_monomorphism = sg.evaluate_supergraph(Gs, S, progress_bar=progress_bar)
    elif supergraph_mode == "generational":
        from supergraph.evaluate import baselines_S

        _, S = baselines_S(Gs, root)
        S_init_to_S = {n: n for n in S.nodes()}
        Gs_monomorphism = sg.evaluate_supergraph(Gs, S, progress_bar=progress_bar)
    else:
        raise ValueError(f"Unknown supergraph mode '{supergraph_mode}'.")
    return S, S_init_to_S, Gs_raw, Gs_monomorphism


if __name__ == "__main__":
    PATH_TO_LOG = "/home/r2ci/rex/logs/vx300s_3winplanner_vx300s_2023-11-23-1722"
    # PATH_TO_LOG = "/home/r2ci/rex/logs/real_0.35umax_vx300s_2023-11-20-1759"

    # Load proto experiment record
    record_pre = log_pb2.ExperimentRecord()
    with open(f"{PATH_TO_LOG}/record_pre.pb", "rb") as f:
        record_pre.ParseFromString(f.read())

    G = create_graph(record_pre.episode[-1])
    G_cp = G.copy(as_view=False)
    if False:
        fig_cg, _ = vx300s.show_computation_graph(G, root="planner", xmax=2.5)

    Gs = envs.vx300s.planner.rex.make_partitions(G_cp, dt_horizon=0.5, progress_bar=True)
    num_cost_est, num_cost_mpc = 8, 8
    Gs_est, Gs_mpc = envs.vx300s.planner.rex.split_partitions_into_estimator_mpc_partitions(Gs, num_cost_est=num_cost_est, num_cost_mpc=num_cost_mpc)

    if False:
        _ = vx300s.show_computation_graph(Gs["planner_6"], root=None, xmax=None, draw_pruned=True)
        _ = vx300s.show_computation_graph(Gs_est["planner_6"], root=None, xmax=None, draw_pruned=True)
        _ = vx300s.show_computation_graph(Gs_mpc["planner_6"], root=None, xmax=None, draw_pruned=True)

    # Get supergraphs
    for num_cost, Gs_split in zip((num_cost_est, num_cost_mpc), (Gs_est, Gs_mpc)):
        for mode in ("MCS",):
            print(f"Mode: {mode}")
            S, S_init_to_S, Gs_raw, Gs_monomorphism = make_supergraph(list(Gs_split.values()), root="cost", supergraph_mode=mode, backtrack=30, progress_bar=True, validate=False)

            # Get timings
            timings = []
            for i, (G, G_monomorphism) in enumerate(zip(list(Gs_split.values()), Gs_monomorphism)):
                t = rex.supergraph.get_timings(S, G, G_monomorphism, num_root_steps=num_cost, root="cost")
                timings.append(t)

            # Stack timings
            import numpy as onp
            timings = jax.tree_util.tree_map(lambda *args: onp.stack(args, axis=0), *timings)

    # todo: integerate into mpc controller
    # todo: Pruned incoming edges should have seq numbers of -1, -2, etc... --> can buffers handle this?
    print("WAIT")

    # for n, G in G_mpc.items():
    #     _, _ = vx300s.show_computation_graph(G, root=None, xmax=5.0, ax=ax, draw_pruned=True)


