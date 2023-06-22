"""
This module is based on the MCS algorithm from the paper:
"DotMotif: an open-source tool for connectome subgraph isomorphism search and graph queries"

The implementation is heavily inspired by, and partly copied from, the implementation in:
https://github.com/aplbrain/grandiso-networkx

The main difference is that this implementation is designed to find the maximum common (monomorphic) subgraph between two graphs,
when no exact match exists.

According to the original implementation:
    In this process, we consider the following operations to be fast:

    - Get degree of node
    - Get downstream targets of node
    - Get attributes on a node

    These operations are medium:

    - Get upstream sources of node
    - Get degrees of all nodes in host graph
    - Get nodes with a certain attribute

    These operations are slow:
    - Get edges where nodes have a certain attribute
        - But you can do the same with get-downstream-targets and filter an
          attribute search.
"""

from typing import Dict, Generator, Hashable, List, Optional, Union, Tuple
from inspect import isclass
import itertools
from functools import lru_cache

import networkx as nx


from collections import deque
import queue
from enum import Enum


try:
    SimpleQueue = queue.SimpleQueue
except:
    SimpleQueue = queue.Queue


class QueuePolicy(Enum):
    """
    An Enum for queue pop policy.

    DEPTHFIRST policy is identified with pushing and popping on the same side
    of the queue. BREADTHFIRST is identified with pushing and popping from the
    opposite side of the queue.

    """

    DEPTHFIRST = 1
    BREADTHFIRST = 2


class Deque:
    """
    A double-ended queue implementation.

    """

    def __init__(self, policy: QueuePolicy = QueuePolicy.DEPTHFIRST):
        """
        Create a new double-ended queue.

        Arguments:
            policy (QueuePolicy): The policy to use when adding and removing
                elements from this queue. Defaults to depth-first.

        Returns:
            None

        """
        self._dq = deque()
        if policy == QueuePolicy.DEPTHFIRST:
            self._put = self._dq.append
            self._get = self._dq.popleft
        elif policy == QueuePolicy.BREADTHFIRST:
            self._get = self._dq.pop
            self._put = self._dq.append

    def put(self, *args, **kwargs):
        """
        Put a new element into the queue.

        Arguments:
            element: The element to add to the queue

        Returns:
            None

        """
        return self._put(args[0])

    def get(self, *args, **kwargs):
        """
        Get a new item from the queue.

        Arguments:
            None

        Returns:
            Any: The element popped from the queue

        """
        return self._get()

    def empty(self):
        """
        Returns True if the queue is empty.

        Arguments:
            None

        Returns:
            bool: True if the queue has nothing more to pop

        """
        return False if self._dq else True


@lru_cache()
def _is_node_attr_match(motif_node_id: str, host_node_id: str, motif: nx.Graph, host: nx.Graph) -> bool:
    """
    Check if a node in the host graph matches the attributes in the motif.

    Arguments:
        motif_node_id (str): The motif node ID
        host_node_id (str): The host graph ID
        motif (nx.Graph): The motif graph
        host (nx.Graph): The host graph

    Returns:
        bool: True if the host node matches the attributes in the motif

    """
    motif_node = motif.nodes[motif_node_id]
    host_node = host.nodes[host_node_id]
    return (
        host_node["name"] == motif_node["name"]
        and host_node["sub_longest_path_length"] >= motif_node["sub_longest_path_length"]
    )


@lru_cache()
def _is_node_structural_match(motif_node_id: str, host_node_id: str, motif: nx.Graph, host: nx.Graph) -> bool:
    """
    Check if the motif node here is a valid structural match.

    Specifically, this requires that a host node has at least the degree as the
    motif node.

    Arguments:
        motif_node_id (str): The motif node ID
        host_node_id (str): The host graph ID
        motif (nx.Graph): The motif graph
        host (nx.Graph): The host graph

    Returns:
        bool: True if the motif node maps to this host node

    """
    return host.degree(host_node_id) >= motif.degree(motif_node_id)


@lru_cache()
def _is_edge_attr_match(
    motif_edge_id: Tuple[str, str],
    host_edge_id: Tuple[str, str],
    motif: nx.Graph,
    host: nx.Graph,
) -> bool:
    """
    Check if an edge in the host graph matches the attributes in the motif.

    Arguments:
        motif_edge_id (str): The motif edge ID
        host_edge_id (str): The host edge ID
        motif (nx.Graph): The motif graph
        host (nx.Graph): The host graph

    Returns:
        bool: True (Always)

    """

    return True


def get_next_backbone_candidates(
    backbone: dict,
    motif: nx.Graph,
    host: nx.Graph,
    interestingness: dict,
    next_node: str = None,
    directed: bool = True,
    is_node_structural_match=_is_node_structural_match,
    is_node_attr_match=_is_node_attr_match,
    is_edge_attr_match=_is_edge_attr_match,
    isomorphisms_only: bool = False,
) -> List[Dict[str, str]]:
    """
    Get a list of candidate node assignments for the next "step" of this map.

    Arguments:
        backbone (dict): Mapping of motif node IDs to one set of host graph IDs
        motif (Graph): A graph representation of the motif
        host (Graph): The host graph, complete
        interestingness (dict): A mapping of motif node IDs to interestingness
        next_node (str: None): Optional suggestion for the next node to assign
        directed (bool: True): Whether host and motif are both directed
        isomorphisms_only (bool: False): If true, only isomorphisms will be
            returned (instead of all monomorphisms)

    Returns:
        List[dict]: A new list of mappings with one additional element mapped

    """
    # NOTE!: is next_node ever != None? I don't think so, so we can remove it

    # Get a list of the "exploration front" of the motif -- nodes that are not
    # yet assigned in the backbone but are connected to at least one assigned
    # node in the backbone.

    # For example, in the motif A -> B -> C, if A is already assigned, then the
    # front is [B] (c is not included because it has not connection to any
    # assigned node).

    # We should prefer nodes that are connected to multiple assigned backbone
    # nodes, because these will filter more rapidly to a smaller set.

    # First check if the backbone is empty. If so, we should choose the most
    # interesting node to start with:
    if next_node is None and len(backbone) == 0:
        # This is the starting-case, where we have NO backbone nodes set yet.
        # NOTE: We iterate over all nodes, not just the most interesting one.
        #       This is because we want to return ALL possible candidates for,
        #       instead of exiting if there exists a node for which no match can be found
        # todo: only calculate this once.
        # todo: put this in back of queue (i.e. depth first search).
        candidates = []
        # NOTE! We assume that the interestingness dict is sorted by interestingness (i.e. max first)
        for next_node in interestingness.keys():
            # Let's return ALL possible node choices for this next_node. To do this
            # without being an insane person, let's filter on max degree in host:
            for n in host.nodes:
                node_attr_match = is_node_attr_match(next_node, n, motif, host)
                node_struct_match = is_node_structural_match(next_node, n, motif, host)
                if node_attr_match and node_struct_match:
                    candidates.append({next_node: n})
        return candidates
        # return [
        #     {next_node: n}
        #     for n in host.nodes
        #     if is_node_attr_match(next_node, n, motif, host)
        #     and is_node_structural_match(next_node, n, motif, host)
        # ]

    else:
        _nodes_with_greatest_backbone_count: List[str] = []
        _greatest_backbone_count = 0
        for motif_node_id in motif.nodes:
            if motif_node_id in backbone:
                continue
            # How many connections to existing backbone?
            # Note that this number is certainly greater than or equal to 1,
            # since a value of 0 would imply that the backbone dict is empty
            # (which we have already handled) or that the motif has more than
            # one connected component, which we check for at prep-time.
            if directed:
                motif_backbone_connections_count = sum(
                    [
                        1
                        for v in list(
                            set(motif.adj[motif_node_id]).union(  # NOTE: adj=adjacent (i.e. neighboring nodes)
                                set(motif.pred[motif_node_id])  # NOTE: pred=predecessors (i.e. nodes that point to this node)
                            )
                        )
                        if v in backbone
                    ]
                )
            else:
                motif_backbone_connections_count = sum([1 for v in motif.adj[motif_node_id] if v in backbone])
            # If this is the most highly connected node visited so far, then
            # set it as the next node to explore:
            if motif_backbone_connections_count > _greatest_backbone_count:
                _nodes_with_greatest_backbone_count.append(motif_node_id)
        # NOTE: if _nodes_with_greatest_backbone_count is empty, then the backbone fully covers a disconnected subgraph motif.
        if len(_nodes_with_greatest_backbone_count) == 0:
            # Add all possible starting nodes (that have a suitable host_node match AND are not yet in the backbone)
            # to the list of candidates that are not yet in the backbone.
            bb_next_nodes = list(backbone.keys())
            bb_n = list(backbone.values())
            candidates = []
            for next_node in interestingness.keys():
                # Skip if next_node is already in the backbone
                if next_node in bb_next_nodes:
                    continue
                for n in host.nodes:
                    # Skip if n is already in the backbone
                    if n in bb_n:
                        continue
                    node_attr_match = is_node_attr_match(next_node, n, motif, host)
                    node_struct_match = is_node_structural_match(next_node, n, motif, host)
                    if node_attr_match and node_struct_match:
                        candidates.append({next_node: n, **backbone})
            # Create backbones that add all possible candidates for the next node
            return candidates
        else:
            # Now we have _node_with_greatest_backbone_count as the best candidate
            # for `next_node`.
            next_node = max(
                _nodes_with_greatest_backbone_count,
                key=lambda node: interestingness.get(node, 0.0),
            )

    # Now we have a node `next_node` which we know is connected to the current
    # backbone. Get all edges between `next_node` and nodes in the backbone,
    # and verify that they exist in the host graph:
    # `required_edges` has the form (prev, self, next), with non-values filled
    # with None. That way we can easily remember and store the roles of the
    # node IDs in the next step.
    required_edges = []
    for other in list(motif.adj[next_node]):
        if other in backbone:
            # edge is (next_node, other)
            required_edges.append((None, next_node, other))
    if directed:
        for other in list(motif.pred[next_node]):
            if other in backbone:
                # edge is (other, next_node)
                required_edges.append((other, next_node, None))

    # `required_edges` now contains a list of all edges that exist in the motif
    # graph, and we must find candidate nodes that have such edges in the host.

    candidate_nodes = []

    # In the worst-case, `required_edges` has length == 1. This is the worst
    # case because it means that ALL edges from/to `other` are valid options.
    if len(required_edges) == 1:
        # :(
        (source, _, target) = required_edges[0]
        if directed:
            if source is not None:
                # this is a "from" edge:
                candidate_nodes = list(host.adj[backbone[source]])
            elif target is not None:
                # this is a "from" edge:
                candidate_nodes = list(host.pred[backbone[target]])
        else:
            candidate_nodes = list(host.adj[backbone[target]])
        # Thus, all candidates for motif ID `$next_node` are stored in the
        # candidate_nodes list.

    elif len(required_edges) > 1:
        # This is neato :) It means that there are multiple edges in the host
        # graph that we can use to downselect the number of candidate nodes.
        candidate_nodes_set = set()
        for source, _, target in required_edges:
            if directed:
                if source is not None:
                    # this is a "from" edge:
                    candidate_nodes_from_this_edge = host.adj[backbone[source]]
                # elif target is not None:
                else:  # target is not None:
                    # this is a "from" edge:
                    candidate_nodes_from_this_edge = host.pred[backbone[target]]
                # else:
                #     raise AssertionError("Encountered an impossible condition: At least one of source or target must be defined.")
            else:
                candidate_nodes_from_this_edge = host.adj[backbone[target]]
            if len(candidate_nodes_set) == 0:
                # This is the first edge we're checking, so set the candidate
                # nodes set to ALL possible candidates.
                candidate_nodes_set.update(candidate_nodes_from_this_edge)
            else:
                candidate_nodes_set = candidate_nodes_set.intersection(candidate_nodes_from_this_edge)
        candidate_nodes = list(candidate_nodes_set)

    elif len(required_edges) == 0:
        # Somehow you found a node that doesn't have any edges. This is bad.
        raise ValueError(
            f"Somehow you found a motif node {next_node} that doesn't have "
            + "any motif-graph edges. This is bad. (Did you maybe pass an "
            + "empty backbone to this function?)"
        )

    tentative_results = [
        {**backbone, next_node: c}
        for c in candidate_nodes
        if c not in backbone.values()
        and is_node_attr_match(next_node, c, motif, host)
        and is_node_structural_match(next_node, c, motif, host)
    ]

    # One last filtering step here. This is to catch the cases where you have
    # successfully mapped each node, and the final node has some valid
    # candidate_nodes (and therefore `tentative_results`).
    # This is important: We must now check that for the assigned nodes, all
    # edges between them DO exist in the host graph. Otherwise, when we check
    # in find_motifs that len(motif) == len(mapping), we will discover that the
    # mapping is "complete" even though we haven't yet checked it at all.

    monomorphism_candidates = []

    for mapping in tentative_results:
        if len(mapping) == len(motif):
            if all(
                [
                    host.has_edge(mapping[motif_u], mapping[motif_v])
                    and is_edge_attr_match(
                        (motif_u, motif_v),
                        (mapping[motif_u], mapping[motif_v]),
                        motif,
                        host,
                    )
                    for motif_u, motif_v in motif.edges
                ]
            ):
                # This is a "complete" match!
                monomorphism_candidates.append(mapping)
        else:
            # This is a partial match, so we'll continue building.
            monomorphism_candidates.append(mapping)

    if not isomorphisms_only:
        return monomorphism_candidates

    # Additionally, if isomorphisms_only == True, we can use this opportunity
    # to confirm that no spurious edges exist in the induced subgraph:
    isomorphism_candidates = []
    for result in monomorphism_candidates:
        for motif_u, motif_v in itertools.product(result.keys(), result.keys()):
            # if the motif has this edge, then it doesn't rule any of the
            # above results out as an isomorphism.
            # if the motif does NOT have the edge, then NO RESULT may have
            # the equivalent edge in the host graph:
            if not motif.has_edge(motif_u, motif_v) and host.has_edge(result[motif_u], result[motif_v]):
                # this is a violation.
                break
        else:
            isomorphism_candidates.append(result)
    return isomorphism_candidates


def uniform_node_interestingness(motif: nx.Graph) -> dict:
    """
    Sort the nodes in a motif by their interestingness.

    Most interesting nodes are defined to be those that most rapidly filter the
    list of nodes down to a smaller set.

    """
    return {n: 1 for n in motif.nodes}


# def find_motifs_iter(
#     motif: nx.Graph,
#     host: nx.Graph,
#     interestingness: dict = None,
#     directed: bool = None,
#     queue_=SimpleQueue,
#     isomorphisms_only: bool = False,
#     hints: List[Dict[Hashable, Hashable]] = None,
#     is_node_structural_match=_is_node_structural_match,
#     is_node_attr_match=_is_node_attr_match,
#     is_edge_attr_match=_is_edge_attr_match,
# ) -> Generator[dict, None, None]:
#     """
#     Yield mappings from motif node IDs to host graph IDs.
#
#     Results are of the form:
#
#     ```
#     {motif_id: host_id, ...}
#     ```
#
#     Arguments:
#         motif (nx.DiGraph): The motif graph (needle) to search for
#         host (nx.DiGraph): The host graph (haystack) to search within
#         interestingness (dict: None): A map of each node in `motif` to a float
#             number that indicates an ordinality in which to address each node
#         directed (bool: None): Whether direction should be considered during
#             search. If omitted, this will be based upon the motif directedness.
#         queue_ (queue.SimpleQueue): What kind of queue to use.
#         hints (dict): A dictionary of initial starting mappings. By default,
#             searches for all instances. You can constrain a node by passing a
#             list with a single dict item: `[{motifId: hostId}]`.
#         isomorphisms_only (bool: False): Whether to return isomorphisms (the
#             default is monomorphisms).
#
#     Returns:
#         Generator[dict, None, None]
#
#     """
#     # Prepare uniform interestingness if none is provided.
#     interestingness = interestingness or uniform_node_interestingness(motif)
#
#     # Make sure all nodes are included in the interestingness dict
#     interestingness = {n: interestingness.get(n, 0.) for n in motif.nodes}
#
#     # Sort the interestingness dict by value:
#     # todo: prepare possible structural matches beforehand.
#     interestingness = {k: v for k, v in sorted(interestingness.items(), reverse=True, key=lambda item: item[1])}
#
#     if directed is None:
#         # guess directedness from motif
#         if isinstance(motif, nx.DiGraph):
#             # This will be a directed query.
#             directed = True
#         else:
#             directed = False
#
#     q = queue_() if isclass(queue_) else queue_
#
#     # Kick off the queue with an empty candidate:
#     if hints is None or hints == []:
#         q.put({})
#     else:
#         for hint in hints:
#             q.put(hint)
#
#     while not q.empty():
#         # NOTE! new_backbone: dict --> previous (failed) candidate
#         new_backbone = q.get()
#         # NOTE! next_candidate_backbones: List[dict]
#         next_candidate_backbones = get_next_backbone_candidates(
#             new_backbone,
#             motif,
#             host,
#             interestingness,
#             directed=directed,
#             isomorphisms_only=isomorphisms_only,
#             is_node_structural_match=is_node_structural_match,
#             is_node_attr_match=is_node_attr_match,
#             is_edge_attr_match=is_edge_attr_match,
#         )
#
#         for candidate in next_candidate_backbones:
#             # NOTE! Cache len(candidate)?
#             # NOTE! Per disconnected graph in motif, we need to find the largest connected component & try to merge them.
#             # NOTE! Viewing the merged graph as the largest monomorphic ignores pruning nodes and rechecking.
#             # NOTE! Test if this algorithm works for disconnected graphs.
#             # NOTE! Can we pre-process motif graph,
#             #       to encode dependency chain when an intermediate node is removed?
#             if len(candidate) == len(motif):
#                 yield candidate
#             else:
#                 # print(candidate)
#                 # NOTE! If candidate is not monomorphic, don't yield, but try with a new candidate
#                 # NOTE! If candidates are empty, try a new backbone.
#                 q.put(candidate)


def find_largest_motifs(
    motif: nx.Graph,
    host: nx.Graph,
    interestingness: dict = None,
    directed: bool = None,
    queue_=SimpleQueue,
    max_evals: int = None,
    isomorphisms_only: bool = False,
    hints: List[Dict[Hashable, Hashable]] = None,
    is_node_structural_match=_is_node_structural_match,
    is_node_attr_match=_is_node_attr_match,
    is_edge_attr_match=_is_edge_attr_match,
) -> Tuple[int, bool, List[Dict[str, str]]]:
    """
    Yield mappings from motif node IDs to host graph IDs.

    Results are of the form:

    ```
    {motif_id: host_id, ...}
    ```

    Arguments:
        motif (nx.DiGraph): The motif graph (needle) to search for
        host (nx.DiGraph): The host graph (haystack) to search within
        interestingness (dict: None): A map of each node in `motif` to a float
            number that indicates an ordinality in which to address each node
        directed (bool: None): Whether direction should be considered during
            search. If omitted, this will be based upon the motif directedness.
        queue_ (queue.SimpleQueue): What kind of queue to use.
        hints (dict): A dictionary of initial starting mappings. By default,
            searches for all instances. You can constrain a node by passing a
            list with a single dict item: `[{motifId: hostId}]`.
        isomorphisms_only (bool: False): Whether to return isomorphisms (the
            default is monomorphisms).

    Returns:
        Generator[dict, None, None]

    """
    # TODO: Possible optimizations:
    # 1. Prepare possible structural matches beforehand.
    # 2. Prune candidates that are monomorphic with other candidates. (i.e. have already been evaluated or are monomorphic to a subgraph of previous candidates)
    # 3. [DONE] Add new candidates to the queue in order of interestingness.
    # 4. Exclude new candidates for which the max number of potentially addable nodes is less than the number of nodes in the largest candidate.
    # 5. [DONE] Add max number of candidate evaluations, and return the largest candidate when this number is reached.
    # 6. Investigate how to use multi-processing with this algorithm (i.e. share a queue?)
    # 7. Can we use lru_cache to detect candidates that have already been evaluated?
    # 8. Define initial starting mapping to use as hints.
    # 9. Can we determine the max number of nodes on the maximal common (monomorphic) subgraph i.e. stop searching when this number is reached?
    #    This should make use of structural mismatches in both graph (i.e. count of node types).
    interestingness = interestingness or uniform_node_interestingness(motif)

    # Make sure all nodes are included in the interestingness dict
    interestingness = {n: interestingness.get(n, 0.0) for n in motif.nodes}

    # Sort the interestingness dict by value:
    interestingness = {k: v for k, v in sorted(interestingness.items(), reverse=True, key=lambda item: item[1])}

    if directed is None:
        # guess directedness from motif
        if isinstance(motif, nx.DiGraph):
            # This will be a directed query.
            directed = True
        else:
            directed = False

    q = queue_() if isclass(queue_) else queue_

    # Kick off the queue with an empty candidate:
    if hints is None or hints == []:
        q.put({})
    else:
        for hint in hints:
            q.put(hint)

    # Prepare and test all possible candidates
    largest_candidates = []
    largest_candidate_size = 0
    num_evals = 0
    while not q.empty():
        if not (max_evals is None or num_evals < max_evals):
            print(f"Reached max number of candidate evaluations per motif: {max_evals}. Returning largest candidates.")
            break
        num_evals += 1
        # NOTE! new_backbone: dict --> previous (failed) candidate
        new_backbone = q.get()
        next_candidate_backbones = get_next_backbone_candidates(
            new_backbone,
            motif,
            host,
            interestingness,
            directed=directed,
            isomorphisms_only=isomorphisms_only,
            is_node_structural_match=is_node_structural_match,
            is_node_attr_match=is_node_attr_match,
            is_edge_attr_match=is_edge_attr_match,
        )

        for candidate in next_candidate_backbones:
            if len(candidate) > largest_candidate_size:
                largest_candidate_size = len(candidate)
                largest_candidates = [candidate]
            elif len(candidate) == largest_candidate_size:
                largest_candidates.append(candidate)
            if len(candidate) == len(motif):
                return num_evals, True, largest_candidates
            else:
                # print(candidate)
                q.put(candidate)
    return num_evals, False, largest_candidates


# def find_motifs(
#     motif: nx.Graph,
#     host: nx.Graph,
#     *args,
#     count_only: bool = False,
#     limit: int = None,
#     is_node_attr_match=_is_node_attr_match,
#     is_node_structural_match=_is_node_structural_match,
#     is_edge_attr_match=_is_edge_attr_match,
#     **kwargs,
# ) -> Union[int, List[dict]]:
#     """
#     Get a list of mappings from motif node IDs to host graph IDs.
#
#     Results are of the form:
#
#     ```
#     [{motif_id: host_id, ...}]
#     ```
#
#     See grandiso#find_motifs_iter for full argument list.
#
#     Arguments:
#         count_only (bool: False): If True, return only an integer count of the
#             number of motifs, rather than a list of mappings.
#         limit (int: None): A limit to place on the number of returned mappings.
#             The search will terminate once the limit is reached.
#
#
#     Returns:
#         int: If `count_only` is True, return the length of the List.
#         List[dict]: A list of mappings from motif node IDs to host graph IDs
#
#     """
#     results = []
#     results_count = 0
#     for qresult in find_motifs_iter(
#         motif,
#         host,
#         *args,
#         is_node_attr_match=is_node_attr_match,
#         is_node_structural_match=is_node_structural_match,
#         is_edge_attr_match=is_edge_attr_match,
#         **kwargs,
#     ):
#
#         result = qresult
#
#         results_count += 1
#         if limit and results_count >= limit:
#             if count_only:
#                 return results_count
#             else:
#                 # Subtract 1 from results_count because we have not yet
#                 # added the new result to the results list, but we HAVE
#                 # already added +1 to the count.
#                 if limit and (results_count - 1) >= limit:
#                     return results
#         if not count_only:
#             results.append(result)
#
#     if count_only:
#         return results_count
#     return results
