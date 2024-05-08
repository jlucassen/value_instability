import networkx as nx
from itertools import combinations

# GPT written, unverified
def is_acyclic(G):
    """ Check if the graph G is acyclic """
    try:
        nx.find_cycle(G)
        return False
    except nx.NetworkXNoCycle:
        return True

# GPT written, unverified
def find_optimal_fas(G):
    """ Find the optimal feedback arc set for the graph G """
    edges = list(G.edges())
    num_edges = len(edges)
    optimal_fas = edges  # Start with all edges as the worst-case scenario
    min_size = num_edges  # Initialize with the maximum number of edges

    # Iterate over all possible subsets of edges
    for i in range(1, num_edges + 1):
        for subset in combinations(edges, i):
            # Create a copy of the graph without the subset of edges
            H = G.copy()
            H.remove_edges_from(subset)

            # Check if the resulting graph is acyclic
            if is_acyclic(H):
                # If the subset results in an acyclic graph and is smaller than the found sets, update optimal_fas
                if i < min_size:
                    optimal_fas = subset
                    min_size = i
                    # Early exit if we find a set with only one edge (can't get smaller)
                    if min_size == 1:
                        return optimal_fas

    return optimal_fas

# Example usage
# G = nx.DiGraph()
# G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (2, 0)])

# fas = find_optimal_fas(G)
# print("Optimal Feedback Arc Set:", fas)
