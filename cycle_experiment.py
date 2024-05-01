# %%
import numpy as np

from llm import LLM

# %%
llm = LLM("gpt-3.5-turbo")

# %%
problem_setting = """You are an AI agent playing a game of Dominion.
Your deck contains 7 Coppers and 3 Estates.
Your hand contains 3 Coppers and 2 Estates.
You have 1 buy and 1 action."""

cards = [
    'Copper',
    'Silver',
    'Gold',
    'Estate',
    'Duchy',
    'Province',
    # "Smithy",
    # "Council Room",
    # "Laboratory",
    # "Market",
    # "Artisan",
    # "Cellar",
    # "Merchant",
    # "Workshop",
    # "Gardens",
    # "Militia",
]

# %%
n = 3
preference_matrix = np.zeros([len(cards), len(cards)])
for card1 in cards:
    for card2 in cards:
        if card1 != card2:
            for _ in range(n):
                prompt = f"{problem_setting}\n\nIf you could add a card to your deck for free, would you prefer a {card1} over a {card2}?"
                completion = llm.choice_completion(prompt, ["Yes", "No"])
                preference_matrix[cards.index(card1), cards.index(card2)] += 1 if completion == "Yes" else 0
preference_matrix /= n
preference_matrix = preference_matrix.round(0)
# %%
def find_all_cycles(adjacency):
    def dfs_cycle(v, graph, path, visited, cycles):
        visited[v] = True
        path.append(v)
        for neighbor, is_edge in enumerate(graph[v]):
            if is_edge:
                if neighbor in path:
                    # Find cycle and add it to the list
                    cycle_index = path.index(neighbor)
                    cycles.add(tuple(path[cycle_index:]))  # Add as a tuple to avoid duplicate cycles
                elif not visited[neighbor]:
                    dfs_cycle(neighbor, graph, path, visited, cycles)
        path.pop()
    n = adjacency.shape[0]
    visited = [False] * n
    cycles = set()

    for node in range(n):
        if not visited[node]:
            dfs_cycle(node, adjacency, [], visited, cycles)
    
    return [list(cycle) for cycle in cycles]

cycles = find_all_cycles(preference_matrix)
print("All cycles in the graph:", cycles)
# %%
m = 3
rates = np.zeros([len(cycles)])
for cycle in cycles:
    prompt = f"""{problem_setting}
    Another instance of you was asked to give preferences between cards, and a cycle was found.
    You said you preferred:
    {" > ".join([cards[i] for i in cycle + [cycle[0]]])}

    Do you want to keep your preferences the same and retain this cycle?"""
    for _ in range(m):
        completion = llm.choice_completion(prompt, ["Yes", "No"])
        rates[cycles.index(cycle)] += 1 if completion == "Yes" else 0
rates /= m
print(rates)
# %%
