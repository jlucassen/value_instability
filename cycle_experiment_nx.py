"""
Refactored into NX by Copilot (haven't verified if everything works the same)
"""

# %%
import numpy as np
import networkx as nx
from llm import LLM
from fas import find_optimal_fas
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    'Province', # six basics
    "Smithy",
    "Council Room",
    "Laboratory",
    "Market",
    "Artisan", # five kingdom cards
    # "Cellar",
    # "Merchant",
    # "Workshop",
    # "Gardens",
    # "Militia", # full ten kingdom cards
]

# %%
n = 1 # greedy sampling with t=0
preference_matrix = np.zeros([len(cards), len(cards)])
for card1 in tqdm(cards):
    for card2 in cards:
        if card1 != card2:
            for _ in range(n):
                prompt = f"{problem_setting}\n\nIf you could add a card to your deck for free, would you prefer a {card1} over a {card2}?"
                completion = llm.choice_completion(prompt, ["Yes", "No"])
                preference_matrix[cards.index(card1), cards.index(card2)] += 1 if completion == "Yes" else 0
preference_matrix /= n
preference_matrix = preference_matrix.round(0)

# Convert preference matrix to a directed graph
G = nx.DiGraph(preference_matrix)

# %%
def find_all_cycles(graph):
    cycles = nx.simple_cycles(graph)
    return [cycle for cycle in cycles]

cycles = find_all_cycles(G)
print("All cycles in the graph:", cycles)
# %%
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

m = 1 # because using t=0
rates = np.zeros([len(cycles)])
lock = Lock()
pbar = tqdm(total=m*len(cycles))
def process_cycle(cycle_index):
    cycle = cycles[cycle_index]
    prompt = f"""{problem_setting}
    Another instance of you was asked to give preferences between cards, and a cycle was found.
    You said you preferred:
    {" > ".join([cards[i] for i in cycle + [cycle[0]]])}

    Do you want to keep your preferences the same and retain this cycle?"""
    for _ in range(m):
        completion = llm.choice_completion(prompt, ["Yes", "No"])
        rates[cycle_index] += 1 if completion == "Yes" else 0
        with lock:
            pbar.update(1)
 #%%
with ThreadPoolExecutor(max_workers=100) as executor:
    executor.map(process_cycle, range(len(cycles)))
rates /= m
print(rates)
rates = rates.round(0)
print(rates)
# %%
# optimal_fas = find_optimal_fas(G)
# print("Optimal Feedback Arc Set:", optimal_fas)
# size_of_optimal_fas = len(optimal_fas)
# print("Size of optimal_fas:", size_of_optimal_fas)
# %%

print(f"number of cycles: {len(cycles)}")
print(f"smallest_cycle_size: {min([len(cycle) for cycle in cycles])}")
print(f"max_cycle_size: {max([len(cycle) for cycle in cycles])}")
print(f"average_cycle_size: {np.mean([len(cycle) for cycle in cycles])}")
print(f"cycle keep rate: {np.mean(rates)}")
# %%
import pandas as pd 
df = pd.DataFrame({'length':[len(c) for c in cycles], 'rate':rates})
plt.plot(df.groupby('length').mean())
plt.ylim(0,1)
# %%
plt.plot(df.groupby('length').count())
# %%
import random

def process_cycle_dummy(ct, cyc_len):
    out = 0
    prompt = f"""{problem_setting}
    Another instance of you was asked to give preferences between cards, and a cycle was found.
    You said you preferred:
    {" > ".join([random.choice(cards) for _ in range(cyc_len)])}

    Do you want to keep your preferences the same and retain this cycle?"""
    for _ in tqdm(list(range(ct))):
        completion = llm.choice_completion(prompt, ["Yes", "No"])
        out += 1 if completion == "Yes" else 0
    return out/ct

print(process_cycle_dummy(50, 3))
print(process_cycle_dummy(50, 9))
# %%
