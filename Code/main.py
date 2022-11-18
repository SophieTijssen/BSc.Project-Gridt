import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import random

from mesa.batchrunner import batch_run
from model import GranovetterModel


def plotDirectedGraph(model):
  G = model.G
  pos = nx.spring_layout(G, seed=model.seed)

  node_sizes = [20] * len(G)
  M = G.number_of_edges()
  edge_colors = range(2, M + 2)
  edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
  cmap = plt.cm.plasma

  nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
  edges = nx.draw_networkx_edges(
    G,
    pos,
    node_size=node_sizes,
    arrowstyle="->",
    arrowsize=10,
    edge_color=edge_colors,
    edge_cmap=cmap,
    width=2,
  )
  # set alpha value for each edge
  for i in range(M):
    edges[i].set_alpha(edge_alphas[i])

  # pc = mpl.collections.PatchCollection(edges, cmap=cmap)
  # pc.set_array(edge_colors)

  ax = plt.gca()
  ax.set_axis_off()
  # plt.colorbar(pc, ax=ax)
  plt.show()


def singleRun():
  model = GranovetterModel(num_of_nodes=100, mu=0.25, sigma=0.1, in_degree=3)
  plotDirectedGraph(model)

  while model.running and model.schedule.steps < 100:
    model.step()

  print('The Granovetter Model ran for {} steps'.format(model.schedule.steps))
  model.datacollector.collect(model)

  agent_out = model.datacollector.get_agent_vars_dataframe()

  agent_out_df = pd.DataFrame(data=agent_out)
  random_agent = random.choice(range(len(agent_out_df.xs(0))))
  agent_results = agent_out_df.xs(random_agent, level=1)

  agent_results.plot()
  plt.show()

  model_out = model.datacollector.get_model_vars_dataframe()
  model_out.plot()
  plt.show()


def batchRun():
  params = {
    "num_of_nodes": 100,
    "mu": 0.25,
    "sigma": np.linspace(0.0, 1.0, 11),
    "in_degree": 3
  }

  results = batch_run(
    GranovetterModel,
    parameters=params,
    iterations=10,
    max_steps=100,
    number_processes=None,
    data_collection_period=-1,
    display_progress=True
  )

  results_df = pd.DataFrame(results)
  print(results_df.head())

  # This doesn't do anything
  results_df.groupby(by=["RunId", "iteration", "Step"]).median()
  print(results_df.head())


def main():
  single = True
  if single:
    # Single Run
    singleRun()

  else:
    # Batch Run
    batchRun()


if __name__ == '__main__':
  main()
