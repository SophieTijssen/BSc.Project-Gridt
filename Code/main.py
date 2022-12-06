import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from mesa.batchrunner import batch_run
from model import GranovetterModel


# https://networkx.org/documentation/stable/auto_examples/drawing/plot_directed.html
from util import *


def getNodeColours(model):
  agent_out = model.datacollector.get_agent_vars_dataframe()
  agent_out_df = pd.DataFrame(data=agent_out)

  maxStep = agent_out_df.index.values[-1:][0][0]
  final_states = agent_out_df.loc[maxStep].state

  colours = ["indigo" if node == 1.0 else "slateblue" for node in final_states]
  return colours


def plotDirectedGraph(model):
  G = model.G
  pos = nx.spring_layout(G, seed=model.seed)

  node_sizes = [20] * len(G)
  node_colours = getNodeColours(model)

  M = G.number_of_edges()
  edge_colors = range(2, M + 2)
  edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
  cmap = plt.cm.plasma

  nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colours)
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

  ax = plt.gca()
  ax.set_axis_off()
  plt.show()


def singleRun(n, distribution, mu, sigma, in_degree):
  # How to choose which can
  # Normal distribution between 0 and 1: mu=0.5, sigma=0.17
  model = GranovetterModel(num_of_nodes=n, distribution=distribution, mu=mu, sigma=sigma, in_degree=in_degree)

  while model.running and model.schedule.steps < 100:
    model.step()

  print('The Granovetter Model ran for {} steps'.format(model.schedule.steps))
  model.datacollector.collect(model)

  plotDirectedGraph(model)

  model_out = model.datacollector.get_model_vars_dataframe()
  model_out.plot()
  plt.show()


def batchRun(n, distribution, mu, sigmas, in_degree):
  params = {
    "num_of_nodes": n,
    "distribution": distribution,
    "mu": mu,
    "sigma": sigmas,
    "in_degree": in_degree
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

  results_df = pd.DataFrame(results).drop('distribution', axis=1)
  print(results_df.head())

  results_df.to_csv('results.csv')

  return results_df


def main():
  n = 100
  mu = 0.25
  sigma = 0.12
  in_degree = 3
  run_type = RunType.Single
  distribution = Distribution.UNIFORM2

  if run_type == RunType.Granovetter:
    # Proof 1
    singleRun(n, Distribution.UNIFORM, 0.0, 0.0, in_degree)

    # Proof 2
    singleRun(n, Distribution.UNIFORM2, 0.0, 0.0, in_degree)

    # Proof 3
    sigmas = np.linspace(0.0, 1.0, 101).round(decimals=2)
    results = batchRun(n, Distribution.NORMAL, mu, sigmas, in_degree)

    median_results = results.groupby(by=["sigma"]).median()[['engagement_ratio']]
    median_results.plot()
    plt.axvline(x=0.12, linestyle='dashed', color='gray')
    plt.show()

    mean_results = results.groupby(by=["sigma"]).mean()[['engagement_ratio']]
    mean_results.plot()
    plt.show()

  elif run_type == RunType.Single:
    # Single Run
    singleRun(n, distribution, mu, sigma, in_degree)

  else:
    # Batch Run
    sigmas = np.linspace(0.0, 1.0, 11).round(decimals=2)

    results = batchRun(n, distribution, mu, sigmas, in_degree)

    # TODO: This doesn't do anything
    # tmp = results.groupby(by=["RunId", "iteration", "Step"]).median()
    # combined_results = tmp.drop(['num_of_nodes', 'AgentID', 'in_degree', 'mu'], axis=1)
    # combined_results.plot()
    # plt.show()

    results.groupby(by=["RunId"]).median().boxplot(by='sigma', column=['engagement_ratio'], grid=False, rot=45)
    plt.show()


if __name__ == '__main__':
  main()
