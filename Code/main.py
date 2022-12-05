import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import random

from mesa.batchrunner import batch_run
from model import GranovetterModel


# https://networkx.org/documentation/stable/auto_examples/drawing/plot_directed.html
from util import *


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

  ax = plt.gca()
  ax.set_axis_off()
  plt.show()


def singleRun(n, sigma, t):
  # How to choose which can
  # Normal distribution between 0 and 1: mu=0.5, sigma=0.17
  model = GranovetterModel(num_of_nodes=n, sigma=sigma, t=t, in_degree=3)
  plotDirectedGraph(model)

  while model.running and model.schedule.steps < 100:
    model.step()

  print('The Granovetter Model ran for {} steps'.format(model.schedule.steps))
  model.datacollector.collect(model)

  agent_out = model.datacollector.get_agent_vars_dataframe()
  # print(agent_out.head())

  agent_out_df = pd.DataFrame(data=agent_out)
  random_agent = random.choice(range(len(agent_out_df.xs(0))))
  # agent_results = agent_out_df.xs(random_agent, level=1)
  # agent_results = agent_out_df.xs(random_agent, level="AgentID").state

  # print(agent_out_df.iloc[-1:])
  maxStep = agent_out_df.index.values[-1:][0][0]

  colours = agent_out_df.loc[maxStep].state

  model_out = model.datacollector.get_model_vars_dataframe()
  # print(model_out.head())
  model_out.plot()
  plt.show()

  # print(model_out['engagement_ratio'].max())


def batchRun(n, sigmas, t):
  params = {
    "num_of_nodes": n,
    "sigma": sigmas,
    "t": t,
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

  results_df = pd.DataFrame(results).drop('t', axis=1)
  # print(results_df.head())

  results_df.to_csv('results.csv')

  return results_df


class RunType(Enum):
  Granovetter = 0
  Single = 1
  Batch = 2


def main():
  run_type = RunType.Granovetter

  n = 100
  mu = 0.25
  sigma = 0.15

  t = Thresholds(n, mu, [sigma])

  if run_type.value == 0:
    # Proof 1
    threshold = np.arange(0.0, 1.0, (1.0/n))
    t.setThresholds(sigma, threshold)
    singleRun(n, sigma, t)

    # Proof 2
    threshold[threshold == 0.01] = 0.02
    t.setThresholds(sigma, threshold)
    singleRun(n, sigma, t)

    # Proof 3
    sigmas = np.linspace(0.0, 1.0, 101).round(decimals=2)
    t = Thresholds(n, mu, sigmas)

    results = batchRun(n, sigmas, t)
    # print(results)

    median_results = results.groupby(by=["sigma"]).median()[['engagement_ratio']]
    median_results.plot()
    plt.axvline(x=0.12, linestyle='dashed', color='gray')
    plt.show()

    # mean_results = results.groupby(by=["sigma"]).mean()[['engagement_ratio']]
    # mean_results.plot()
    # plt.show()

  elif run_type.value == 1:
    # Single Run
    sigma = 0.15
    t = Thresholds(n, mu, [sigma])
    singleRun(n, sigma, t)

  else:
    # Batch Run
    sigmas = np.linspace(0.0, 1.0, 11).round(decimals=2)
    t = Thresholds(n, mu, sigmas)

    results = batchRun(n, sigmas, t)

    # This doesn't do anything
    # tmp = results.groupby(by=["RunId", "iteration", "Step"]).median()
    # combined_results = tmp.drop(['num_of_nodes', 'AgentID', 'in_degree', 'mu'], axis=1)
    # combined_results.plot()
    # plt.show()

    results.groupby(by=["RunId"]).median().boxplot(by='sigma', column=['engagement_ratio'], grid=False, rot=45)
    plt.show()


if __name__ == '__main__':
  main()
