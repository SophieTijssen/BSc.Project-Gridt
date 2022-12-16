import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import networkx as nx
import pandas as pd

from mesa.batchrunner import batch_run
from model import GranovetterModel

from util import *


def showDegreeHistogram(G):
  """
  Plot a degree histogram showing the in- and out-degree.
  :param G: The network used in the mesa model.
  """

  in_degrees = [val for (node, val) in G.in_degree()]
  out_degrees = [val for (node, val) in G.out_degree()]

  d1 = np.array(in_degrees)
  d2 = np.array(out_degrees)

  plt.hist([d1, d2], label=['in-degrees', 'out-degrees'], align='left')
  plt.legend(loc='upper right')
  plt.show()


def getNodeColours(model):
  """
  Assign node colours to each node based on whether the agent in that node cooperated or not.
  :param model: The mesa model.
  :return: A list of strings containing the node colours.
  """
  agent_out = model.datacollector.get_agent_vars_dataframe()
  agent_out_df = pd.DataFrame(data=agent_out)

  maxStep = agent_out_df.index.values[-1:][0][0]
  final_states = agent_out_df.loc[maxStep].state

  colours = ["indigo" if node == 1.0 else "slateblue" for node in final_states]
  return colours


def plotDirectedGraph(model):
  """
  Plot the network used in the mesa model.
  :param model: The mesa model

  source: https://networkx.org/documentation/stable/auto_examples/drawing/plot_directed.html
  """
  G = model.G
  pos = nx.spring_layout(G, seed=model.seed)
  # pos = nx.spring_layout(G)

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


def singleRun(n, distribution, mu, sigma, in_degree, style):
  """
  Run the model for a single iteration.
  :param n: The number of agents in the network.
  :param distribution: The distribution used for sampling the agent thresholds.
  :param mu: The mean of the threshold distribution (in case of a normal distribution).
  :param sigma: The standard deviation of the threshold distribution (in case of a normal distribution).
  :param in_degree: The in-degree of all agents.
  :param style: The style of the plots.
  """
  # How to choose which can
  # Normal distribution between 0 and 1: mu=0.5, sigma=0.17
  model = GranovetterModel(num_of_nodes=n, distribution=distribution.value, mu=mu, sigma=sigma, in_degree=in_degree)

  while model.running and model.schedule.steps < 100:
    model.step()

  print('The Granovetter Model ran for {} steps'.format(model.schedule.steps))
  model.datacollector.collect(model)

  plt.style.use('default')
  plotDirectedGraph(model)

  plt.style.use(style)

  showDegreeHistogram(model.G)

  plt.show()

  model_out = model.datacollector.get_model_vars_dataframe()

  plt.style.use(style)

  fig = model_out.plot()  # color='#EE0000')

  print(type(fig))

  # plt.title('Progression of engagements of agents')
  plt.title('Progression of engagements of agents: ' + style)
  plt.xlabel('Steps')
  plt.ylabel('Percentage of engaged agents')

  # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # https://matplotlib.org/stable/api/ticker_api.html
  fig.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
  fig.set(facecolor='white')
  fig.get_legend().remove()

  plt.show()


def batchRun(n, distribution, mu, sigmas, in_degree):
  """
  Run the model for multiple iterations (using BatchRun).
  :param n: The number of agents in the network.
  :param distribution: The distribution used for sampling the agent thresholds.
  :param mu: The mean of the threshold distribution (in case of a normal distribution).
  :param sigmas: A list containing the standard deviations of the threshold distributions
                 for each iteration (in case of a normal distribution).
  :param in_degree: The in-degree of all agents.
  :return: A pandas dataframe containing the results/data from the DataCollector.
  """
  params = {
    "num_of_nodes": n,
    "distribution": distribution.value,
    "mu": mu,
    "sigma": sigmas,
    "in_degree": in_degree
  }

  results = batch_run(
    GranovetterModel,
    parameters=params,
    iterations=100,
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
  """
  The main function used to run the project.
  """
  n = 100
  mu = 0.25
  sigma = 0.2
  in_degree = 3
  run_type = RunType.Granovetter
  distribution = Distribution.NORMAL
  style = 'seaborn-poster'

  if run_type == RunType.Granovetter:
    # Proof 1
    singleRun(n, Distribution.UNIFORM, 0.0, 0.0, in_degree, style)

    # Proof 2
    singleRun(n, Distribution.UNIFORM_MODIFIED, 0.0, 0.0, in_degree, style)

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
    # styles = ['bmh', 'ggplot', 'seaborn', 'seaborn-notebook', 'seaborn-poster', 'seaborn-whitegrid', 'tableau-colorblind10']
    # print(plt.style.available)
    # for style in styles:
    #   singleRun(n, distribution, mu, sigma, in_degree, style)
    singleRun(n, distribution, mu, sigma, in_degree, style)

  else:
    # Batch Run
    sigmas = np.linspace(0.0, 1.0, 11).round(decimals=2)

    results = batchRun(n, distribution, mu, sigmas, in_degree)

    results.groupby(by=["RunId"]).median().boxplot(by='sigma', column=['engagement_ratio'], grid=False, rot=45)

    plt.style.use(style)

    plt.show()


if __name__ == '__main__':
  main()
