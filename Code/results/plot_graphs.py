import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.ticker as mtick

style = 'seaborn-poster'


def showDegreeHistogram(G, specification):
  """
  Plot a degree histogram showing the in- and out-degree.

  :param G: The network used in the mesa model.
  """

  in_degrees = [val for (node, val) in G.in_degree()]
  out_degrees = [val for (node, val) in G.out_degree()]

  d1 = np.array(in_degrees)
  d2 = np.array(out_degrees)

  plt.style.use(style)

  plt.hist([d1, d2], label=['in-degrees', 'out-degrees'], align='left')
  plt.legend(loc='upper right')

  plt.savefig('results/figures/degree_histogram_' + specification + '.png')
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


def plotDirectedGraph(model, specification):
  """
  Plot the network used in the mesa model.
  source: https://networkx.org/documentation/stable/auto_examples/drawing/plot_directed.html

  :param model: The mesa model
  """

  G = model.G
  pos = nx.spring_layout(G, seed=model.seed)

  node_sizes = [20] * len(G)
  node_colours = getNodeColours(model)

  M = G.number_of_edges()
  edge_colors = range(2, M + 2)
  edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

  plt.style.use('default')
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

  plt.savefig('results/figures/network_visualisation_' + specification + '.png')
  plt.show()


def sigmaPlot(results):
  """
  Plot a boxplot showing the engagement ratio for each sigma.

  :param results: The results from a batch run.
  """

  plt.style.use(style)

  # TODO: Choose whether to use median or mean results.

  # Median
  median_results = results.groupby(by=["sigma"]).median()[['engagement_ratio']]

  fig = median_results.plot(color='#EE0000')
  plt.axvline(x=0.12, linestyle='dashed', color='gray')

  plt.title('Median agent engagement for normal distributions with varying sigmas')
  plt.xlabel('Sigma')
  plt.ylabel('Percentage of engaged agents')

  # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # https://matplotlib.org/stable/api/ticker_api.html
  fig.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

  plt.savefig('results/figures/granovetter_sigma.png')
  # plt.show()

  # # Mean
  # mean_results = results.groupby(by=["sigma"]).mean()[['engagement_ratio']]
  #
  # fig_mean = mean_results.plot(color='#EE0000')
  # plt.axvline(x=0.12, linestyle='dashed', color='gray')
  #
  # plt.title('Mean agent engagement for normal distributions with varying sigmas')
  # plt.xlabel('Sigma')
  # plt.ylabel('Percentage of engaged agents')
  #
  # # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # # https://matplotlib.org/stable/api/ticker_api.html
  # fig_mean.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
  # fig_mean.xaxis.set_ticks(np.arange(0.0, 1.1, 0.1))
  #
  # plt.show()


def sigmaBoxPlot(results):
  """
  Plot a boxplot of the engagement equilibrium for every sigma.

  :param results: The results from a batch run.
  """

  plt.style.use(style)

  results.groupby(by=["RunId"]).median().boxplot(by='sigma', column=['engagement_ratio'], grid=False, rot=45)

  plt.savefig('results/figures/sigma_boxplot.png')
  plt.show()


def singleRunPlot(results, titleSpecification, filename):
  """
  Plot the progression of engaged agents during a single simulation.

  :param results: The results from a single run.
  :param titleSpecification: Specification of the title to add at the end of the standard title.
  """

  plt.style.use(style)

  fig = results.plot(color='#EE0000')

  plt.title('Progression of agent engagement' + titleSpecification)
  plt.xlabel('Steps')
  plt.ylabel('Percentage of engaged agents')

  # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # https://matplotlib.org/stable/api/ticker_api.html
  fig.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
  fig.set(facecolor='white')
  fig.get_legend().remove()

  plt.savefig('results/figures/' + filename + '.png')
  plt.show()


def multipleRunPlot(results, maxSteps, titleSpecification, filename):
  """
  Plot the progression of engaged agents for multiple simulations.

  :param results: The results from a batch run.
  :param maxSteps:
  :param titleSpecification: Specification of the title to add at the end of the standard title.
  """

  plt.style.use(style)

  fig, ax = plt.subplots()

  for value in results.iteration.unique():
    x = results[results['iteration'] == value].Step
    y = results[results['iteration'] == value].engagement_ratio
    ax.plot(x, y, label=value)  # , color='#EE0000')

  plt.title('Progression of agent engagement' + titleSpecification)
  plt.xlabel('Steps')
  plt.ylabel('Percentage of engaged agents')

  # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # https://matplotlib.org/stable/api/ticker_api.html
  ax.get_yaxis().set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
  fig.set(facecolor='white')

  plt.xlim(0, maxSteps)
  plt.legend()

  plt.savefig('results/figures/' + filename + '.png')
  plt.show()
