import random

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.ticker as mtick
from matplotlib.pylab import MaxNLocator

from utilities.network_util import NetworkType, KnowledgeType

style = 'seaborn-poster'
title_size = 32
axis_label_size = 28
tick_size = 24

steps_axis = 'Number of steps'
engagement_axis = 'Equilibrium of engaged agents'
sigma_axis = 'Standard deviation'
n_axis = 'Number of agents'
in_degree_axis = 'In-degree'


def getAxisLabel(variable):
  if variable == 'Step':
    return steps_axis

  elif variable == 'engagement_ratio':
    return engagement_axis

  elif variable == 'sigma':
    return sigma_axis

  elif variable == 'num_of_nodes':
    return n_axis

  elif variable == 'in_degree':
    return in_degree_axis

  else:
    return variable


def labelConversion(variable, label):
  if variable == 'neighbourhood':
    return KnowledgeType(label).name

  elif variable == 'networkType':
    return NetworkType(label).name

  elif variable == 'knowledge':
    return KnowledgeType(label).name


def showDegreeHistogram(path, G, specification):
  """
  Plot a degree histogram showing the in- and out-degree.

  :param path:
  :param G: The network used in the mesa model.
  :param specification:
  """

  plt.style.use(style)

  fig, ax = plt.subplots(figsize=(15, 10))

  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  in_degrees = [val for (node, val) in G.in_degree()]
  out_degrees = [val for (node, val) in G.out_degree()]

  d1 = np.array(in_degrees)
  d2 = np.array(out_degrees)

  maxValue = max(max(in_degrees), max(out_degrees))
  bins = np.arange(maxValue) - 0.5

  _, final_bins, _ = ax.hist([d1, d2], bins=bins, label=['in-degrees', 'out-degrees'])

  plt.xticks(range(maxValue))
  plt.xlim([-0.5, max(final_bins)])
  plt.ylim([0, G.number_of_nodes()])

  # plt.title('Histogram of degree distribution in the network', size=title_size)
  plt.xlabel('Degree', size=axis_label_size)
  plt.ylabel('Frequency', size=axis_label_size)
  for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(tick_size)

  plt.legend()

  plt.gcf().set_size_inches(15, 10)
  plt.savefig(path + 'degree_histogram_' + specification + '.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
  plt.close()


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

  plt.show()


def sigmaPlot(path, results):
  """
  Plot a boxplot showing the engagement ratio for each sigma.

  :param path:
  :param results: The results from a batch run.
  """

  plt.style.use(style)

  fig, ax = plt.subplots(figsize=(15, 10))

  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  mean_sdSubplot(data=(None, results), independent_variable='sigma', dependent_variable='engagement_ratio',
                 comparison_variable=None, colour='#EE0000', error_bar=False)

  plt.axvline(x=0.12, linestyle='dashed', color='gray')
  plt.axhline(y=0.5, linestyle='dashed', color='gray')

  # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # https://matplotlib.org/stable/api/ticker_api.html
  ax.get_yaxis().set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
  ax.get_xaxis().set_ticks(np.arange(0.0, 2.1, 0.25))

  # plt.title('Median agent engagement for normal distributions with varying sigmas', size=title_size)
  plt.xlabel(sigma_axis, size=axis_label_size)
  plt.ylabel(engagement_axis, size=axis_label_size)
  ax.tick_params(axis='both', labelsize=tick_size)

  plt.legend()

  plt.gcf().set_size_inches(15, 10)
  plt.savefig(path + 'granovetter_sigma.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
  plt.close()


def multipleVariablesPlot(path, data_1, data_2, varying_variable, comparison_variable):
  """
  Plot a boxplot showing the engagement ratio for each sigma.

  :param path:
  :param data_1:
  :param data_2:
  :param varying_variable:
  :param comparison_variable:
  """

  plt.style.use(style)

  fig, ax = plt.subplots(figsize=(15, 10))

  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  mean_sdSubplot(data=data_1, independent_variable=varying_variable, dependent_variable='engagement_ratio',
                 comparison_variable=comparison_variable, colour='tab:blue')

  mean_sdSubplot(data=data_2, independent_variable=varying_variable, dependent_variable='engagement_ratio',
                 comparison_variable=comparison_variable, colour='tab:orange')

  # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # https://matplotlib.org/stable/api/ticker_api.html
  ax.get_yaxis().set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

  if varying_variable == 'sigma':
    ax.get_xaxis().set_ticks(np.arange(0.0, 1.1, 0.1))
  elif varying_variable == 'num_of_nodes':
    ax.get_xaxis().set_ticks(range(80, 150, 10))
  elif varying_variable == 'in_degree':
    ax.get_xaxis().set_ticks(range(0, 11, 1))

  # y_min = max(plt.gca().get_lines()[0].get_ydata())
  # y_max = 0
  # for line in plt.gca().get_lines():
  #   line_min = min(line.get_ydata())
  #   if line_min <= y_min:
  #     y_min = line_min
  #
  #   line_max = max(line.get_ydata())
  #   if line_max >= y_max:
  #     y_max = line_max
  #
  # print(y_min)
  # print(y_max)
  #
  # if y_max - y_min >= 0.1:
  #   plt.yticks(np.arange(y_min, y_max + 0.1, 0.2))
  # else:
  #   plt.yticks(np.arange(y_min-0.1, y_max + 0.005, 0.01))

  # plt.title('Median agent engagement for normal distributions with varying ' + getAxisLabel(varying_variable),
  #           size=title_size)
  plt.xlabel(getAxisLabel(varying_variable), size=axis_label_size)
  plt.ylabel(engagement_axis, size=axis_label_size)
  ax.tick_params(axis='both', labelsize=tick_size)

  plt.legend()

  plt.gcf().set_size_inches(15, 10)
  plt.savefig(path + varying_variable + '_comparison_engagement.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
  plt.close()

  # Mean diffusion rate
  fig, ax = plt.subplots(figsize=(15, 10))

  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  mean_sdSubplot(data=data_1, independent_variable=varying_variable, dependent_variable='Step',
                 comparison_variable=comparison_variable, colour='tab:blue')

  mean_sdSubplot(data=data_2, independent_variable=varying_variable, dependent_variable='Step',
                 comparison_variable=comparison_variable, colour='tab:orange')

  if varying_variable == 'sigma':
    ax.get_xaxis().set_ticks(np.arange(0.0, 1.1, 0.1))
  elif varying_variable == 'num_of_nodes':
    ax.get_xaxis().set_ticks(range(80, 150, 10))
  elif varying_variable == 'in_degree':
    ax.get_xaxis().set_ticks(range(0, 11, 1))

  # plt.title('Mean diffusion rate for normal distributions with varying ' + getAxisLabel(varying_variable),
  #           size=title_size)
  plt.xlabel(getAxisLabel(varying_variable), size=axis_label_size)
  plt.ylabel('Number of steps', size=axis_label_size)
  ax.tick_params(axis='both', labelsize=tick_size)

  plt.legend()

  plt.gcf().set_size_inches(15, 10)
  plt.savefig(path + varying_variable + '_comparison_steps.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
  plt.close()


def mean_sdSubplot(data, independent_variable, dependent_variable, comparison_variable, colour, error_bar=True):
  (comparison_value, results) = data

  mean_step_results_2 = results.groupby(by=[independent_variable])[[dependent_variable]].mean()
  sd_step_results_2 = results.groupby(by=[independent_variable])[[dependent_variable]].std()

  # TODO: For sigma plot, only plot error bars every 0.1 not every 0.01

  x2 = mean_step_results_2.index.values
  y2 = np.array(mean_step_results_2[dependent_variable])

  if dependent_variable == 'engagement_ratio':
    y2errorUp = [1.0 - y if (y + err > 1.0) else err for (y, err) in zip(y2, sd_step_results_2[dependent_variable].to_list())]
  else:
    y2errorUp = sd_step_results_2[dependent_variable].to_list()

  y2errorLow = [y if (y - err < 0) else err for (y, err) in zip(y2, y2errorUp)]
  y2error = np.array([y2errorLow, y2errorUp])

  y2min = y2 - y2errorLow
  y2max = y2 + y2errorUp

  if comparison_variable is None:
    label_specification = ''
  else:
    label_specification = labelConversion(comparison_variable, comparison_value)

  plt.plot(x2, y2, alpha=1, linewidth=2, color=colour)
  if error_bar:
    plt.errorbar(x=x2, y=y2, yerr=y2error, capsize=5, capthick=2, linewidth=1, color=colour,
                 alpha=0.5, label='Mean ' + label_specification)
  plt.fill_between(x=x2, y1=y2min, y2=y2max, alpha=0.2, color=colour,
                   label='Standard deviation ' + label_specification)


def sigmaBoxPlot(path, results):
  """
  Plot a boxplot of the engagement equilibrium for every sigma.

  :param path:
  :param results: The results from a batch run.
  """

  plt.style.use(style)

  # fig = plt.figure(figsize=(15, 10))
  # results.groupby(by=["RunId"]).median().boxplot(by='sigma', column=['engagement_ratio'], grid=False, rot=45)
  grouped_results = results.groupby(by=["RunId"]).median()

  plt.boxplot(x='sigma', y='engagement_ratio', data=grouped_results, rot=45)

  plt.gcf().set_size_inches(15, 10)
  plt.savefig(path + 'sigma_boxplot.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
  plt.close()


def singleRunPlot(path, results, filename):
  """
  Plot the progression of engaged agents during a single simulation.

  :param path:
  :param results: The results from a single run.
  :param filename:
  """

  plt.style.use(style)

  fig = results.plot(color='#EE0000')

  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # https://matplotlib.org/stable/api/ticker_api.html
  fig.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
  fig.yaxis.set_major_locator(MaxNLocator(integer=True))
  fig.xaxis.set_major_locator(MaxNLocator(integer=True))

  line = fig.lines[0]
  x_max = max(line.get_xdata())
  y_max = max(line.get_ydata())

  # plt.xlim([0, x_max + 0.1 * x_max])
  # plt.ylim([0, y_max + 0.1 * y_max])
  if x_max > 20:
    plt.xticks(range(0, x_max + 1, 20))
  else:
    plt.xticks(range(x_max + 1))

  if y_max >= 0.2:
    plt.yticks(np.arange(0.0, y_max + 0.2, 0.2))
  else:
    plt.yticks(np.arange(0.0, y_max + 0.01, 0.01))

  # plt.title('Progression of agent engagement ' + titleSpecification, size=title_size)
  # plt.title(titleSpecification, size=title_size)
  plt.xlabel(steps_axis, size=axis_label_size)
  plt.ylabel(engagement_axis, size=axis_label_size)
  fig.axes.tick_params(axis='both', labelsize=tick_size)

  plt.legend()

  plt.gcf().set_size_inches(15, 10)
  plt.savefig(path + filename + '.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
  plt.close()


def multipleRunPlot(path, results, maxSteps, filename):
  """
  Plot the progression of engaged agents for multiple simulations.

  :param path:
  :param results: The results from a batch run.
  :param maxSteps:
  :param filename:
  """

  plt.style.use(style)

  fig, ax = plt.subplots(figsize=(15, 10))

  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  for value in results.iteration.unique():
    x = results[results['iteration'] == value].Step
    y = results[results['iteration'] == value].engagement_ratio
    ax.plot(x, y, label=value)

  # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # https://matplotlib.org/stable/api/ticker_api.html
  ax.get_yaxis().set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
  ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))

  plt.xlim(0, maxSteps)

  # plt.title('Progression of agent engagement ' + titleSpecification, size=title_size)
  plt.xlabel(steps_axis, size=axis_label_size)
  plt.ylabel(engagement_axis, size=axis_label_size)
  fig.axes.tick_params(axis='both', labelsize=tick_size)

  plt.legend()

  plt.gcf().set_size_inches(15, 10)
  plt.savefig(path + filename + '.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
  plt.close()


def comparisonPlot(path, results, filename, variable):
  """
  Plot the progression of engaged agents for multiple simulations.

  :param path:
  :param results: The results from a batch run.
  :param filename:
  :param variable:
  """

  plt.style.use(style)

  if len(results.iteration.unique()) < 3:
    random_i = results.iteration.unique()
  else:
    random_i = random.sample(sorted(results.iteration.unique()), k=3)

  for i in random_i:
    fig, ax = plt.subplots(figsize=(15, 10))

    plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

    results_i = results[results['iteration'] == i]

    for value in results[variable].unique():
      x = results_i[results_i[variable] == value].Step
      y = results_i[results_i[variable] == value].engagement_ratio
      ax.plot(x, y, label=labelConversion(variable, value))

    # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
    # https://matplotlib.org/stable/api/ticker_api.html
    ax.get_yaxis().set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    # TODO: Fix the axis for plot with n

    # plt.title('Progression of agent engagement ' + "(iteration: " + str(i) + ", variable: " + str(variable) + ")", size=title_size)
    plt.xlabel(steps_axis, size=axis_label_size)
    plt.ylabel(engagement_axis, size=axis_label_size)
    ax.tick_params(axis='both', labelsize=tick_size)

    plt.legend()

    plt.gcf().set_size_inches(15, 10)
    plt.savefig(path + filename + '(' + str(np.where(random_i == i)[0][0]) + ').png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def boxplotComparison(path, results, independent_variable, dependent_variable):
  """
  Plot the progression of engaged agents for multiple simulations.

  :param path:
  :param results: The results from a batch run.
  :param independent_variable:
  :param dependent_variable:
  """

  plt.style.use(style)

  fig = plt.figure(figsize=(15, 10))

  grouped_results = results.groupby(by=["RunId"]).max()

  bp = sns.boxplot(x=independent_variable, y=dependent_variable, data=grouped_results)  # , rot=45)

  # new_results = results.groupby(by=["RunId"]).max()
  # bp = new_results.boxplot(by=independent_variable, column=[dependent_variable], grid=False, rot=45)

  # plt.title(title + " (variable: " + str(independent_variable) + ")", size=title_size)
  fig.texts = []
  plt.xlabel(getAxisLabel(independent_variable), size=axis_label_size)
  plt.ylabel(getAxisLabel(dependent_variable), size=axis_label_size)
  bp.axes.tick_params(axis='both', labelsize=tick_size)

  x_ticks = bp.get_xticks()
  plt.xticks(x_ticks, labels=[labelConversion(independent_variable, bool(x)) for x in x_ticks], rotation=0)
  bp.axes.get_yaxis().set_major_locator(MaxNLocator(integer=True))

  plt.gcf().set_size_inches(15, 10)
  plt.savefig(path + independent_variable + '_boxplot_' + dependent_variable + '.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
  plt.close()
