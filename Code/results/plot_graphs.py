import random

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.ticker as mtick
from matplotlib.pylab import MaxNLocator

from utilities.network_util import NetworkType, KnowledgeType, getComparisonValue
from utilities.model_util import sigmas, nums, out_degrees, getVariableName

style = 'seaborn-poster'
title_size = 32
axis_label_size = 26
tick_size = 20
# colour1 = '#2d4263'
colour11 = '#2A497C'
colour12 = '#5982c7'
# colour1 = '#31589C'
# colour2 = '#c84b31'
colour21 = '#D75135'
colour22 = '#e2816c'
# colour2 = '#E45638'
line1 = '-'
line2 = '--'
marker1 = 'o'
marker2 = 's'

steps_axis = 'Number of steps'
engagement_axis = 'Equilibrium of adopting agents'
sigma_axis = 'Standard deviation of decision threshold distribution'
n_axis = 'Number of total agents'
out_degree_axis = 'Out-degree'


def getAxisLabel(variable):
  if variable == 'Step':
    return steps_axis

  elif variable == 'engagement_ratio':
    return engagement_axis

  elif variable == 'sigma':
    return sigma_axis

  elif variable == 'num_of_nodes':
    return n_axis

  elif variable == 'out_degree':
    return out_degree_axis

  else:
    return variable


def labelConversion(variable, label):
  if variable == 'neighbourhood':
    return KnowledgeType(label).name

  elif variable == 'networkType':
    return NetworkType(label).name

  elif variable == 'knowledge':
    return KnowledgeType(label).name


def rand_jitter(arr, position=None):
  """
  https://stackoverflow.com/a/21276920
  :param arr:
  :param position:
  :return:
  """
  stdev = .01 * (max(arr) - min(arr))
  # print(stdev)
  r = np.random.randn(len(arr))
  # print(type(r))
  if position is None:
    return arr + r * stdev
  else:
    # r2 = np.absolute(r)
    # print(r2)
    return arr + position * np.absolute(r) * stdev


def jitter(x, y, position, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, **kwargs):
  """
  https://stackoverflow.com/a/21276920
  """
  return plt.scatter(rand_jitter(x, position), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)


def prepareData(df, comparison_variable, comparison_values, independent_variable, dependent_variable):

  if 'AgentID' in df:
    results1 = df.loc[df['AgentID'] == 0]
    results = results1.drop(['AgentID', 'state'], axis=1)
  else:
    results = df

  grouped_results = results.groupby([comparison_variable, independent_variable])[dependent_variable].apply(list).groupby(comparison_variable).apply(list)

  data_0 = grouped_results[comparison_values[0]]
  data_1 = grouped_results[comparison_values[1]]

  ticks = results[independent_variable].unique()

  return data_0, data_1, ticks


# def createMultipleBoxplots():
def createMultipleBoxplots(results, comparison_variable, independent_variable, dependent_variable, filename):
  # # Some fake data to plot
  # data_0 = [[1, 2, 5], [5, 7, 2, 2, 5], [7, 2, 5]]
  # data_1 = [[6, 4, 2], [1, 2, 5, 3, 2], [2, 3, 5, 1]]
  #
  # ticks = ['A', 'B', 'C']
  comparison_values = getComparisonValue(comparison_variable)

  data_0, data_1, ticks = prepareData(results, comparison_variable, comparison_values, independent_variable, dependent_variable)

  # function for setting the colors of the box plots pairs
  def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

  plt.style.use(style)

  fig, ax = plt.subplots(figsize=(15, 10))
  # fig = plt.figure(figsize=(15, 10))
  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  bpl = plt.boxplot(data_0, positions=np.array(range(len(data_0))) * 2.0 - 0.4, sym='', widths=0.6, medianprops=dict(linestyle='-', linewidth=2.5, color='#D7191C'))
  bpr = plt.boxplot(data_1, positions=np.array(range(len(data_1))) * 2.0 + 0.4, sym='', widths=0.6, medianprops=dict(linestyle='-', linewidth=2.5, color='#2C7BB6'))
  set_box_color(bpl, '#D7191C')  # colors are from http://colorbrewer2.org/
  set_box_color(bpr, '#2C7BB6')

  # draw temporary red and blue lines and use them to create a legend
  plt.plot([], c='#D7191C', label=labelConversion(comparison_variable, comparison_values[0]))
  plt.plot([], c='#2C7BB6', label=labelConversion(comparison_variable, comparison_values[1]))
  plt.legend()

  if dependent_variable == 'engagement_ratio':
    # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
    # https://matplotlib.org/stable/api/ticker_api.html
    ax.get_yaxis().set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

  # if independent_variable == 'sigma':
  #   ax.get_xaxis().set_ticks(np.arange(0.0, max(sigmas) + 0.1, 0.1))
  # elif independent_variable == 'num_of_nodes':
  #   ax.get_xaxis().set_ticks(
  #     data_1[1][independent_variable].unique())  # TODO: change to this statement below when rerunning simulations
  #   # ax.get_xaxis().set_ticks(range(min(nums), max(nums) + 1, 10))
  #   # ax.get_xaxis().set_ticks(nums)
  # elif independent_variable == 'out_degree':
  #   ax.get_xaxis().set_ticks(out_degrees)

  # plt.title('Median agent engagement for normal distributions with varying ' + getAxisLabel(independent_variable),
  #           size=title_size)
  plt.xlabel(getAxisLabel(independent_variable), size=axis_label_size)
  plt.ylabel(getAxisLabel(dependent_variable), size=axis_label_size)
  ax.tick_params(axis='both', labelsize=tick_size)

  plt.xticks(range(0, len(ticks) * 2, 2), ticks)
  plt.xlim(-2, len(ticks) * 2)
  # plt.ylim(0, 1)

  plt.gcf().set_size_inches(15, 10)
  plt.savefig('results/figures/boxplots/' + filename + '.png', pad_inches=0.1, dpi=300)
  plt.close()


def generateFigures():
  path_knowledge = 'results/raw_data/knowledge_comparison/'
  path_network = 'results/raw_data/network_comparison/'

  # results_knowledge = pd.read_csv(path_knowledge + 'knowledge_comparison.csv')
  # results_network = pd.read_csv(path_network + 'network_comparison.csv')

  plot = 'multipleBoxplot'
  # i = 1

  if plot == 'multipleBoxplot':
    paths = [(path_knowledge, 'knowledge'), (path_network, 'networkType')]
    for path, comparison_variable in paths:

      for csv_file, independent_variable in [('out-degree_comparison.csv', 'out_degree'), ('n_comparison.csv', 'num_of_nodes'), ('sigma_comparison.csv', 'sigma')]:
        results = pd.read_csv(path + csv_file)

        for dependent_variable in ['engagement_ratio', 'Step']:

          createMultipleBoxplots(results, comparison_variable, independent_variable, dependent_variable, filename=comparison_variable + '_' + independent_variable + '_' + dependent_variable)

          print('finished', comparison_variable, independent_variable, dependent_variable)
          # i += 1

  elif plot == 'multipleVariablesPlot':

    paths = [(path_knowledge, 'knowledge'), (path_network, 'networkType')]
    # paths = [(path_knowledge, 'knowledge')]
    for path, comparison_variable in paths:

      comparison_values = getComparisonValue(comparison_variable)

      # results_out_degree = pd.read_csv(path + 'out-degree_comparison.csv')
      # results_n = pd.read_csv(path + 'n_comparison.csv')
      # results_sigma = pd.read_csv(path + 'sigma_comparison.csv')

      # for csv_file, independent_variable in [('out-degree_comparison.csv', 'out_degree'), ('n_comparison.csv', 'num_of_nodes'), ('sigma_comparison.csv', 'sigma')]:
      for csv_file, independent_variable in [('n_comparison.csv', 'num_of_nodes')]:
        results = pd.read_csv(path + csv_file)

        data_group0 = results[results[comparison_variable] == comparison_values[0]]
        data_group1 = results[results[comparison_variable] == comparison_values[1]]

        for dependent_variable in ['engagement_ratio', 'Step']:

          multipleVariablesPlot('results/figures/presentation/' + comparison_variable + '/',
                                (comparison_values[0], data_group0),
                                (comparison_values[1], data_group1),
                                comparison_variable, independent_variable, dependent_variable)

          print('finished', comparison_variable, independent_variable, dependent_variable)

        # for dependent_variable in ['engagement_ratio', 'Step']:
        #   multipleBoxplots(path, results, comparison_variable, independent_variable, dependent_variable)


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

  ins = [val for (node, val) in G.in_degree()]
  outs = [val for (node, val) in G.out_degree()]

  d1 = np.array(ins)
  d2 = np.array(outs)

  maxValue = max(max(ins), max(outs))
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
  plt.savefig(path + 'degree_histogram_' + specification + '.png', pad_inches=0.1, dpi=300)
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

  mean_sdSubplot(ax, data=(None, results), independent_variable='sigma', dependent_variable='engagement_ratio',
                 comparison_variable=None, graph_style=((colour11, colour12), line1, marker1))

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
  plt.savefig(path + 'granovetter_sigma.png', pad_inches=0.1, dpi=300)
  plt.close()


def multipleVariablesPlot(path, data_1, data_2, comparison_variable, independent_variable, dependent_variable):
  """
  Plot a boxplot showing the engagement ratio for each sigma.

  :param path:
  :param data_1:
  :param data_2:
  :param comparison_variable:
  :param independent_variable:
  :param dependent_variable:
  """

  plt.style.use(style)

  fig, ax = plt.subplots(figsize=(15, 10))

  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  mean_sdSubplot(ax, data=data_1, independent_variable=independent_variable, dependent_variable=dependent_variable,
                 comparison_variable=comparison_variable, graph_style=((colour11, colour12), line1, marker1))

  mean_sdSubplot(ax, data=data_2, independent_variable=independent_variable, dependent_variable=dependent_variable,
                 comparison_variable=comparison_variable, graph_style=((colour21, colour22), line2, marker2))

  if dependent_variable == 'engagement_ratio':
    # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
    # https://matplotlib.org/stable/api/ticker_api.html
    ax.get_yaxis().set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

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

  if independent_variable == 'sigma':
    ax.get_xaxis().set_ticks(np.arange(0.0, max(sigmas)+0.1, 0.1))
  elif independent_variable == 'num_of_nodes':
    ax.get_xaxis().set_ticks(data_1[1][independent_variable].unique())  # TODO: change to this statement below when rerunning simulations
    # ax.get_xaxis().set_ticks(range(min(nums), max(nums) + 1, 10))
    # ax.get_xaxis().set_ticks(nums)
  elif independent_variable == 'out_degree':
    ax.get_xaxis().set_ticks(out_degrees)

  # plt.title('Median agent engagement for normal distributions with varying ' + getAxisLabel(independent_variable),
  #           size=title_size)
  plt.xlabel(getAxisLabel(independent_variable), size=axis_label_size)
  plt.ylabel(getAxisLabel(dependent_variable), size=axis_label_size)
  ax.tick_params(axis='both', labelsize=tick_size)

  plt.legend()

  plt.gcf().set_size_inches(15, 10)
  plt.savefig(path + independent_variable + '_comparison_' + getVariableName(dependent_variable) + '.png', pad_inches=0.1, dpi=300)
  plt.close()

  print('finished ', dependent_variable)


def mean_sdSubplot(ax, data, independent_variable, dependent_variable, comparison_variable, graph_style, error_bar=True):
  (comparison_value, results) = data
  ((colour_dark, colour_light), line, marker) = graph_style

  mean_results = results.groupby(by=[independent_variable])[[dependent_variable]].mean()
  sd_results = results.groupby(by=[independent_variable])[[dependent_variable]].std()

  # TODO: For sigma plot, only plot error bars every 0.1 not every 0.01

  x = mean_results.index.values
  y = np.array(mean_results[dependent_variable])

  if dependent_variable == 'engagement_ratio':
    errorUp = [1.0 - y if (y + err > 1.0) else err for (y, err) in zip(y, sd_results[dependent_variable].to_list())]
  else:
    errorUp = sd_results[dependent_variable].to_list()

  errorLow = [y if (y - err < 0) else err for (y, err) in zip(y, errorUp)]
  yerror = np.array([errorLow, errorUp])

  yMin = y - errorLow
  # min_results = results.groupby(by=[independent_variable])[[dependent_variable]].min()
  # yMin = np.array(min_results[dependent_variable])
  # print(yMin)
  yMax = y + errorUp
  # max_results = results.groupby(by=[independent_variable])[[dependent_variable]].max()
  # yMax = np.array(max_results[dependent_variable])
  # print(yMax)

  # results.groupby(by=[independent_variable])[[dependent_variable]].plot.scatter(x=independent_variable, y=dependent_variable)

  if comparison_variable is None:
    label_specification = ''
  else:
    label_specification = labelConversion(comparison_variable, comparison_value)

  ax.plot(results[independent_variable], results[dependent_variable], markersize=10, c=colour_light, marker=marker, alpha=0.3, linewidth=2, edgecolors='face', label=label_specification)
  # jitter(x=results[independent_variable], y=results[dependent_variable], position=position, s=50, c=colour_light, marker=marker, alpha=0.3, linewidths=2, edgecolors='face', label=label_specification)

  ax.plot(x, y, linestyle=line, linewidth=3, marker=marker,  markersize=15, alpha=1.0, color=colour_dark, label='Mean ' + label_specification)
  if error_bar:
    plt.errorbar(x=x, y=y, yerr=yerror, capsize=5, capthick=2, linewidth=1, color=colour_light,
                 alpha=0.5, label='Mean ' + label_specification)
  plt.fill_between(x=x, y1=yMin, y2=yMax, alpha=0.3, color=colour_light,
                   label='Standard deviation ' + label_specification)

  print(' finished', label_specification)


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
  plt.savefig(path + 'sigma_boxplot.png', pad_inches=0.1, dpi=300)
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
  plt.savefig(path + filename + '.png', pad_inches=0.1, dpi=300)
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
  plt.savefig(path + filename + '.png', pad_inches=0.1, dpi=300)
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
    plt.savefig(path + filename + '(' + str(np.where(random_i == i)[0][0]) + ').png', pad_inches=0.1, dpi=300)
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
  plt.savefig(path + independent_variable + '_boxplot_' + dependent_variable + '.png', pad_inches=0.1, dpi=300)
  plt.close()


def multipleBoxplots(data, comparison_variable, independent_variable, dependent_variable):
  """
  Plot the progression of engaged agents for multiple simulations.

  :param data: The results from a batch run.
  :param comparison_variable:
  :param independent_variable:
  :param dependent_variable:
  """

  plt.style.use(style)

  fig = plt.figure(figsize=(15, 10))

  grouped_results = data.groupby(by=["RunId"]).max()

  bp = sns.boxplot(x=independent_variable, y=dependent_variable, hue=comparison_variable, data=grouped_results)  # , rot=45)

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

  plt.show()

  # plt.savefig(path + independent_variable + '_mutipleBoxplots_' + dependent_variable + '.png', pad_inches=0.1, dpi=300)
  # plt.close()
