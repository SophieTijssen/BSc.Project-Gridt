import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.ticker as mtick
from matplotlib.pylab import MaxNLocator
import matplotlib.transforms as transforms
from matplotlib.legend_handler import HandlerBase

from utilities.network_util import NetworkType, KnowledgeType, getComparisonValue, getComparisonValueName
from utilities.model_util import sigmas1, sigmas2, nums, out_degrees, getVariableName, RunType

style = 'seaborn-poster'
# title_size = 32
axis_label_size = 32
legend_label_size = 26
tick_size = 24
colour1 = '#D7191C'
bg_colour1 = '#f07b7d'
colour2 = '#2C7BB6'
bg_colour2 = '#74b1dd'
line1 = '-'
line2 = '--'
marker1 = 'o'
marker2 = 's'

steps_axis = 'Steps'
engagement_axis = 'Equilibrium of adopting agents'
engagement_axis_single = 'Percentage of adopting agents'
diffusion_axis = 'Average diffusion rate (nodes/step)'
sigma_axis = 'Standard deviation of decision threshold distribution'
n_axis = 'Population size'
out_degree_axis = 'Out-degree'

figure_size = (15, 10)
dpi = 400


def removeItemsFromList(positions, items, deletes):
  for item in deletes:
    indexes, = np.where(items == item)
    positions = np.delete(positions, indexes)
  return positions


def getAxisLabel(variable):
  if variable == 'Step':
    return steps_axis

  elif variable == 'engagement_ratio':
    return engagement_axis

  elif variable == 'diffusion_rate':
    return diffusion_axis

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


class HandlerBoxPlot(HandlerBase):
  def create_artists(self, legend, orig_handle,
                     xDescent, yDescent, width, height, fontsize,
                     trans):
    """
        Source: https://stackoverflow.com/a/59276383
        :param legend:
        :param orig_handle:
        :param xDescent:
        :param yDescent:
        :param width:
        :param height:
        :param fontsize:
        :param trans:
        :return:
        """
    lw = 1.5
    a_list = [matplotlib.lines.Line2D(np.array([0, 0, 1, 1, 0]) * width - xDescent,
                                      np.array([0.25, 0.75, 0.75, 0.25, 0.25]) * height - yDescent, lw=lw),
              matplotlib.lines.Line2D(np.array([0.5, 0.5]) * width - xDescent,
                                      np.array([0.75, 1]) * height - yDescent, lw=lw),
              matplotlib.lines.Line2D(np.array([0.5, 0.5]) * width - xDescent,
                                      np.array([0.25, 0]) * height - yDescent, lw=lw),
              matplotlib.lines.Line2D(np.array([0.25, 0.75]) * width - xDescent,
                                      np.array([1, 1]) * height - yDescent, lw=lw),
              matplotlib.lines.Line2D(np.array([0.25, 0.75]) * width - xDescent,
                                      np.array([0, 0]) * height - yDescent, lw=lw),
              matplotlib.lines.Line2D(np.array([0, 1]) * width - xDescent,
                                      np.array([0.5, 0.5]) * height - yDescent, lw=lw)]

    for a in a_list:
      a.set_color(orig_handle.get_color())
    return a_list


def prepareDataMultipleBoxplots(df, comparison_variable, comparison_values, independent_variable, dependent_variable):
  if 'AgentID' in df:
    results1 = df.loc[df['AgentID'] == 0]
    results = results1.drop(['AgentID', 'state'], axis=1)
  else:
    results = df

  grouped_results = results.groupby([comparison_variable, independent_variable])[dependent_variable].apply(
    list).groupby(comparison_variable).apply(list)
  # print(grouped_results)
  data_0 = grouped_results[comparison_values[0]]
  data_1 = grouped_results[comparison_values[1]]

  ticks = results[independent_variable].unique()

  return data_0, data_1, ticks


def createMultipleBoxplots(results, comparison_variable, independent_variable, dependent_variable, path, filename,
                           sigma_identifier=''):
  """

  source: https://stackoverflow.com/a/20132614
  """
  comparison_values = getComparisonValue(comparison_variable)

  data_0, data_1, ticks = prepareDataMultipleBoxplots(results, comparison_variable, comparison_values,
                                                      independent_variable,
                                                      dependent_variable)

  # function for setting the colors of the box plots pairs
  # def set_box_color(bp, color):
  #   plt.setp(bp['boxes'], color=color)
  #   plt.setp(bp['whiskers'], color=color)
  #   plt.setp(bp['caps'], color=color)
  #   plt.setp(bp['medians'], color=color)

  def set_box_color(bp, colours, marker, line):
    colour, bg_colour = colours
    plt.setp(bp['boxes'], color=colour, linestyle=line)
    plt.setp(bp['medians'], color=colour, linestyle=line)
    plt.setp(bp['fliers'], markeredgecolor=colour, markerfacecolor=bg_colour, marker=marker)
    plt.setp(bp['whiskers'], color=colour, linestyle=line)
    plt.setp(bp['caps'], color=colour, linestyle=line)

  plt.style.use(style)

  fig, ax = plt.subplots(figsize=figure_size)
  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  bpl = plt.boxplot(data_0, positions=np.array(range(len(data_0))) * 2.0 - 0.4, sym='', widths=0.6,
                    medianprops=dict(linestyle='-', linewidth=2.5, color=colour1))
  bpr = plt.boxplot(data_1, positions=np.array(range(len(data_1))) * 2.0 + 0.4, sym='', widths=0.6,
                    medianprops=dict(linestyle='-', linewidth=2.5, color=colour2))
  # set_box_color(bpl, colour1)  # colors are from http://colorbrewer2.org/
  set_box_color(bpl, colours=[colour1, '#f4a0a2'], marker=marker1, line=line1)
  set_box_color(bpr, colours=[colour2, '#99c5e6'], marker=marker2, line=line2)
  # set_box_color(bpr, colour2)

  # draw temporary red and blue lines and use them to create a legend
  plt.plot([], c=colour1, linestyle=line1, label=labelConversion(comparison_variable, comparison_values[0]))
  plt.plot([], c=colour2, linestyle=line2, label=labelConversion(comparison_variable, comparison_values[1]))
  plt.legend(prop={'size': legend_label_size})

  if dependent_variable == 'engagement_ratio':
    # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
    # https://matplotlib.org/stable/api/ticker_api.html
    ax.get_yaxis().set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

  if independent_variable == 'sigma':
    # ax.get_xaxis().set_ticks(np.arange(0.0, max(sigmas) + 0.1, 0.1))
    # plt.xticks(range(0, len(data_0), 25), np.arange(0.0, max(ticks) + 0.1, 0.25))
    # plt.xlim(-5, len(ticks) + 5)
    # print(np.arange(0.1, max(ticks) + 0.1, 0.1))
    if sigma_identifier == '10':
      plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    elif sigma_identifier == '100':
      # TODO: Change on 10/01/2024 ---
      print(list(range(20, len(ticks) * 2 + 20, 20)))
      print(np.arange(0.1, max(ticks) + 0.1, 0.1).round(decimals=1))
      plt.xticks(range(20, len(ticks) * 2 + 20, 20), np.arange(0.1, max(ticks) + 0.1, 0.1).round(decimals=1))
      # TODO: -----
    # plt.xticks(range(0, len(ticks) * 2, 2), ticks)

  elif independent_variable == 'num_of_nodes':
    # ax.get_xaxis().set_ticks(
    #   data_1[1][independent_variable].unique())  # TODO: change to this statement below when rerunning simulations
    # # ax.get_xaxis().set_ticks(range(min(nums), max(nums) + 1, 10))
    # # ax.get_xaxis().set_ticks(nums)
    # ax.get_xaxis().set_ticks(range(10, 210, 10))
    plt.xticks(range(0, len(ticks) * 2, 4), range(min(ticks), max(ticks) + 5, 10))
    plt.xlim(-2, len(ticks) * 2)
  elif independent_variable == 'out_degree':
    # ax.get_xaxis().set_ticks(out_degrees)
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)

  plt.xlabel(getAxisLabel(independent_variable), size=axis_label_size)
  plt.ylabel(getAxisLabel(dependent_variable), size=axis_label_size)
  ax.tick_params(axis='both', labelsize=tick_size)

  # plt.xticks(range(0, len(ticks) * 2, 2), ticks)
  # plt.xlim(-2, len(ticks) * 2)

  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  plt.tight_layout()
  plt.savefig(path + filename + '.png', pad_inches=0.1, dpi=dpi)
  plt.close()


def prepareDataGranovetter(df, independent_variable, dependent_variable):
  if 'AgentID' in df:
    results1 = df.loc[df['AgentID'] == 0]
    results = results1.drop(['AgentID', 'state'], axis=1)
  else:
    results = df

  grouped_results = results.groupby([independent_variable])[dependent_variable]

  ticks = results[independent_variable].unique()

  # print(grouped_results)

  return grouped_results, ticks


def createGranovetterBoxplots(path, results):
  """

  source: https://stackoverflow.com/a/20132614
  """

  independent_variable = 'sigma'
  dependent_variable = 'engagement_ratio'
  filename = "granovetter_sigma_boxplots"

  data, ticks = prepareDataGranovetter(results, independent_variable, dependent_variable)

  # function for setting the colors of the box plots pairs
  # def set_box_color(bp, color):
  #   plt.setp(bp['boxes'], color=color)
  #   plt.setp(bp['whiskers'], color=color)
  #   plt.setp(bp['caps'], color=color)
  #   plt.setp(bp['medians'], color=color)

  def set_box_color(bp, colours, marker, line):
    colour, bg_colour = colours
    plt.setp(bp['boxes'], color=colour, linestyle=line)
    plt.setp(bp['medians'], color=colour, linestyle=line)
    plt.setp(bp['fliers'], markeredgecolor=colour, markerfacecolor=bg_colour, marker=marker)
    plt.setp(bp['whiskers'], color=colour, linestyle=line)
    plt.setp(bp['caps'], color=colour, linestyle=line)

  plt.style.use(style)

  fig, ax = plt.subplots(figsize=figure_size)
  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  bpl = plt.boxplot(data.apply(list), positions=np.array(range(len(data))), sym='', widths=0.6,
                    medianprops=dict(linestyle='-', color=colour1))
  # set_box_color(bpl, colour1)  # colors are from http://colorbrewer2.org/
  set_box_color(bpl, colours=[colour1, '#f4a0a2'], marker=marker1, line=line1)
  bpl_line, = ax.plot([], c=colour1, label="Boxplot engagement ratio")

  plt.plot(np.array(range(len(data))), data.median(), color=colour1, label="Median engagement ratio")

  plt.axvline(x=12.2, linestyle='dashed', color=colour2, linewidth=2)
  plt.axhline(y=0.5, linestyle='dashed', color=colour2, linewidth=2)

  # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # https://matplotlib.org/stable/api/ticker_api.html
  ax.get_yaxis().set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
  plt.xticks(range(0, len(data), 25), np.arange(0.0, max(ticks) + 0.1, 0.25))
  plt.xlim(-5, len(ticks) + 5)

  plt.xlabel(sigma_axis, size=axis_label_size)
  plt.ylabel(engagement_axis, size=axis_label_size)
  ax.tick_params(axis='both', labelsize=tick_size)

  # https://stackoverflow.com/a/42879040
  trans_x = transforms.blended_transform_factory(ax.get_xticklabels()[0].get_transform(), ax.transData)
  trans_y = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
  ax.text(0, 0.5, "50%", color="#2C7BB6", transform=trans_y, ha="right", va="center", size=tick_size)
  ax.text(12.2, -.094, "0.12", color="#2C7BB6", transform=trans_x, ha="center", va="center", size=tick_size)

  ax.legend(handler_map={bpl_line: HandlerBoxPlot()}, handleheight=3, prop={'size': legend_label_size})

  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  plt.tight_layout()

  plt.savefig(path + filename + '.png', pad_inches=0.1, dpi=dpi)
  plt.close()


def generateFigures(run_type, n=100, out_degree=3):
  if run_type == RunType.Granovetter:
    path_data = 'results/raw_data/granovetter/'
    path_figure = 'results/figures/granovetter/'

    # Normal distribution with varying sigmas experiment
    results = pd.read_csv(path_data + 'sigma.csv')
    createGranovetterBoxplots(path_figure, results)

  else:

    if run_type == RunType.KnowledgeComparison:
      path_data = 'results/raw_data/knowledge_comparison/'
      path_figure = 'results/figures/knowledge_comparison/'
      comparison_variable = 'knowledge'
    else:  # run_type == RunType.NetworkComparison
      path_data = 'results/raw_data/network_comparison/'
      path_figure = 'results/figures/network_comparison/'
      comparison_variable = 'networkType'

    # Standard comparison simulations on normal distribution
    normal_csv_file = comparison_variable + "_comparison_normal.csv"
    results_normal = pd.read_csv(path_data + normal_csv_file)

    for dependent_variable in ['engagement_ratio', 'diffusion_rate']:
      createSingleBoxplots(results=results_normal, comparison_variable=comparison_variable,
                           dependent_variable=dependent_variable,
                           path=path_figure, filename=comparison_variable + '_' + dependent_variable)

    # Alternate independent variable simulations
    for csv_file, independent_variable, file_identifier in [('out-degree_comparison.csv', 'out_degree', ''),
                                                            ('n_comparison.csv', 'num_of_nodes', ''),
                                                            ('sigma10_comparison.csv', 'sigma', '10'),
                                                            ('sigma100_comparison.csv', 'sigma', '100')]:
      results = pd.read_csv(path_data + csv_file)

      for dependent_variable in ['engagement_ratio', 'diffusion_rate']:
        createMultipleBoxplots(results=results, comparison_variable=comparison_variable,
                               independent_variable=independent_variable, dependent_variable=dependent_variable,
                               path=path_figure, filename=comparison_variable + '_' + independent_variable + file_identifier + '_' + dependent_variable,
                               sigma_identifier=file_identifier)


def showDegreeHistogram(path, G, specification):
  """
  Plot a degree histogram showing the in- and out-degree.

  :param path:
  :param G: The network used in the mesa model.
  :param specification:
  """

  plt.style.use(style)

  fig, ax = plt.subplots(figsize=figure_size)

  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  ins = [val for (node, val) in G.in_degree()]
  outs = [val for (node, val) in G.out_degree()]

  d1 = np.array(ins)
  d2 = np.array(outs)

  maxValue = max(max(ins), max(outs))
  bins = np.arange(maxValue) - 0.5

  _, final_bins, _ = ax.hist([d1, d2], bins=bins, label=['in-degrees', 'out-degrees'], color=[bg_colour1, bg_colour2])

  plt.xticks(range(maxValue))
  plt.xlim([-0.5, max(final_bins)])
  plt.ylim([0, G.number_of_nodes()])

  # plt.title('Histogram of degree distribution in the network', size=title_size)
  plt.xlabel('Degree size', size=axis_label_size)
  plt.ylabel('Frequency', size=axis_label_size)
  for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(tick_size)

  plt.legend(prop={'size': legend_label_size})

  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  plt.tight_layout()
  plt.savefig(path + 'degree_histogram_' + specification + '.png', pad_inches=0.1, dpi=dpi)
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

  fig, ax = plt.subplots(figsize=figure_size)

  # # Median
  # median_results = results.groupby(by=["sigma"]).median()[['engagement_ratio']]
  # median_results.plot()
  #
  # fig = median_results.plot(color='#EE0000')
  # plt.axvline(x=0.12, linestyle='dashed', color='gray')
  #
  # # Box plots
  # grouped_results = results.groupby(by=["RunId"]).median()
  # plt.boxplot(x='sigma', y='engagement_ratio', data=grouped_results, rot=45)

  # plt.title('Median agent engagement for normal distributions with varying sigmas')
  # plt.xlabel('Sigma')
  # plt.ylabel('Percentage of engaged agents')
  #
  # # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # # https://matplotlib.org/stable/api/ticker_api.html
  # fig.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

  # plt.show()
  #
  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  mean_sdSubplot(ax, data=(None, results), independent_variable='sigma', dependent_variable='engagement_ratio',
                 comparison_variable=None, graph_style=((colour1, bg_colour1), line1, marker1))

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

  plt.legend(prop={'size': legend_label_size})

  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  plt.tight_layout()
  plt.savefig(path + 'granovetter_sigma.png', pad_inches=0.1, dpi=dpi)
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

  fig, ax = plt.subplots(figsize=figure_size)

  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  mean_sdSubplot(ax, data=data_1, independent_variable=independent_variable, dependent_variable=dependent_variable,
                 comparison_variable=comparison_variable, graph_style=((colour1, bg_colour1), line1, marker1))

  mean_sdSubplot(ax, data=data_2, independent_variable=independent_variable, dependent_variable=dependent_variable,
                 comparison_variable=comparison_variable, graph_style=((colour2, bg_colour2), line2, marker2))

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

  # if independent_variable == 'sigma':
  # ax.get_xaxis().set_ticks(np.arange(0.0, max(sigmas) + 0.1, 0.1))
  elif independent_variable == 'num_of_nodes':
    ax.get_xaxis().set_ticks(
      data_1[1][independent_variable].unique())  # TODO: change to this statement below when rerunning simulations
    # ax.get_xaxis().set_ticks(range(min(nums), max(nums) + 1, 10))
    # ax.get_xaxis().set_ticks(nums)
  elif independent_variable == 'out_degree':
    ax.get_xaxis().set_ticks(out_degrees)

  # plt.title('Median agent engagement for normal distributions with varying ' + getAxisLabel(independent_variable),
  #           size=title_size)
  plt.xlabel(getAxisLabel(independent_variable), size=axis_label_size)
  plt.ylabel(getAxisLabel(dependent_variable), size=axis_label_size)
  ax.tick_params(axis='both', labelsize=tick_size)

  plt.legend(prop={'size': legend_label_size})

  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  plt.tight_layout()
  plt.savefig(path + independent_variable + '_comparison_' + getVariableName(dependent_variable) + '.png',
              pad_inches=0.1, dpi=dpi)
  plt.close()

  # print('finished ', dependent_variable)


def mean_sdSubplot(ax, data, independent_variable, dependent_variable, comparison_variable, graph_style,
                   error_bar=True):
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

  ax.plot(results[independent_variable], results[dependent_variable], markersize=10, c=colour_light, marker=marker,
          alpha=0.3, linewidth=2, label=label_specification)  # , edgecolors='face', label=label_specification)
  # jitter(x=results[independent_variable], y=results[dependent_variable], position=position, s=50, c=colour_light, marker=marker, alpha=0.3, linewidths=2, edgecolors='face', label=label_specification)

  ax.plot(x, y, linestyle=line, linewidth=3, marker=marker, markersize=15, alpha=1.0, color=colour_dark,
          label='Mean ' + label_specification)
  if error_bar:
    plt.errorbar(x=x, y=y, yerr=yerror, capsize=5, capthick=2, linewidth=1, color=colour_light,
                 alpha=0.5, label='Mean ' + label_specification)
  plt.fill_between(x=x, y1=yMin, y2=yMax, alpha=0.3, color=colour_light,
                   label='Standard deviation ' + label_specification)

  # print(' finished', label_specification)


def sigmaBoxPlot(path, results):
  """
  Plot a boxplot of the engagement equilibrium for every sigma.

  :param path:
  :param results: The results from a batch run.
  """

  plt.style.use(style)

  # fig = plt.figure(figsize=figure_size)
  # results.groupby(by=["RunId"]).median().boxplot(by='sigma', column=['engagement_ratio'], grid=False, rot=45)
  grouped_results = results.groupby(by=["RunId"]).median()

  plt.boxplot(x='sigma', y='engagement_ratio', data=grouped_results, rot=45)

  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  plt.tight_layout()
  plt.savefig(path + 'sigma_boxplot.png', pad_inches=0.1, dpi=dpi)
  plt.close()


def singleRunPlot(path, results, filename):
  """
  Plot the progression of engaged agents during a single simulation.

  :param path:
  :param results: The results from a single run.
  :param filename:
  """

  plt.style.use(style)

  fig, ax = plt.subplots(figsize=figure_size)

  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  data = results['engagement_ratio']
  plt.plot(data, marker='o', markersize=8, color='#EE0000', label='Engagement ratio')

  # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
  # https://matplotlib.org/stable/api/ticker_api.html
  ax.get_yaxis().set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
  ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
  ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))

  x_max = len(data) - 1
  y_max = max(data)

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
  plt.ylabel(engagement_axis_single, size=axis_label_size)
  ax.tick_params(axis='both', labelsize=tick_size)

  plt.legend(prop={'size': legend_label_size})

  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  plt.tight_layout()
  plt.savefig(path + filename + '.png', pad_inches=0.1, dpi=dpi)
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

  fig, ax = plt.subplots(figsize=figure_size)

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

  plt.legend(prop={'size': legend_label_size})

  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  plt.tight_layout()
  plt.savefig(path + filename + '.png', pad_inches=0.1, dpi=dpi)
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
    fig, ax = plt.subplots(figsize=figure_size)

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

    plt.legend(prop={'size': legend_label_size})

    plt.gcf().set_size_inches(figure_size[0], figure_size[1])
    plt.tight_layout()
    plt.savefig(path + filename + '(' + str(np.where(random_i == i)[0][0]) + ').png', pad_inches=0.1, dpi=dpi)
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

  fig = plt.figure(figsize=figure_size)

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

  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  plt.tight_layout()
  plt.savefig(path + independent_variable + '_boxplot_' + dependent_variable + '.png', pad_inches=0.1, dpi=dpi)
  plt.close()


# def multipleBoxplots(data, comparison_variable, independent_variable, dependent_variable):
#   """
#   Plot the progression of engaged agents for multiple simulations.
#
#   :param data: The results from a batch run.
#   :param comparison_variable:
#   :param independent_variable:
#   :param dependent_variable:
#   """
#
#   plt.style.use(style)
#
#   fig = plt.figure(figsize=figure_size)
#
#   grouped_results = data.groupby(by=["RunId"]).max()
#
#   bp = sns.boxplot(x=independent_variable, y=dependent_variable, hue=comparison_variable,
#                    data=grouped_results)  # , rot=45)
#
#   # new_results = results.groupby(by=["RunId"]).max()
#   # bp = new_results.boxplot(by=independent_variable, column=[dependent_variable], grid=False, rot=45)
#
#   # plt.title(title + " (variable: " + str(independent_variable) + ")", size=title_size)
#   fig.texts = []
#   plt.xlabel(getAxisLabel(independent_variable), size=axis_label_size)
#   plt.ylabel(getAxisLabel(dependent_variable), size=axis_label_size)
#   bp.axes.tick_params(axis='both', labelsize=tick_size)
#
#   x_ticks = bp.get_xticks()
#   plt.xticks(x_ticks, labels=[labelConversion(independent_variable, bool(x)) for x in x_ticks], rotation=0)
#   bp.axes.get_yaxis().set_major_locator(MaxNLocator(integer=True))
#
#   plt.gcf().set_size_inches(figure_size[0], figure_size[1])
#   plt.tight_layout()
#
#   plt.show()
#
#   # plt.savefig(path + independent_variable + '_mutipleBoxplots_' + dependent_variable + '.png', pad_inches=0.1, dpi=dpi)
#   # plt.close()


def prepareDataSingleBoxplots(df, comparison_variable, comparison_values, dependent_variable):
  if 'AgentID' in df:
    results1 = df.loc[df['AgentID'] == 0]
    results = results1.drop(['AgentID', 'state'], axis=1)
  else:
    results = df

  grouped_results = results.groupby([comparison_variable])[dependent_variable].apply(list)
  # print(grouped_results)
  data_0 = grouped_results[comparison_values[0]]
  data_1 = grouped_results[comparison_values[1]]

  return [data_0, data_1]


def createSingleBoxplots(results, comparison_variable, dependent_variable, path, filename):
  """

  source: https://stackoverflow.com/a/20132614
  """

  comparison_values = getComparisonValue(comparison_variable)

  data = prepareDataSingleBoxplots(results, comparison_variable, comparison_values, dependent_variable)

  # function for setting the colors of the box plots pairs
  def set_box_color(bp, box, whiskers, colours, marker, line):
    colour, bg_colour = colours
    plt.setp(bp['boxes'][box], color=colour, linestyle=line, linewidth=3)
    # bp['boxes'][box].set_facecolor(bg_colour)
    plt.setp(bp['medians'][box], color=colour, linestyle=line, linewidth=5)
    plt.setp(bp['fliers'][box], markeredgecolor=colour, markersize=8, markerfacecolor=bg_colour, marker=marker)
    for w in whiskers:
      plt.setp(bp['whiskers'][w], color=colour, linestyle=line, linewidth=3)
      plt.setp(bp['caps'][w], color=colour, linestyle=line, linewidth=3)

  plt.style.use(style)

  fig, ax = plt.subplots(figsize=figure_size)
  plt.grid(linestyle='--', linewidth=1.0, alpha=1.0)

  bpl = plt.boxplot(data, positions=np.array(range(len(data))), patch_artist=True,
                    widths=0.8)  # , boxprops=dict(facecolor='#f07b7d', color='#D7191C'))  #, medianprops=dict(linestyle='-', color='#D7191C'))

  # Design boxplots
  set_box_color(bpl, box=0, whiskers=[0, 1], colours=[colour1, '#f4a0a2'], marker=marker1, line=line1)
  set_box_color(bpl, box=1, whiskers=[2, 3], colours=[colour2, '#99c5e6'], marker=marker2, line=line2)

  for patch, color in zip(bpl['boxes'], [bg_colour1, bg_colour2]):
    patch.set_facecolor(color)

  # Set axis ticks, labels and layout
  if dependent_variable == 'engagement_ratio':
    # https://stackoverflow.com/questions/62610215/percentage-sign-in-matplotlib-on-y-axis
    # https://matplotlib.org/stable/api/ticker_api.html
    ax.get_yaxis().set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

  xticks_labels = [getComparisonValueName(comparison_variable=comparison_variable, value=comparison_values[0]),
                   getComparisonValueName(comparison_variable=comparison_variable, value=comparison_values[1])]
  plt.xticks(range(len(data)), xticks_labels)
  # plt.xlim(-5, len(ticks) + 5)

  plt.xlabel(getVariableName(comparison_variable), size=axis_label_size)
  plt.ylabel(getAxisLabel(dependent_variable), size=axis_label_size)
  ax.tick_params(axis='both', labelsize=tick_size)

  # Save figure
  plt.gcf().set_size_inches(figure_size[0], figure_size[1])
  plt.tight_layout()

  plt.savefig(path + filename + '.png', pad_inches=0.1, dpi=dpi)
  plt.close()
