import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy.stats import kruskal

from runs.run_types import *
from utilities.model_util import sigmas, nums, out_degrees, getVariableName
from utilities.network_util import *


def createDataFrame(independent_variable, values):
  # indexes = [
  #   np.array(["Knowledge type", "Knowledge type", "Knowledge type", "Knowledge type", "Knowledge type", "Knowledge type",
  #             "Network type", "Network type", "Network type", "Network type", "Network type", "Network type"]),
  #   np.array(["Equilibrium", "Equilibrium", "Equilibrium", "Diffusion", "Diffusion", "Diffusion",
  #             "Equilibrium", "Equilibrium", "Equilibrium", "Diffusion", "Diffusion", "Diffusion"]),
  #   np.array(["Levene's Test", "Paired Two-Sample T-test", "Wilcoxon Signed-Rank Test", "Levene's Test", "Paired Two-Sample T-test", "Wilcoxon Signed-Rank Test",
  #             "Levene's Test", "Paired Two-Sample T-test", "Wilcoxon Signed-Rank Test", "Levene's Test", "Paired Two-Sample T-test", "Wilcoxon Signed-Rank Test"])
  # ]
  indexes = [
    np.array(["Behaviour spread", "Behaviour spread", "Behaviour spread", "Diffusion speed", "Diffusion speed", "Diffusion speed"]),
    np.array(["Levene's Test", "Paired Two-Sample T-test", "Wilcoxon Signed-Rank Test", "Levene's Test", "Paired Two-Sample T-test", "Wilcoxon Signed-Rank Test"])
  ]

  columns = [
    # np.array([''] + ([getVariableName(independent_variable)] * (len(values) - 1))),
    np.array([getVariableName(independent_variable)] * len(values)),
    np.array(values),
  ]

  # data = np.random.randn(8, len(values))
  # df1 = pd.DataFrame(data, index=indexes, columns=columns)
  #
  # print(df1.to_string())
  #
  # print(df1.at[('Knowledge type', 'Equilibrium', "Levene's test"), (getVariableName(independent_variable), 0.0)])
  #
  # return df1
  # data = [[(np.nan, np.nan)] * len(values)] * 8
  empty_df = pd.DataFrame(index=indexes, columns=columns)

  # print(empty_df.to_string())

  return empty_df


def paired_t_tests(statistics_df, analysed_statistics_df, filename, comparison_variable, independent_variable):
  # t_tests(path_knowledge + 'knowledge_comparison.csv', 'knowledge',
  #         KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value)

  results = pd.read_csv(filename)
  comparison_df = results.groupby(by=["RunId"]).max()

  comparison_values = getComparisonValue(comparison_variable)
  independent_values = comparison_df[independent_variable].unique()

  # print(independent_values)

  for (result_type) in ['engagement_ratio', 'Step']:

    # print('\n---' + independent_variable + ' comparison for ' + getVariableName(result_type) + '---')

    for value in independent_values:
      if value != 0.0:
        independent_df = comparison_df[comparison_df[independent_variable] == value]

        comparison_group1 = independent_df[independent_df[comparison_variable] == comparison_values[0]]
        comparison_group2 = independent_df[independent_df[comparison_variable] == comparison_values[1]]

        # print(comparison_group1[result_type])
        # print(comparison_group2[result_type])

        # pvalue < 0.05 = variances are significantly different, otherwise they are approximately identical
        levene_test = stats.levene(comparison_group1[result_type], comparison_group2[result_type], center='mean')
        statistics_df.at[(getVariableName(result_type), "Levene's Test"), (getVariableName(independent_variable), value)] = "(statistic=" + str(round(levene_test[0], 4)) + ", pvalue=" + str('{:.4g}'.format(levene_test[1])) + ")"
        # statistics_df.at[(getVariableName(comparison_variable), getVariableName(result_type), "Levene's Test"), (getVariableName(independent_variable), value)] = levene_test
        if levene_test[1] < 0.05:
          analysed_statistics_df.at[(getVariableName(result_type), "Levene's Test"), (getVariableName(independent_variable), value)] = 'Different variances'
          # analysed_statistics_df.at[(getVariableName(comparison_variable), getVariableName(result_type), "Levene's Test"), (getVariableName(independent_variable), value)] = 'Different variances'
        else:
          analysed_statistics_df.at[(getVariableName(result_type), "Levene's Test"), (getVariableName(independent_variable), value)] = 'Equal variances'
          # analysed_statistics_df.at[(getVariableName(comparison_variable), getVariableName(result_type), "Levene's Test"), (getVariableName(independent_variable), value)] = 'Equal variances'

        # print("Levene's Test for " + str(value) + ":")
        # print(levene_test, '\n')

        levene_stat, levene_pvalue = levene_test
        if levene_pvalue > 0.05:
          paired_t_test = stats.ttest_rel(comparison_group1[result_type], comparison_group2[result_type])
          statistics_df.at[(getVariableName(result_type), "Paired Two-Sample T-test"), (getVariableName(independent_variable), value)] = "(statistic=" + str(round(paired_t_test[0], 4)) + ", pvalue=" + str('{:.4g}'.format(paired_t_test[1])) + ")"
          # statistics_df.at[(getVariableName(comparison_variable), getVariableName(result_type), "Paired Two-Sample T-test"), (getVariableName(independent_variable), value)] = paired_t_test
          # print('Paired Two-Sample T-test for ' + str(value) + ':')
          # print(paired_t_test, '\n')
          if paired_t_test[1] < 0.05:
            analysed_statistics_df.at[(getVariableName(result_type), "Paired Two-Sample T-test"), (getVariableName(independent_variable), value)] = 'Statistical difference'
            # analysed_statistics_df.at[(getVariableName(comparison_variable), getVariableName(result_type), "Paired Two-Sample T-test"), (getVariableName(independent_variable), value)] = 'Statistical difference'
          else:
            analysed_statistics_df.at[(getVariableName(result_type), "Paired Two-Sample T-test"), (getVariableName(independent_variable), value)] = 'No difference'
            # analysed_statistics_df.at[(getVariableName(comparison_variable), getVariableName(result_type), "Paired Two-Sample T-test"), (getVariableName(independent_variable), value)] = 'No difference'

        # else:
        wilcoxon_test = stats.wilcoxon(comparison_group1[result_type], comparison_group2[result_type], zero_method="zsplit")
        statistics_df.at[(getVariableName(result_type), "Wilcoxon Signed-Rank Test"), (getVariableName(independent_variable), value)] = "(statistic=" + str(round(wilcoxon_test[0], 4)) + ", pvalue=" + str('{:.4g}'.format(wilcoxon_test[1])) + ")"
        # statistics_df.at[(getVariableName(comparison_variable), getVariableName(result_type), "Wilcoxon Signed-Rank Test"), (getVariableName(independent_variable), value)] = wilcoxon_test
        # print('Wilcoxon Signed-Rank Test for ' + str(value) + ':')
        # print(wilcoxon_test, '\n')
        if wilcoxon_test[1] < 0.05:
          analysed_statistics_df.at[(getVariableName(result_type), "Wilcoxon Signed-Rank Test"), (getVariableName(independent_variable), value)] = 'Statistical difference'
          # analysed_statistics_df.at[(getVariableName(comparison_variable), getVariableName(result_type), "Wilcoxon Signed-Rank Test"), (getVariableName(independent_variable), value)] = 'Statistical difference'
        else:
          analysed_statistics_df.at[(getVariableName(result_type), "Wilcoxon Signed-Rank Test"), (getVariableName(independent_variable), value)] = 'No difference'
          # analysed_statistics_df.at[(getVariableName(comparison_variable), getVariableName(result_type), "Wilcoxon Signed-Rank Test"), (getVariableName(independent_variable), value)] = 'No difference'


def t_tests(path, comparison_variable):
  results = pd.read_csv(path)
  comparison_df = results.groupby(by=["RunId"]).max()

  comparison_values = getComparisonValue(comparison_variable)

  comparison_group1 = comparison_df[comparison_df[comparison_variable] == comparison_values[0]]
  comparison_group2 = comparison_df[comparison_df[comparison_variable] == comparison_values[1]]

  # Equilibrium
  print('\n---' + comparison_variable + ' comparison for equilibrium---')

  t_test_equil = stats.ttest_ind(comparison_group1['engagement_ratio'], comparison_group2['engagement_ratio'])
  print('Independent two-sample t-test:')
  print(t_test_equil, '\n')

  Welch_t_test_equil = stats.ttest_ind(comparison_group1['engagement_ratio'], comparison_group2['engagement_ratio'],
                                       equal_var=False)
  print("Welch's two-sample t-test:")
  print(Welch_t_test_equil, '\n')

  # Diffusion
  print('---' + comparison_variable + ' comparison for diffusion---')

  t_test_diff = stats.ttest_ind(comparison_group1['Step'], comparison_group2['Step'])
  print('Independent two-sample t-test:')
  print(t_test_diff, '\n')

  Welch_t_test_diff = stats.ttest_ind(comparison_group1['Step'], comparison_group2['Step'], equal_var=False)
  print("Welch's two-sample t-test:")
  print(Welch_t_test_diff, '\n')


def testEqualVariance(comparison_df, comparison_variable, independent_variable):
  print('\n--- Test of equal variances for ' + comparison_variable + ' comparison---')

  comparison_values = getComparisonValue(comparison_variable)

  comparison_group1 = comparison_df[comparison_df[comparison_variable] == comparison_values[0]]
  comparison_group2 = comparison_df[comparison_df[comparison_variable] == comparison_values[1]]

  independent_group0 = comparison_df[comparison_df[independent_variable] == 0.0]
  independent_group1 = comparison_df[comparison_df[independent_variable] == 0.1]
  independent_group2 = comparison_df[comparison_df[independent_variable] == 0.2]
  independent_group3 = comparison_df[comparison_df[independent_variable] == 0.3]
  independent_group4 = comparison_df[comparison_df[independent_variable] == 0.4]
  independent_group5 = comparison_df[comparison_df[independent_variable] == 0.5]
  independent_group6 = comparison_df[comparison_df[independent_variable] == 0.6]
  independent_group7 = comparison_df[comparison_df[independent_variable] == 0.7]
  independent_group8 = comparison_df[comparison_df[independent_variable] == 0.8]
  independent_group9 = comparison_df[comparison_df[independent_variable] == 0.9]
  independent_group10 = comparison_df[comparison_df[independent_variable] == 1.0]

  print(comparison_df.sigma.unique())

  group10 = comparison_group1[comparison_group1[independent_variable] == 0.0]
  group11 = comparison_group1[comparison_group1[independent_variable] == 0.1]
  group12 = comparison_group1[comparison_group1[independent_variable] == 0.2]
  group13 = comparison_group1[comparison_group1[independent_variable] == 0.3]
  group14 = comparison_group1[comparison_group1[independent_variable] == 0.4]
  group15 = comparison_group1[comparison_group1[independent_variable] == 0.5]
  group16 = comparison_group1[comparison_group1[independent_variable] == 0.6]
  group17 = comparison_group1[comparison_group1[independent_variable] == 0.7]
  group18 = comparison_group1[comparison_group1[independent_variable] == 0.8]
  group19 = comparison_group1[comparison_group1[independent_variable] == 0.9]
  group110 = comparison_group1[comparison_group1[independent_variable] == 1.0]

  group20 = comparison_group2[comparison_group2[independent_variable] == 0.0]
  group21 = comparison_group2[comparison_group2[independent_variable] == 0.1]
  group22 = comparison_group2[comparison_group2[independent_variable] == 0.2]
  group23 = comparison_group2[comparison_group2[independent_variable] == 0.3]
  group24 = comparison_group2[comparison_group2[independent_variable] == 0.4]
  group25 = comparison_group2[comparison_group2[independent_variable] == 0.5]
  group26 = comparison_group2[comparison_group2[independent_variable] == 0.6]
  group27 = comparison_group2[comparison_group2[independent_variable] == 0.7]
  group28 = comparison_group2[comparison_group2[independent_variable] == 0.8]
  group29 = comparison_group2[comparison_group2[independent_variable] == 0.9]
  group210 = comparison_group2[comparison_group2[independent_variable] == 1.0]

  result = 'engagement_ratio'

  for result, name in [('engagement_ratio', 'Equilibrium'), ('Step', 'Diffusion')]:
    levene_test_equil = stats.levene(comparison_group1[result], comparison_group2[result])
    print(name)
    print(levene_test_equil, '\n')

    levene_test_equil_indep = stats.levene(group10[result], group11[result], group12[result],
                                           group13[result], group14[result],
                                           group15[result], group16[result],
                                           group17[result], group18[result],
                                           group19[result], group110[result], group20[result],
                                           group21[result], group22[result],
                                           group23[result], group24[result],
                                           group25[result], group26[result],
                                           group27[result], group28[result],
                                           group29[result], group210[result])
    print('Equal variances for comparison and independent variables:')
    print(levene_test_equil_indep, '\n')

    levene_test_equil_group1_indep = stats.levene(group10[result], group11[result], group12[result],
                                                  group13[result], group14[result],
                                                  group15[result], group16[result],
                                                  group17[result], group18[result],
                                                  group19[result], group110[result])
    print('Comparison group 1 - equal variances for independent variable:')
    print(levene_test_equil_group1_indep, '\n')

    levene_test_equil_group2_indep = stats.levene(group20[result], group21[result], group22[result],
                                                  group23[result], group24[result],
                                                  group25[result], group26[result],
                                                  group27[result], group28[result],
                                                  group29[result], group210[result])
    print('Comparison group 2 - equal variances for independent variable:')
    print(levene_test_equil_group2_indep, '\n')

  # print(comparison_variable, ": ", stats.friedmanchisquare(comparison_group1, comparison_group2))

  # print(independent_variable, ": ", stats.friedmanchisquare(independent_group0[result], independent_group1[result],
  #                                                     independent_group2[result], independent_group3[result],
  #                                                     independent_group4[result], independent_group5[result],
  #                                                     independent_group6[result], independent_group7[result],
  #                                                     independent_group8[result], independent_group9[result],
  #                                                     independent_group10[result]))
  #
  # print(comparison_variable, " + ", independent_variable, ": ",
  #       stats.friedmanchisquare(group10[result], group11[result], group12[result], group13[result], group14[result],
  #                               group15[result], group16[result], group17[result], group18[result], group19[result],
  #                               group110[result], group20[result], group21[result], group22[result], group23[result],
  #                               group24[result], group25[result], group26[result], group27[result], group28[result],
  #                               group29[result], group210[result]))


def anova_tests(path, independent_variable, comparison_variable):
  results = pd.read_csv(path)
  comparison_df = results.groupby(by=["RunId"]).max()[
    [comparison_variable, independent_variable, 'Step', 'engagement_ratio']]
  print(comparison_df)
  # comparison_df_equil = results.groupby(by=["RunId"]).max()[[comparison_variable, independent_variable, 'engagement_ratio']]
  # comparison_df_diff = results.groupby(by=["RunId"]).max()[[comparison_variable, independent_variable, 'Step']]
  #
  # Test equal variances
  testEqualVariance(comparison_df, comparison_variable, independent_variable)
  #
  # print('\n---' + comparison_variable + ' comparison for in-degree---')
  #
  # # Equilibrium
  # model_equil = ols('engagement_ratio ~ C(' + comparison_variable + ') + C(' + independent_variable + ') + C(' +
  #                   comparison_variable + '):C(' + independent_variable + ')', data=comparison_df).fit()
  # anova_equil = sm.stats.anova_lm(model_equil, typ=2)
  # print("Two-sample anova for equilibrium:")
  # print(anova_equil, '\n')
  #
  # # stat_equil, p_val_equil = kruskal(comparison_df_equil, nan_policy='omit')
  # # print(f"Kruskal-Wallis test statistic: {stat_equil:.4f}, p-value: {p_val_equil:.4f}")
  #
  # # Diffusion
  # model_diff = ols('Step ~ C(' + comparison_variable + ') + C(' + independent_variable + ') + C(' +
  #                  comparison_variable + '):C(' + independent_variable + ')', data=comparison_df).fit()
  # anova_diff = sm.stats.anova_lm(model_diff, typ=2)
  # print("Two-sample anova for diffusion:")
  # print(anova_diff, '\n')
  #
  # # stat_diff, p_val_diff = kruskal(comparison_df_diff, nan_policy='omit')
  # # print(f"Kruskal-Wallis test statistic: {stat_diff:.4f}, p-value: {p_val_diff:.4f}")


def statistical_tests():
  info_knowledge = ('results/raw_data/knowledge_comparison/', 'knowledge_comparison.csv', 'knowledge')
  info_network = ('results/raw_data/network_comparison/', 'network_comparison.csv', 'networkType')

  sigma_info = ('sigma_comparison.csv', 'sigma', list(sigmas))
  num_of_nodes_info = ('n_comparison.csv', 'num_of_nodes', nums)
  out_degree_info = ('out-degree_comparison.csv', 'out_degree', out_degrees)
  
  for (file, independent_variable, values) in [sigma_info]:  # , num_of_nodes_info, out_degree_info]:
    # statistics_df = createDataFrame(independent_variable, values)
    # analysed_statistics_df = createDataFrame(independent_variable, values)

    for (path, file_general, comparison_variable) in [info_knowledge, info_network]:

      # print(path + file_general, comparison_variable)

      statistics_df = createDataFrame(independent_variable, values)
      analysed_statistics_df = createDataFrame(independent_variable, values)

      # t_tests(path + file_general, comparison_variable, comparison_value1, comparison_value2)
      #
      # anova_tests(path + 'out-degree_comparison.csv', 'out_degree', comparison_variable)

      # Paired t-tests
      paired_t_tests(statistics_df, analysed_statistics_df, path + file, comparison_variable, independent_variable)

      print("Statistical analysis for " + comparison_variable)
      print("------------------------------------")
      print(statistics_df.transpose().style.to_latex())
      print(statistics_df.transpose().to_string())
      print(analysed_statistics_df.to_string())
  # # T-tests
  # t_tests(path_knowledge + 'knowledge_comparison.csv', 'knowledge',
  #         KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value)
  #
  # t_tests(path_network + 'network_comparison.csv', 'networkType',
  #         NetworkType.Directed.value, NetworkType.Undirected.value)

  # # ANOVA test
  # anova_tests(path_knowledge + 'sigma_comparison.csv', 'sigma', 'knowledge')

  # anova_tests(path_network + 'sigma_comparison.csv', 'sigma', 'networkType')
