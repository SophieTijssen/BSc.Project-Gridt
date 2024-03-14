import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import pingouin as pt

from results.plot_graphs import labelConversion
from runs.run_types import *
from utilities.model_util import sigmas1, sigmas2, nums, out_degrees, getVariableName, getRuntypeName
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
    np.array(["Behaviour spread", "Behaviour spread", "Behaviour spread", "Behaviour spread", "Behaviour spread", "Diffusion speed", "Diffusion speed", "Diffusion speed", "Diffusion speed", "Diffusion speed"]),
    np.array(["Levene's Test", "Shapiro-Wilks test 1", "Shapiro-Wilks test 2", "Paired Two-Sample T-test", "Wilcoxon Signed-Rank Test", "Levene's Test", "Shapiro-Wilks test 1", "Shapiro-Wilks test 2", "Paired Two-Sample T-test", "Wilcoxon Signed-Rank Test"])
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


def prepareData(df, comparison_variable, comparison_values, independent_variable, dependent_variable):
  if 'AgentID' in df:
    results1 = df.loc[df['AgentID'] == 0]
    results = results1.drop(['AgentID', 'state'], axis=1)
  else:
    results = df

  useful_results = results[[comparison_variable, independent_variable, dependent_variable]]

  comparison_group1 = useful_results[useful_results[comparison_variable] == comparison_values[0]]
  comparison_group2 = useful_results[useful_results[comparison_variable] == comparison_values[1]]

  # data_1 = comparison_group1.groupby([independent_variable])[dependent_variable].apply(list)  # .groupby(comparison_variable).apply(list)
  # data_2 = comparison_group2.groupby([independent_variable])[dependent_variable].apply(list)  # .groupby(comparison_variable).apply(list)
  data_1 = comparison_group1.groupby([independent_variable])[dependent_variable]
  data_2 = comparison_group2.groupby([independent_variable])[dependent_variable]

  # for value, group in [(comparison_values[0], data_1), (comparison_values[1], data_2)]:
  #   print(labelConversion(comparison_variable, value))
  #   print(group.describe())

  return data_1.apply(list), data_2.apply(list)


def multiple_paired_t_tests(statistics_df, analysed_statistics_df, results, comparison_variable, independent_variable, dependent_variable):
  comparison_values = getComparisonValue(comparison_variable)
  independent_values = results[independent_variable].unique()

  # print(results)

  # useful_results = results[[comparison_variable, independent_variable, dependent_variable]]

  # comparison_group1 = useful_results[useful_results[comparison_variable] == comparison_values[0]]
  # comparison_group2 = useful_results[useful_results[comparison_variable] == comparison_values[1]]
  # print(comparison_group1)
  # print(comparison_group2)

  comparison_groups1, comparison_groups2 = prepareData(results, comparison_variable, comparison_values, independent_variable, dependent_variable)
  # print(comparison_group1)
  # print(comparison_group2)

  for value in independent_values:
    # TODO: Remove below line
    if independent_variable != 'sigma' or value >= 0.1:
      # independent_df = results[results[independent_variable] == value]
      # print(independent_df)

      # comparison_group1 = independent_df[independent_df[comparison_variable] == comparison_values[0]]
      # comparison_group2 = independent_df[independent_df[comparison_variable] == comparison_values[1]]
      # print(comparison_group1)
      # print(comparison_group2)
      # print(comparison_group1[dependent_variable])
      # print(comparison_group2[dependent_variable])

      group1 = comparison_groups1[value]
      group2 = comparison_groups2[value]

      # pvalue <= 0.05 = variances are significantly different, otherwise they are approximately identical
      # print('test1')
      # levene_test = stats.levene(group1, group2, center='median')
      # # print('test1')
      # statistics_df.at[(getVariableName(dependent_variable), "Levene's Test"), (getVariableName(independent_variable), value)] = "(statistic=" + str(round(levene_test[0], 4)) + ", pvalue=" + str('{:.4g}'.format(levene_test[1])) + ")"
      # if levene_test[1] <= 0.05:
      #   analysed_statistics_df.at[(getVariableName(dependent_variable), "Levene's Test"), (getVariableName(independent_variable), value)] = 'Different variances'
      # else:
      #   analysed_statistics_df.at[(getVariableName(dependent_variable), "Levene's Test"), (getVariableName(independent_variable), value)] = 'Equal variances'

      # pvalue <= 0.05 = variances are significantly different, otherwise they are approximately identical
      data = {comparison_values[0]: group1, comparison_values[1]: group2}
      df = pd.DataFrame(data)
      shapiro_stat1, shapiro_p1 = stats.shapiro(df[comparison_values[0]])
      shapiro_stat2, shapiro_p2 = stats.shapiro(df[comparison_values[1]])
      statistics_df.at[(getVariableName(dependent_variable), "Shapiro-Wilks test 1"), (getVariableName(independent_variable), value)] = "(statistic=" + str(round(shapiro_stat1, 4)) + ", pvalue=" + str('{:.4g}'.format(shapiro_p1)) + ")"
      statistics_df.at[(getVariableName(dependent_variable), "Shapiro-Wilks test 2"), (getVariableName(independent_variable), value)] = "(statistic=" + str(round(shapiro_stat2, 4)) + ", pvalue=" + str('{:.4g}'.format(shapiro_p2)) + ")"

      for p_value, label in [(shapiro_p1, "Shapiro-Wilks test 1"), (shapiro_p2, "Shapiro-Wilks test 2")]:
        if p_value <= 0.05:
          analysed_statistics_df.at[(getVariableName(dependent_variable), label), (getVariableName(independent_variable), value)] = 'Not normally distributed'
        else:
          analysed_statistics_df.at[(getVariableName(dependent_variable), label), (getVariableName(independent_variable), value)] = 'Normally distributed'

      # for group, title in [(group1, labelConversion(comparison_variable, comparison_values[0]) + "" + independent_variable + "" + str(value) + "" + dependent_variable), (group2, labelConversion(comparison_variable, comparison_values[1]) + "" + independent_variable + "" + str(value) + "" + dependent_variable)]:
      #   plt.hist(group)
      #   plt.title(title)
      #   # plt.show()
      #   plt.savefig("results/statistical_analysis/rerun/histograms/" + comparison_variable + "/" + title + '.png', pad_inches=0.1, dpi=300)
      #   plt.close()

      # print('test2')
      # paired_t_test = stats.ttest_rel(group1, group2)
      # print(paired_t_test)
      # pg_paired_t_test = pt.ttest(group1, group2, paired=True)
      # # print(pg_paired_t_test.to_string() + "\n")
      #
      # # statistics_df.at[(getVariableName(dependent_variable), "Paired Two-Sample T-test"), (getVariableName(independent_variable), value)] = "(statistic=" + str(round(paired_t_test[0], 4)) + ", pvalue=" + str('{:.4g}'.format(paired_t_test[1])) + ")"
      # statistics_df.at[(getVariableName(dependent_variable), "Paired Two-Sample T-test"), (getVariableName(independent_variable), value)] = "t(" + str(pg_paired_t_test['dof'][0]) + ") = " + str(round(pg_paired_t_test['T'][0], 5)) + ", p = " + str('{:.4g}'.format(pg_paired_t_test['p-val'][0]))
      # # if paired_t_test[1] <= 0.05:
      # if pg_paired_t_test['p-val'][0] <= 0.05:
      #   analysed_statistics_df.at[(getVariableName(dependent_variable), "Paired Two-Sample T-test"), (getVariableName(independent_variable), value)] = 'Significant difference'
      # else:
      #   analysed_statistics_df.at[(getVariableName(dependent_variable), "Paired Two-Sample T-test"), (getVariableName(independent_variable), value)] = 'No difference'

      # print('test3')
      # wilcoxon_test = stats.wilcoxon(group1, group2, zero_method="zsplit")
      # print(wilcoxon_test)
      pg_wilcoxon_test = pt.wilcoxon(group1, group2, alternative="two-sided", zero_method="zsplit")\
      # print(pg_wilcoxon_test['W-val'][0], pg_wilcoxon_test['p-val'][0])
      # print('test3')
      # statistics_df.at[(getVariableName(dependent_variable), "Wilcoxon Signed-Rank Test"), (getVariableName(independent_variable), value)] = "(statistic=" + str(round(wilcoxon_test[0], 4)) + ", pvalue=" + str('{:.4g}'.format(wilcoxon_test[1])) + ")"
      statistics_df.at[(getVariableName(dependent_variable), "Wilcoxon Signed-Rank Test"), (getVariableName(independent_variable), value)] = "(statistic=" + str(round(pg_wilcoxon_test['W-val'][0], 4)) + ", pvalue=" + str('{:.4g}'.format(pg_wilcoxon_test['p-val'][0])) + ")"
      # if wilcoxon_test[1] <= 0.05:
      if pg_wilcoxon_test['p-val'][0] <= 0.05:
        analysed_statistics_df.at[(getVariableName(dependent_variable), "Wilcoxon Signed-Rank Test"), (getVariableName(independent_variable), value)] = 'Significant difference'
      else:
        analysed_statistics_df.at[(getVariableName(dependent_variable), "Wilcoxon Signed-Rank Test"), (getVariableName(independent_variable), value)] = 'No difference'


def prepareDataSinglePairedTtest(df, comparison_variable, comparison_values, dependent_variable):
  if 'AgentID' in df:
    results1 = df.loc[df['AgentID'] == 0]
    results = results1.drop(['AgentID', 'state'], axis=1)
  else:
    results = df

  grouped_results = results.groupby([comparison_variable])[dependent_variable].apply(list)
  data_0 = grouped_results[comparison_values[0]]
  data_1 = grouped_results[comparison_values[1]]

  # TODO: commented out on 10/01/2024
  # for value, group in [(comparison_values[0], data_0), (comparison_values[1], data_1)]:
    # print(labelConversion(comparison_variable, value))
    # print(pd.DataFrame(group).describe())

  return [data_0, data_1]


def single_paired_t_test(file_paths, results, comparison_variable):
  comparison_values = getComparisonValue(comparison_variable)

  file_path_normal, file_path_latex = file_paths

  file_normal = open(file_path_normal, 'a')
  file_latex = open(file_path_latex, 'a')

  for file in [file_normal, file_latex]:
    file.write("\n\nStatistical analysis on " + comparison_variable + "\n")
    file.write("---------------------------------")

  for dependent_variable in ['engagement_ratio', 'diffusion_rate']:
    group1_original, group2_original = prepareDataSinglePairedTtest(results, comparison_variable, comparison_values, dependent_variable)

    if dependent_variable == 'engagement_ratio':
      group1 = [i * 100 for i in group1_original]
      group2 = [i * 100 for i in group2_original]
    else:
      group1 = group1_original
      group2 = group2_original

    levene_test = stats.levene(group1, group2, center='median')

    shapiro1 = stats.shapiro(group1)
    shapiro_stat1, shapiro_p1 = shapiro1
    # TODO: commented out on 10/01/2024
    # print(shapiro1)
    shapiro_stat2, shapiro_p2 = stats.shapiro(group2)

    # TODO: commented out on 10/01/2024
    # for group, title in [(group1, labelConversion(comparison_variable, comparison_values[0]) + "" + dependent_variable), (group2, labelConversion(comparison_variable, comparison_values[1]) + "" + dependent_variable)]:
    #   plt.hist(group)
    #   plt.title(title)
    #   plt.show()

    # paired_t_test = stats.ttest_rel(group1, group2)
    pg_paired_t_test = pt.ttest(group1, group2, paired=True)

    # TODO: commented out on 10/01/2024
    # print(pg_paired_t_test)

    # wilcoxon_test = stats.wilcoxon(group1, group2, zero_method="zsplit")
    pg_wilcoxon_test = pt.wilcoxon(group1, group2, alternative="two-sided", zero_method="zsplit")
    # TODO: commented out on 10/01/2024
    # print(pg_wilcoxon_test)

    for file in [file_normal, file_latex]:
      file.write("\n\n------" + getVariableName(dependent_variable) + "------\n")

      # pvalue <= 0.05 = variances are significantly different, otherwise they are approximately identical
      file.write("Levene's test:\n")
      file.write(str(levene_test))
      if levene_test[1] <= 0.05:
        file.write('\nConclusion: Different variances\n\n')
      else:
        file.write('\nConclusion: Equal variances\n\n')

      file.write("Shapire-Wilks  test:\n")
      file.write("  group 1: ")
      file.write("W(???) = " + str(round(shapiro_stat1, 4)) + ", p =" + str('{:.4g}'.format(shapiro_p1)))
      if shapiro_p1 <= 0.05:
        file.write('\n  Conclusion: Not normally distributed\n')
      else:
        file.write('\n  Conclusion: Normally distributed\n')
      file.write("\n  group 2: ")
      file.write("W(???) = " + str(round(shapiro_stat2, 4)) + ", p =" + str('{:.4g}'.format(shapiro_p2)))
      if shapiro_p2 <= 0.05:
        file.write('\n  Conclusion: Not normally distributed\n')
      else:
        file.write('\n  Conclusion: Normally distributed\n')

      file.write('Paired two-sample t-test:\n')
      file.write("t(" + str(pg_paired_t_test['dof'][0]) + ") = " + str(round(pg_paired_t_test['T'][0], 5)) + ", p = " + str('{:.4g}'.format(pg_paired_t_test['p-val'][0])))
      if pg_paired_t_test['p-val'][0] <= 0.05:
        file.write('\nConclusion: Significant difference\n\n')
      else:
        file.write('\nConclusion: No difference\n\n')

      file.write("Wilcoxon Signed-Rank test:\n")
      file.write("W(???) = " + str(round(pg_wilcoxon_test['W-val'][0], 4)) + ", p =" + str('{:.4g}'.format(pg_wilcoxon_test['p-val'][0])) + ")")
      if pg_wilcoxon_test['p-val'][0] <= 0.05:
        file.write('\nConclusion: Significant difference\n\n')
      else:
        file.write('\nConclusion: No difference\n')


def t_tests(file_paths, comparison_df, comparison_variable):
  comparison_values = getComparisonValue(comparison_variable)

  comparison_group1 = comparison_df[comparison_df[comparison_variable] == comparison_values[0]]
  comparison_group2 = comparison_df[comparison_df[comparison_variable] == comparison_values[1]]

  file_path_normal, file_path_latex = file_paths

  file_normal = open(file_path_normal, 'a')
  file_latex = open(file_path_latex, 'a')

  for file in [file_normal, file_latex]:
    file.write("\n\nStatistical analysis on " + comparison_variable + "\n")
    file.write("---------------------------------")

  for dependent_variable in ['engagement_ratio', 'diffusion_rate']:

    t_test = stats.ttest_ind(comparison_group1[dependent_variable], comparison_group2[dependent_variable])
    Welch_t_test = stats.ttest_ind(comparison_group1[dependent_variable], comparison_group2[dependent_variable],
                                   equal_var=False)

    for file in [file_normal, file_latex]:
      file.write("\n\n------" + getVariableName(dependent_variable) + "------\n")

      file.write('Independent two-sample t-test:\n')
      file.write(str(t_test))

      file.write("\n\nWelch's two-sample t-test:\n")
      file.write(str(Welch_t_test))


# def testEqualVariance(comparison_df, comparison_variable, independent_variable):
#   print('\n--- Test of equal variances for ' + comparison_variable + ' comparison---')
#
#   comparison_values = getComparisonValue(comparison_variable)
#
#   comparison_group1 = comparison_df[comparison_df[comparison_variable] == comparison_values[0]]
#   comparison_group2 = comparison_df[comparison_df[comparison_variable] == comparison_values[1]]
#
#   independent_group0 = comparison_df[comparison_df[independent_variable] == 0.0]
#   independent_group1 = comparison_df[comparison_df[independent_variable] == 0.1]
#   independent_group2 = comparison_df[comparison_df[independent_variable] == 0.2]
#   independent_group3 = comparison_df[comparison_df[independent_variable] == 0.3]
#   independent_group4 = comparison_df[comparison_df[independent_variable] == 0.4]
#   independent_group5 = comparison_df[comparison_df[independent_variable] == 0.5]
#   independent_group6 = comparison_df[comparison_df[independent_variable] == 0.6]
#   independent_group7 = comparison_df[comparison_df[independent_variable] == 0.7]
#   independent_group8 = comparison_df[comparison_df[independent_variable] == 0.8]
#   independent_group9 = comparison_df[comparison_df[independent_variable] == 0.9]
#   independent_group10 = comparison_df[comparison_df[independent_variable] == 1.0]
#
#   print(comparison_df.sigma.unique())
#
#   group10 = comparison_group1[comparison_group1[independent_variable] == 0.0]
#   group11 = comparison_group1[comparison_group1[independent_variable] == 0.1]
#   group12 = comparison_group1[comparison_group1[independent_variable] == 0.2]
#   group13 = comparison_group1[comparison_group1[independent_variable] == 0.3]
#   group14 = comparison_group1[comparison_group1[independent_variable] == 0.4]
#   group15 = comparison_group1[comparison_group1[independent_variable] == 0.5]
#   group16 = comparison_group1[comparison_group1[independent_variable] == 0.6]
#   group17 = comparison_group1[comparison_group1[independent_variable] == 0.7]
#   group18 = comparison_group1[comparison_group1[independent_variable] == 0.8]
#   group19 = comparison_group1[comparison_group1[independent_variable] == 0.9]
#   group110 = comparison_group1[comparison_group1[independent_variable] == 1.0]
#
#   group20 = comparison_group2[comparison_group2[independent_variable] == 0.0]
#   group21 = comparison_group2[comparison_group2[independent_variable] == 0.1]
#   group22 = comparison_group2[comparison_group2[independent_variable] == 0.2]
#   group23 = comparison_group2[comparison_group2[independent_variable] == 0.3]
#   group24 = comparison_group2[comparison_group2[independent_variable] == 0.4]
#   group25 = comparison_group2[comparison_group2[independent_variable] == 0.5]
#   group26 = comparison_group2[comparison_group2[independent_variable] == 0.6]
#   group27 = comparison_group2[comparison_group2[independent_variable] == 0.7]
#   group28 = comparison_group2[comparison_group2[independent_variable] == 0.8]
#   group29 = comparison_group2[comparison_group2[independent_variable] == 0.9]
#   group210 = comparison_group2[comparison_group2[independent_variable] == 1.0]
#
#   result = 'engagement_ratio'
#
#   for result, name in [('engagement_ratio', 'Equilibrium'), ('Step', 'Diffusion')]:
#     levene_test_equil = stats.levene(comparison_group1[result], comparison_group2[result])
#     print(name)
#     print(levene_test_equil, '\n')
#
#     levene_test_equil_indep = stats.levene(group10[result], group11[result], group12[result],
#                                            group13[result], group14[result],
#                                            group15[result], group16[result],
#                                            group17[result], group18[result],
#                                            group19[result], group110[result], group20[result],
#                                            group21[result], group22[result],
#                                            group23[result], group24[result],
#                                            group25[result], group26[result],
#                                            group27[result], group28[result],
#                                            group29[result], group210[result])
#     print('Equal variances for comparison and independent variables:')
#     print(levene_test_equil_indep, '\n')
#
#     levene_test_equil_group1_indep = stats.levene(group10[result], group11[result], group12[result],
#                                                   group13[result], group14[result],
#                                                   group15[result], group16[result],
#                                                   group17[result], group18[result],
#                                                   group19[result], group110[result])
#     print('Comparison group 1 - equal variances for independent variable:')
#     print(levene_test_equil_group1_indep, '\n')
#
#     levene_test_equil_group2_indep = stats.levene(group20[result], group21[result], group22[result],
#                                                   group23[result], group24[result],
#                                                   group25[result], group26[result],
#                                                   group27[result], group28[result],
#                                                   group29[result], group210[result])
#     print('Comparison group 2 - equal variances for independent variable:')
#     print(levene_test_equil_group2_indep, '\n')
#
#   # print(comparison_variable, ": ", stats.friedmanchisquare(comparison_group1, comparison_group2))
#
#   # print(independent_variable, ": ", stats.friedmanchisquare(independent_group0[result], independent_group1[result],
#   #                                                     independent_group2[result], independent_group3[result],
#   #                                                     independent_group4[result], independent_group5[result],
#   #                                                     independent_group6[result], independent_group7[result],
#   #                                                     independent_group8[result], independent_group9[result],
#   #                                                     independent_group10[result]))
#   #
#   # print(comparison_variable, " + ", independent_variable, ": ",
#   #       stats.friedmanchisquare(group10[result], group11[result], group12[result], group13[result], group14[result],
#   #                               group15[result], group16[result], group17[result], group18[result], group19[result],
#   #                               group110[result], group20[result], group21[result], group22[result], group23[result],
#   #                               group24[result], group25[result], group26[result], group27[result], group28[result],
#   #                               group29[result], group210[result]))


# def anova_tests(path, independent_variable, comparison_variable):
#   results = pd.read_csv(path)
#   comparison_df = results.groupby(by=["RunId"]).max()[
#     [comparison_variable, independent_variable, 'Step', 'engagement_ratio']]
#   print(comparison_df)
#   # comparison_df_equil = results.groupby(by=["RunId"]).max()[[comparison_variable, independent_variable, 'engagement_ratio']]
#   # comparison_df_diff = results.groupby(by=["RunId"]).max()[[comparison_variable, independent_variable, 'Step']]
#   #
#   # Test equal variances
#   testEqualVariance(comparison_df, comparison_variable, independent_variable)
#   #
#   # print('\n---' + comparison_variable + ' comparison for in-degree---')
#   #
#   # # Equilibrium
#   # model_equil = ols('engagement_ratio ~ C(' + comparison_variable + ') + C(' + independent_variable + ') + C(' +
#   #                   comparison_variable + '):C(' + independent_variable + ')', data=comparison_df).fit()
#   # anova_equil = sm.stats.anova_lm(model_equil, typ=2)
#   # print("Two-sample anova for equilibrium:")
#   # print(anova_equil, '\n')
#   #
#   # # stat_equil, p_val_equil = kruskal(comparison_df_equil, nan_policy='omit')
#   # # print(f"Kruskal-Wallis test statistic: {stat_equil:.4f}, p-value: {p_val_equil:.4f}")
#   #
#   # # Diffusion
#   # model_diff = ols('Step ~ C(' + comparison_variable + ') + C(' + independent_variable + ') + C(' +
#   #                  comparison_variable + '):C(' + independent_variable + ')', data=comparison_df).fit()
#   # anova_diff = sm.stats.anova_lm(model_diff, typ=2)
#   # print("Two-sample anova for diffusion:")
#   # print(anova_diff, '\n')
#   #
#   # # stat_diff, p_val_diff = kruskal(comparison_df_diff, nan_policy='omit')
#   # # print(f"Kruskal-Wallis test statistic: {stat_diff:.4f}, p-value: {p_val_diff:.4f}")


def performStatisticalAnalysis(file_paths, data, comparison_variable, independent_variable, independent_values, file_identifier=''):
  file_path_normal, file_path_latex = file_paths

  file_normal = open(file_path_normal, 'a')
  file_latex = open(file_path_latex, 'a')

  for file in [file_normal, file_latex]:
    file.write("\n\n\nStatistical analysis for " + independent_variable + file_identifier + " on " + comparison_variable + "\n")
    file.write("----------------------------------------------------\n")

  statistics_df = createDataFrame(independent_variable, independent_values)
  analysed_statistics_df = createDataFrame(independent_variable, independent_values)

  # Paired t-tests
  for (dependent_variable) in ['engagement_ratio', 'diffusion_rate']:
    multiple_paired_t_tests(statistics_df, analysed_statistics_df, data, comparison_variable, independent_variable, dependent_variable)

  df_min_max = statistics_df.iloc[:, 0:1].copy()
  df_min_max[df_min_max.columns[0][0], 'min'] = statistics_df.min(axis=1)
  df_min_max[df_min_max.columns[0][0], 'max'] = statistics_df.max(axis=1)

  file_normal.write(statistics_df.to_string())
  file_normal.write("\n\n")
  file_normal.write(analysed_statistics_df.to_string())
  file_normal.write("\n\n")
  file_normal.write(df_min_max.to_string())

  file_latex.write(statistics_df.transpose().style.to_latex())
  file_latex.write("\n\n")
  file_latex.write(analysed_statistics_df.transpose().style.to_latex())

  file_normal.close()
  file_latex.close()


def statistical_tests(run_type):
  if run_type == RunType.Granovetter:
    return

  else:
    if run_type == RunType.KnowledgeComparison:
      path_data = 'results/raw_data/knowledge_comparison/'
      comparison_variable = 'knowledge'
    else:  # run_type == RunType.NetworkComparison
      path_data = 'results/raw_data/network_comparison/'
      comparison_variable = 'networkType'

    general_file = comparison_variable + '_comparison_normal.csv'

    file_path_normal = "results/statistical_analysis/rerun/" + getRuntypeName(run_type=run_type, return_type='folder') + "/statistical_analysis.txt"
    file_path_latex = "results/statistical_analysis/rerun/" + getRuntypeName(run_type=run_type, return_type='folder') + "/statistical_analysis_latex.txt"

    for file_path in [file_path_normal, file_path_latex]:
      if os.path.exists(file_path):
        os.remove(file_path)

      file = open(file_path, 'x')
      file.write("Statistical analysis for " + getRuntypeName(run_type=run_type, return_type='name'))
      file.write("\n----------------------------------------------------\n")
      file.write("----------------------------------------------------")
      file.close()

    general_results = pd.read_csv(path_data + general_file)
    single_paired_t_test(file_paths=[file_path_normal, file_path_latex], results=general_results, comparison_variable=comparison_variable)

    for file, independent_variable, values, file_identifier in [('out-degree_comparison.csv', 'out_degree', out_degrees, ''),
                                                                ('n_comparison.csv', 'num_of_nodes', nums, ''),
                                                                ('sigma10_comparison.csv', 'sigma', sigmas1, '10'),
                                                                ('sigma100_comparison.csv', 'sigma', sigmas2, '100')]:
      results = pd.read_csv(path_data + file)

      performStatisticalAnalysis(file_paths=(file_path_normal, file_path_latex), data=results,
                                 comparison_variable=comparison_variable, independent_variable=independent_variable,
                                 independent_values=values, file_identifier=file_identifier)


  # # T-tests
  # t_tests(path_knowledge + 'knowledge_comparison.csv', 'knowledge',
  #         KnowledgeType.Network.value, KnowledgeType.Neighbourhood.value)
  #
  # t_tests(path_network + 'network_comparison.csv', 'networkType',
  #         NetworkType.Directed.value, NetworkType.Undirected.value)

  # # ANOVA test
  # anova_tests(path_knowledge + 'sigma_comparison.csv', 'sigma', 'knowledge')

  # anova_tests(path_network + 'sigma_comparison.csv', 'sigma', 'networkType')
