from mesa.batchrunner import batch_run
from mesa_model.granovetter_model import GranovetterModel
from mesa_model.neighbourhood_model import NeighbourhoodModel
from results.plot_graphs import showDegreeHistogram, singleRunPlot, plotDirectedGraph
from utilities.model_util import RunType
from utilities.network_util import NetworkData

import pandas as pd


def singleRun(run, n, network, knowledge, distribution, mu, sigma, a, b, out_degree, networkData, path, filename):
  """
  Run the model for a single iteration.

  :param run: The type of run which specifies which model/agent to use.
  :param n: The number of agents in the network.
  :param network: The type of network used for the model (directed/undirected).
  :param knowledge: Boolean that shows whether an agent can see the whole network or only its neighbourhood.
  :param distribution: The distribution used for sampling the agent thresholds.
  :param mu: The mean of the threshold distribution (in case of a normal distribution).
  :param sigma: The standard deviation of the threshold distribution (in case of a normal distribution).
  :param a:
  :param b:
  :param out_degree: The out-degree of all agents.
  :param networkData: An object to store the network that should be used (can be empty).
  :param path:
  :param filename:
  """

  if run == RunType.Granovetter.value:
    model = GranovetterModel(num_of_nodes=n, networkType=network, distributionType=distribution,
                             mu=mu, sigma=sigma, out_degree=out_degree)
  else:
    model = NeighbourhoodModel(run=run, num_of_nodes=n, knowledge=knowledge, networkType=network,
                               distributionType=distribution, mu=mu, sigma=sigma, beta_parameters=(a, b),
                               out_degree=out_degree, networkData=networkData)

  while model.running and model.schedule.steps < 100:
    model.step()

  model.datacollector.collect(model)
  model_out = model.datacollector.get_model_vars_dataframe()

  # plotDirectedGraph(model)

  showDegreeHistogram(path, model.G, filename)

  singleRunPlot(path, model_out, filename)


def batchRunGranovetter(n, i, network, distributions, mu, sigma, out_degree):
  """
  Run the model for multiple iterations (using BatchRun).

  :param n: The number of agents in the network.
  :param i: The number of iterations of the batch run.
  :param network: The type of network used for the model (directed/undirected).
  :param distributions: The distribution used for sampling the agent thresholds.
  :param mu: The mean of the threshold distribution (in case of a normal distribution).
  :param sigma: A list containing the standard deviations of the threshold distributions
                 for each iteration (in case of a normal distribution).
  :param out_degree: The out-degree of all agents.
  :return: A pandas dataframe containing the results/data from the DataCollector.
  """

  params = {
    "num_of_nodes": n,
    "networkType": network,
    "distributionType": distributions,
    "mu": mu,
    "sigma": sigma,
    "out_degree": out_degree,
  }

  results = batch_run(
    GranovetterModel,
    parameters=params,
    iterations=i,
    max_steps=100,
    number_processes=1,
    data_collection_period=-1,
    display_progress=True
  )

  results_df = pd.DataFrame(results)

  return results_df


def batchRunNeighbourhood(run, n, i, network, knowledge, distribution, mu, sigma, beta_parameters, out_degree, collectionPeriod=1):
  """
  Run the model for multiple iterations (using BatchRun).

  :param run:
  :param n: The number of agents in the network.
  :param i: The number of iterations of the batch run.
  :param network: The type of network used for the model (directed/undirected).
  :param knowledge: Boolean that shows whether an agent can see the whole network or only its neighbourhood.
  :param distribution: The distribution used for sampling the agent thresholds.
  :param mu: The mean of the threshold distribution (in case of a normal distribution).
  :param sigma: A list containing the standard deviations of the threshold distributions
                 for each iteration (in case of a normal distribution).
  :param beta_parameters:
  :param out_degree: The out-degree of all agents.
  :param collectionPeriod:
  :return: A pandas dataframe containing the results/data from the DataCollector.
  """

  params = {
    "run": run,
    "num_of_nodes": n,
    "out_degree": out_degree,
    "mu": mu,
    "sigma": sigma,
    "beta_parameters": beta_parameters,
    "networkType": network,
    "knowledge": knowledge,
    "distributionType": distribution,
    "networkData": NetworkData()
  }

  results = batch_run(
    NeighbourhoodModel,
    parameters=params,
    iterations=i,
    max_steps=100,
    number_processes=1,
    data_collection_period=collectionPeriod,
    display_progress=True
  )

  results_df = pd.DataFrame(results).drop('networkData', axis=1)

  return results_df
