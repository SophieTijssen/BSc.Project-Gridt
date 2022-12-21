from enum import Enum

from mesa.batchrunner import batch_run
from mesa_model.model import GranovetterModel
from results.plot_graphs import *


class RunType(Enum):
  """
  Enumerate class for the possible runs types.
  """
  Granovetter = 0
  NetworkComparison = 1
  Single = 2
  Batch = 3


def singleRun(n, network, neighbourhood, distribution, mu, sigma, in_degree):
  """
  Run the model for a single iteration.

  :param n: The number of agents in the network.
  :param network: The type of network used for the model (directed/undirected).
  :param neighbourhood: Boolean that shows whether an agent can see the whole network or only its neighbourhood.
  :param distribution: The distribution used for sampling the agent thresholds.
  :param mu: The mean of the threshold distribution (in case of a normal distribution).
  :param sigma: The standard deviation of the threshold distribution (in case of a normal distribution).
  :param in_degree: The in-degree of all agents.
  """

  model = GranovetterModel(num_of_nodes=n, neighbourhood=neighbourhood, network=network.value,
                           distribution=distribution.value, mu=mu, sigma=sigma, in_degree=in_degree)

  while model.running and model.schedule.steps < 100:
    model.step()

  model.datacollector.collect(model)
  model_out = model.datacollector.get_model_vars_dataframe()

  plotDirectedGraph(model)

  showDegreeHistogram(model.G)

  plotEngagementProgression(model_out)


def batchRun(n, i, network, neighbourhood, distribution, mu, sigmas, in_degree):
  """
  Run the model for multiple iterations (using BatchRun).

  :param n: The number of agents in the network.
  :param i: The number of iterations of the batch run.
  :param network: The type of network used for the model (directed/undirected).
  :param neighbourhood: Boolean that shows whether an agent can see the whole network or only its neighbourhood.
  :param distribution: The distribution used for sampling the agent thresholds.
  :param mu: The mean of the threshold distribution (in case of a normal distribution).
  :param sigmas: A list containing the standard deviations of the threshold distributions
                 for each iteration (in case of a normal distribution).
  :param in_degree: The in-degree of all agents.
  :return: A pandas dataframe containing the results/data from the DataCollector.
  """
  params = {
    "num_of_nodes": n,
    "network": network.value,
    "neighbourhood": neighbourhood,
    "distribution": distribution.value,
    "mu": mu,
    "sigma": sigmas,
    "in_degree": in_degree
  }

  results = batch_run(
    GranovetterModel,
    parameters=params,
    iterations=i,
    max_steps=100,
    number_processes=None,
    data_collection_period=-1,
    display_progress=True
  )

  results_df = pd.DataFrame(results).drop('distribution', axis=1)

  return results_df
