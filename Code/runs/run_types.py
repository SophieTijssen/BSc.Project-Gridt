from mesa.batchrunner import batch_run
from mesa_model.granovetter_model import GranovetterModel
from mesa_model.neighbourhood_model import NeighbourhoodModel
from results.plot_graphs import *
from utilities.model_util import RunType
from utilities.network_util import NetworkData


def singleRun(run, n, network, neighbourhood, distribution, mu, sigma, in_degree, networkData, titleSpecification):
  """
  Run the model for a single iteration.

  :param run: The type of run which specifies which model/agent to use.
  :param n: The number of agents in the network.
  :param network: The type of network used for the model (directed/undirected).
  :param neighbourhood: Boolean that shows whether an agent can see the whole network or only its neighbourhood.
  :param distribution: The distribution used for sampling the agent thresholds.
  :param mu: The mean of the threshold distribution (in case of a normal distribution).
  :param sigma: The standard deviation of the threshold distribution (in case of a normal distribution).
  :param in_degree: The in-degree of all agents.
  :param networkData: An object to store the network that should be used (can be empty).
  :param titleSpecification: Specification of the title to add at the end of the standard title.
  """

  if run == RunType.Granovetter:
    model = GranovetterModel(num_of_nodes=n, networkType=network.value, distributionType=distribution.value,
                             mu=mu, sigma=sigma, in_degree=in_degree)
  else:
    model = NeighbourhoodModel(run=run, num_of_nodes=n, neighbourhood=neighbourhood, networkType=network.value,
                               distributionType=distribution.value, mu=mu, sigma=sigma, in_degree=in_degree,
                               networkData=networkData)

  while model.running and model.schedule.steps < 100:
    model.step()

  model.datacollector.collect(model)
  model_out = model.datacollector.get_model_vars_dataframe()

  plotDirectedGraph(model)

  showDegreeHistogram(model.G)

  singleRunPlot(model_out, titleSpecification)


def batchRunGranovetter(n, i, network, distributions, mu, sigmas, in_degree):
  """
  Run the model for multiple iterations (using BatchRun).

  :param n: The number of agents in the network.
  :param i: The number of iterations of the batch run.
  :param network: The type of network used for the model (directed/undirected).
  :param distributions: The distribution used for sampling the agent thresholds.
  :param mu: The mean of the threshold distribution (in case of a normal distribution).
  :param sigmas: A list containing the standard deviations of the threshold distributions
                 for each iteration (in case of a normal distribution).
  :param in_degree: The in-degree of all agents.
  :return: A pandas dataframe containing the results/data from the DataCollector.
  """

  params = {
    "num_of_nodes": n,
    "networkType": network.value,
    "distributionType": distributions.value,
    "mu": mu,
    "sigma": sigmas,
    "in_degree": in_degree,
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

  results_df = pd.DataFrame(results)  # .drop('networkData', axis=1)

  return results_df


def batchRunNeighbourhood(run, n, i, networks, neighbourhoods, distributions, mu, sigmas, in_degree):
  """
  Run the model for multiple iterations (using BatchRun).

  :param run:
  :param n: The number of agents in the network.
  :param i: The number of iterations of the batch run.
  :param networks: The type of network used for the model (directed/undirected).
  :param neighbourhoods: Boolean that shows whether an agent can see the whole network or only its neighbourhood.
  :param distributions: The distribution used for sampling the agent thresholds.
  :param mu: The mean of the threshold distribution (in case of a normal distribution).
  :param sigmas: A list containing the standard deviations of the threshold distributions
                 for each iteration (in case of a normal distribution).
  :param in_degree: The in-degree of all agents.
  :return: A pandas dataframe containing the results/data from the DataCollector.
  """

  if type(networks) == list:
    network_values = [ntype.value for ntype in networks]
  else:
    network_values = networks.value

  params = {
    "run": run.value,
    "num_of_nodes": n,
    "networkType": network_values,
    "neighbourhood": neighbourhoods,
    "distributionType": distributions.value,
    "mu": mu,
    "sigma": sigmas,
    "in_degree": in_degree,
    "networkData": NetworkData()
  }

  results = batch_run(
    NeighbourhoodModel,
    parameters=params,
    iterations=i,
    max_steps=100,
    number_processes=1,
    data_collection_period=1,
    display_progress=True
  )

  results_df = pd.DataFrame(results).drop('networkData', axis=1)

  return results_df
