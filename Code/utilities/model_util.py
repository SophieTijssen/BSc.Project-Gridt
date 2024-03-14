from enum import Enum
import numpy as np


sigmas1 = np.delete(np.linspace(0.0, 1.0, 11).round(decimals=2), 0)
sigmas2_tmp = np.linspace(0.0, 1.0, 101).round(decimals=2)
sigmas2 = sigmas2_tmp[sigmas2_tmp >= 0.1]
nums = range(10, 205, 5)
out_degrees = range(1, 11, 1)
beta_parameters = [(0.5, 0.5), (1, 2), (2, 1), (2, 2), (2, 8), (8, 2), (8, 8)]


class RunType(Enum):
  """
  Enumerate class for the possible runs types.
  """
  Granovetter = 0
  KnowledgeComparison = 1
  NetworkComparison = 2
  SigmaComparison = 3
  Single = 4
  Batch = 5


class State(Enum):
  """
  Enumerate class for the different states agents can have.
  """
  DEFECT = 0
  COOPERATE = 1


def getRuntypeName(run_type, return_type):
  if run_type == RunType.Granovetter:
    if return_type == 'name':
      return run_type.name
    if return_type == 'folder':
      return 'granovetter'
  elif run_type == RunType.KnowledgeComparison:
    if return_type == 'name':
      return 'Knowledge comparison'
    if return_type == 'parameter':
      return 'knowledge'
    if return_type == 'folder':
      return 'knowledge_comparison'
  elif run_type == RunType.NetworkComparison:
    if return_type == 'name':
      return 'Network comparison'
    if return_type == 'parameter':
      return 'networkType'
    if return_type == 'folder':
      return 'network_comparison'


def getVariableName(variable):
  if variable == 'knowledge':
    return 'Knowledge type'
  elif variable == 'networkType':
    return 'Network type'
  elif variable == 'sigma':
    return 'Standard deviation of threshold distribution'
  elif variable == 'num_of_nodes':
    return 'Number of agents'
  elif variable == 'out_degree':
    return 'Out-degree'
  elif variable == 'engagement_ratio':
    return 'Behaviour spread'
  elif variable == 'diffusion_rate':
    return 'Diffusion speed'
  # TODO: Remove below statement
  # elif variable == 'Step':
  #   return 'Diffusion speed'


def number_state(model, state):
  """
  Count the total number of agents in a specific state in the network.
  :param model: The mesa model.
  :param state: The specific state.
  :return The total number of agents in the specific state.
  """

  return sum(1 for a in model.grid.get_all_cell_contents() if a.state is state)


def number_cooperating(model):
  """
  Count the number of cooperating agents using the number_state() function.
  :param model: The mesa model.
  :return The total number of cooperating agents.
  """

  return number_state(model, State.COOPERATE)


def number_defecting(model):
  """
  Count the number of defecting agents using the number_state() function.
  :param model: The mesa model.
  :return The total number of defecting agents.
  """

  return number_state(model, State.DEFECT)


def calculate_engagement_ratio(model):
  """
  Calculate the engagement ratio (the ratio of cooperating agents over all agents).
  :param model: The mesa model.
  :return The engagement ratio.
  """

  cooperating = number_cooperating(model)
  total_agents = len(model.grid.get_all_cell_contents())
  cooperating_rate = cooperating/total_agents
  return cooperating_rate


def calculate_diffusion_rate(model):
  """
  Calculate the diffusion rate (how fast a new behaviour spreads through a network).
  Unit = agents/step or nodes/s
  :param model: The mesa model.
  :return The engagement ratio.
  """

  cooperating = number_cooperating(model)
  steps = model.schedule.steps
  if steps != 0:
    diffusion_rate = cooperating/steps
  else:
    if steps == 0 and cooperating != 0:
      print('huhh')
      print(cooperating)
    diffusion_rate = 0
  return diffusion_rate
