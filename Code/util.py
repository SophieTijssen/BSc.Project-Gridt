from enum import Enum


class State(Enum):
  DEFECT = 0
  COOPERATE = 1


def number_state(model, state):
  return sum(1 for a in model.grid.get_all_cell_contents() if a.state is state)


def number_cooperating(model):
  return number_state(model, State.COOPERATE)


def number_defecting(model):
  return number_state(model, State.DEFECT)
