import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

from model import NetworkModel


def plotDirectedGraph(mdl):
  G = mdl.G
  pos = nx.spring_layout(G, seed=mdl.seed)

  node_sizes = [3 + 10 * i for i in range(len(G))]
  M = G.number_of_edges()
  edge_colors = range(2, M + 2)
  edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
  cmap = plt.cm.plasma

  nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
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

  pc = mpl.collections.PatchCollection(edges, cmap=cmap)
  pc.set_array(edge_colors)

  ax = plt.gca()
  ax.set_axis_off()
  plt.colorbar(pc, ax=ax)
  plt.show()


def main():
  model = NetworkModel()
  plotDirectedGraph(model)

  model.run_model()

  state_agents = model.datacollector.get_model_vars_dataframe()
  # print(state_agents)
  state_agents.plot()
  plt.show()


if __name__ == '__main__':
  main()
