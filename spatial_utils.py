from __future__ import division
import axelrod
import networkx as nx
import numpy
import random
from functools import partial

from axelrod_utils import *

strategies = [s() for s in axelrod.strategies]

# SCORING FUNCTION SPATIAL

def ranks_for(my_strategy_factory, max_size, min_size, topology, eval_function,
              turns = 200, repetitions=16, max_graph_attempt=500, ub_parameter=10,
                                parameter_repetitions=3, strategies=strategies):
    """
    Given a function that will return a strategy, calculate the evaluation
    function for the strategy after participating in various spatial tournament.
    The evaluation function could be either the median, min or max rank. The
    spatial topologies can be that of a binomial graph, a small world graph
    and a complete graph.
    """

    sample_size = random.randint(min_size, max_size)
    sample_strategies = random.sample(strategies, sample_size)
    sample_strategies.append(my_strategy_factory())
    ranks =[]
    for parameter in range(1, ub_parameter + 1):

        p = parameter/ub_parameter
        for _ in range(parameter_repetitions):
            ranking = False
            graph_seed = 0
            graph_attempt_count = 0
            while (ranking is False):  # repeating until having connected graph
                G = define_topology(topology, sample_strategies, p, graph_seed)

                # check that all nodes are connected
                connections = [len(c) for c in sorted(nx.connected_components(G),
                                                         key=len, reverse=True)]

                if connections and (1 not in connections) :
                    edges = G.edges()
                    ranking = rank_strategy(sample_strategies, edges, turns,
                                                                    repetitions)
                else:
                    graph_seed += 1
                    graph_attempt_count += 1
                    if graph_attempt_count > max_graph_attempt:
                        raise ValueError("No valid graphs were found after {} attempts".format(graph_attempt_count))

            ranks.append(ranking)

    eval_rank = eval_function(ranks)/sample_size #between 0 and 1
    return eval_rank


def rank_strategy(sample_strategies, edges, turns, repetitions):
    """
    A function that performs a spatial tournament and returns the LookerUp's
    rank in that tournament
    """
    tournament = axelrod.SpatialTournament(sample_strategies, edges=edges,
                                           turns=turns, repetitions=repetitions)
    results = tournament.play()
    return(results.ranking[-1])


def define_topology(topology, sample_strategies, p, graph_seed):
    """
    A function that returns a graph, based on the topology argument. The graph
    can be a binomial, a watts strogatz graph, a complete or any of the
    aforementioned.
    """
    # set seeds for graphs
    axelrod.seed(graph_seed)

    if topology == 'binomial':
        G = nx.binomial_graph(len(sample_strategies), p)
    if topology == 'small' :
        initial_neighborhood_size = random.randint(1, len(sample_strategies) -1)
        G = nx.watts_strogatz_graph(len(sample_strategies), initial_neighborhood_size, p)
    if topology == 'complete' :
        G = nx.complete_graph(len(sample_strategies))
    if topology == 'random' :
        random_topology_seed = 1
        random.seed(random_topology_seed)

        random_topology = random.randint(0,2)
        if random_topology == 0 :
            G = nx.complete_graph(len(sample_strategies))

        if random_topology == 1 :
            G = nx.binomial_graph(len(sample_strategies), p)

        else :
            initial_neighborhood_size = random.randint(1, len(sample_strategies) -1)
            random.seed(initial_neighborhood_size)
            G = nx.watts_strogatz_graph(len(sample_strategies),
                                                          initial_neighborhood_size, p)
    return(G)


def do_table(table, parameters):
    """
    Take a lookup table dict, construct a lambda factory for it, and return
    a tuple of the score and the table itself. The score this time is based
    on ranks
    """
    max_size = parameters[0]
    min_size = parameters[1]
    eval_function = parameters[2]
    topology = parameters[3]
    fac = lambda: axelrod.LookerUp(lookup_table=table)
    return(ranks_for(my_strategy_factory=fac, max_size=max_size, min_size=min_size,
                     eval_function=eval_function,topology=topology), table)

def score_tables(tables, pool, parameters):
    """Use a multiprocessing Pool to take a bunch of tables and score them,
       based on ranks
    """
    func = partial(do_table, parameters=parameters)
    return sorted(pool.map(func, tables), key=lambda x: x[0], reverse=True)
