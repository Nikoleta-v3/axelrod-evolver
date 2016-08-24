from __future__ import division
import axelrod
import networkx as nx
import numpy
import random

from axelrod_utils import *

strategies = [s() for s in axelrod.strategies]

# FUNCTIONS FOR SMALL TOPOLOGY

# For evaluation function is median rank
def ranks_for_median_small(my_strategy_factory, iterations=200, max_size=50,
              strategies=strategies, eval_function=numpy.median):
    """
    Given a function that will return a strategy, calculate the evaluation
    function for the strategy after participating in various spatial tournament.
    The evaluation function could be either the median, min or max rank. The
    spatial topologies can be that of a binomial graph, a small world graph
    and a complete graph.
    """

    sample_size = random.randint(1,max_size)

    sample_strategies = random.sample(strategies, sample_size)
    sample_strategies.append(my_strategy_factory())

    ranks = []
    for parameter in range(10):

        p = parameter/10
        axelrod.seed(parameter)
        initial_neighborhood_size = random.randint(1, len(sample_strategies) -1)
        random.seed(initial_neighborhood_size)
        G = nx.watts_strogatz_graph(len(sample_strategies), initial_neighborhood_size, p)

        # check that all nodes are connected
        connections = [len(c) for c in sorted(nx.connected_components(G), key=len,
                                                                  reverse=True)]

        if connections and (1 not in connections) :
            edges = G.edges()

            tournament = axelrod.SpatialTournament(sample_strategies, edges=edges)
            results = tournament.play()

            ranks.append(results.ranking[-1])
    eval_rank = eval_function(ranks)
    return eval_rank


def do_table_median_small(table):
    """
    Take a lookup table dict, construct a lambda factory for it, and return
    a tuple of the score and the table itself. The score this time is based
    on ranks
    """
    fac = lambda: axelrod.LookerUp(lookup_table=table)
    return(ranks_for_median_small(fac), table)


def score_tables_median_small(tables, pool):
    """Use a multiprocessing Pool to take a bunch of tables and score them,
       based on ranks
    """
    return sorted(pool.map(do_table_median_small, tables),key=lambda x: x[0], reverse=True)

# For evaluation function is min rank
def ranks_for_min_small(my_strategy_factory, iterations=200, max_size=50,
              strategies=strategies, eval_function=numpy.min):
    """
    Given a function that will return a strategy, calculate the evaluation
    function for the strategy after participating in various spatial tournament.
    The evaluation function could be either the median, min or max rank. The
    spatial topologies can be that of a binomial graph, a small world graph
    and a complete graph.
    """

    sample_size = random.randint(1,max_size)

    sample_strategies = random.sample(strategies, sample_size)
    sample_strategies.append(my_strategy_factory())

    ranks = []
    for parameter in range(10):

        p = parameter/10
        axelrod.seed(parameter)
        initial_neighborhood_size = random.randint(1, len(sample_strategies) -1)
        random.seed(initial_neighborhood_size)
        G = nx.watts_strogatz_graph(len(sample_strategies), initial_neighborhood_size, p)

        # check that all nodes are connected
        connections = [len(c) for c in sorted(nx.connected_components(G), key=len,
                                                                  reverse=True)]

        if connections and (1 not in connections) :
            edges = G.edges()

            tournament = axelrod.SpatialTournament(sample_strategies, edges=edges)
            results = tournament.play()

            ranks.append(results.ranking[-1])
    eval_rank = eval_function(ranks)
    return eval_rank

def do_table_min_small(table):
    """
    Take a lookup table dict, construct a lambda factory for it, and return
    a tuple of the score and the table itself. The score this time is based
    on ranks
    """
    fac = lambda: axelrod.LookerUp(lookup_table=table)
    return(ranks_for_min_small(fac), table)


def score_tables_min_small(tables, pool):
    """Use a multiprocessing Pool to take a bunch of tables and score them,
       based on ranks
    """
    return sorted(pool.map(do_table_min_small, tables),key=lambda x: x[0], reverse=True)

# For evaluation function is max rank
def ranks_for_max_small(my_strategy_factory, iterations=200, max_size=50,
              strategies=strategies, eval_function=numpy.max):
    """
    Given a function that will return a strategy, calculate the evaluation
    function for the strategy after participating in various spatial tournament.
    The evaluation function could be either the median, min or max rank. The
    spatial topologies can be that of a binomial graph, a small world graph
    and a complete graph.
    """

    sample_size = random.randint(1,max_size)

    sample_strategies = random.sample(strategies, sample_size)
    sample_strategies.append(my_strategy_factory())

    ranks = []
    for parameter in range(10):

        p = parameter/10
        axelrod.seed(parameter)
        initial_neighborhood_size = random.randint(1, len(sample_strategies) -1)
        random.seed(initial_neighborhood_size)
        G = nx.watts_strogatz_graph(len(sample_strategies), initial_neighborhood_size, p)

        # check that all nodes are connected
        connections = [len(c) for c in sorted(nx.connected_components(G), key=len,
                                                                  reverse=True)]

        if connections and (1 not in connections) :
            edges = G.edges()

            tournament = axelrod.SpatialTournament(sample_strategies, edges=edges)
            results = tournament.play()

            ranks.append(results.ranking[-1])
    eval_rank = eval_function(ranks)
    return eval_rank

def do_table_max_small(table):
    """
    Take a lookup table dict, construct a lambda factory for it, and return
    a tuple of the score and the table itself. The score this time is based
    on ranks
    """
    fac = lambda: axelrod.LookerUp(lookup_table=table)
    return(ranks_for_max_small(fac), table)


def score_tables_max_small(tables, pool):
    """Use a multiprocessing Pool to take a bunch of tables and score them,
       based on ranks
    """
    return sorted(pool.map(do_table_max_small, tables),key=lambda x: x[0], reverse=True)
