from __future__ import division
import axelrod

import axelrod_utils

strategies = [s() for s in axl.strategies]

def ranks_for(my_strategy_factory, iterations=200, max_size=50,
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
        axl.seed(parameter)
        G = nx.binomial_graph(len(strategies), p)
        edges = G.edges()
        tournament = axl.SpatialTournament(sample_strategies, edges=edges)
        results = tournament.play(processes=0)

        ranks.append(results.ranking[-1])
    eval_rank = eval_function(ranks)
    return eval_rank


def do_table(table):
    """
    Take a lookup table dict, construct a lambda factory for it, and return
    a tuple of the score and the table itself. The score this time is based
    on ranks
    """
    fac = lambda: axelrod.LookerUp(lookup_table=table)
    return(ranks_for(fac), table)


def score_tables(tables, pool):
    """Use a multiprocessing Pool to take a bunch of tables and score them,
       based on ranks
    """
    return sorted(pool.map(do_table, tables), reverse=True)
