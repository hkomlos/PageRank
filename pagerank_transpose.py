import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
# import winsound
import random
from datetime import datetime
import json
import time

# get graph set
def get_graph_set(filename):
    with open(filename, 'r') as f:
        nodes_num = f.readline()
    return {node for node in range(int(nodes_num))}


# get strongly connected component
def get_strongly_connected_component(filename):
    with open(filename, 'r') as f:
        nodes_combined = f.read()
    nodes = nodes_combined.split(', ')
    for node in nodes:
        if '\n' in node:
            strongly_connected_component.add(int(node.strip('\n')))
        else:
            strongly_connected_component.add(int(node))


# Setting the personalization vector
def set_personalization(already_used, without_spammers_flag):
    # setting the reset vector to zeros

    ## when testing SCC only
    # personalization = dict.fromkeys(strongly_connected_component, 0)
    # when testing the whole graph
    personalization = dict.fromkeys(graph_set, 0)

    # if eeded graph without spammers this part delete the spammers slot in reset vector dictionary
    if without_spammers_flag:
        for spammer in spammers:
            del personalization[spammer]

    # this part randomly select trusted site which wasn't selected yet
    while True:
        selected_trusted_site = random.sample(trusted, 1)[0]
        if selected_trusted_site not in already_used:
            break
    # this part sets the selected site reset probability to 1
    personalization.update({selected_trusted_site: 1})
    # this part is meant to prevent the situation in which the same trusted site was selected more then once
    already_used.append(selected_trusted_site)
    return personalization


# this function creates the web graph
def create_graph(filename):
    G = nx.DiGraph()
    f = open(filename, "r")
    line = f.readline()
    hostID_num = int(line)
    for node in range(0, hostID_num):
        if node in trusted:
            color_map.append('green')
        elif node in spammers:
            color_map.append('red')
        else:
            color_map.append('blue')

        ## adding only nodes in the strogly connected component
        # if node in strongly_connected_component:
        #   G.add_node(node)
        # adding nodes to graph
        G.add_node(node)

        # if node in spammers and without_spammers_flag == 1:
        #     G.add_node(node)
        # elif node in trusted and num_of_trusted_sites > 0:
        #     G.add_node(node)
        #     # personalization.update({node: weight_of_trusted_site})
        #     # nstart.update({node: 1})
        #     # num_of_trusted_sites -= 1
        # else:
        #     G.add_node(node)
        #     # personalization.update({node: 1})
        #     # nstart.update({node: 0})

    for node in range(0, hostID_num):
        line = f.readline()
        if line != "\n":
            dests = line.split(" ")
            for dest in dests:
                dest_parts = dest.split(":")
                G.add_edge(node, int(dest_parts[0]), weight=1)
                ## build graph from SCC only
                # if node in strongly_connected_component and int(dest_parts[0]) in strongly_connected_component:
                #   G.add_edge(node, int(dest_parts[0]), weight=1)


                ## 'weighted graph
                # G.add_edge(node, int(dest_parts[0]), weight=int(dest_parts[1]))

                ## mutiple links
                # if for multiDiGraph - takes a lot of time
                # for link in range(0, int(dest_parts[1])):
                #     G.add_edge(node, int(dest_parts[0]))
    return G


# this function classify the sites to trusted and spam sites
def get_assessments(filename):
    f = open(filename, "r")
    line = f.readline()
    while line != '':
        line_parts = line.split(" ")
        ## adding spammers only from SCC
        # if line_parts[1] == "spam" and int(line_parts[0]) in strongly_connected_component:
        # adding spammers
        if line_parts[1] == "spam":
            spammers.add(int(line_parts[0]))
        elif line_parts[1] == "nonspam" and int(line_parts[0]) in strongly_connected_component:
            trusted.add(int(line_parts[0]))
        line = f.readline()


# page rank function
def pagerank(G, alpha=0.99, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
             dangling=None):
    """Return the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
    A NetworkX graph. Undirected graphs will be converted to a directed
    graph with two directed edges for each undirected edge.

    alpha : float, optional
    Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
    The "personalization vector" consisting of a dictionary with a
    key for every graph node and nonzero personalization value for each node.
    By default, a uniform distribution is used.

    max_iter : integer, optional
    Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
    Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
    Starting value of PageRank iteration for each node.

    weight : key, optional
    Edge data key to use as weight. If None weights are set to 1.

    dangling: dict, optional
    The outedges to be assigned to any "dangling" nodes, i.e., nodes without
    any outedges. The dict key is the node the outedge points to and the dict
    value is the weight of that outedge. By default, dangling nodes are given
    outedges according to the personalization vector (uniform if not
    specified). This must be selected to result in an irreducible transition
    matrix (see notes under google_matrix). It may be common to have the
    dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
    Dictionary of nodes with PageRank as value

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence. The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The PageRank algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs by converting each edge in the
    directed graph to two edges.


    """

    # controlling the alpha parameter
    #alpha = 0.99

    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:

        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError('Personalization dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:

        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:

            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise NetworkXError('pagerank: power iteration failed to converge '
                        'in %d iterations.' % max_iter)


# this function calculates the L1 distance between PPR and PR without spam sites
def calculate_dist_l1(no_spammers, temp_pr):
    sum_of_ranks = 0
    for i in temp_pr:
        sum_of_ranks += abs(temp_pr[i] - no_spammers[i])
    dist = sum_of_ranks
    return dist


# this function calculates the L2 distance between PPR and PR without spam sites
def calculate_dist_l2(no_spammers, temp_pr):
    sum_of_ranks = 0
    for i in range(0, len(spammers)):
        sum_of_ranks += abs(math.pow(temp_pr[i], 2) - math.pow(no_spammers[i], 2))
    dist = math.sqrt(sum_of_ranks)
    return dist


# this function calculates the norm L1 of Ranks
def calculate_norm_l1(ranks):
    norm = sum(ranks.values())
    return norm


# clear from spammers
def clear_spamers(pr):
    for spamer in spammers:
        del pr[spamer]


# delete spammers from graph
def delete_spammers_from_graph(G):
    for spamer in spammers:
        G.remove_node(spamer)


# getting spam ranks
def get_spam_ranks(pr):
    spam_ranks = dict()
    for spamer in spammers:
        spam_ranks.update({spamer: pr.get(spamer)})

    return spam_ranks


# this function plots the example of network
def print_network(test_graph_file):
    G = create_graph(test_graph_file)
    nx.draw(G, node_color=color_map, with_labels=True, font_weight='bold')
    plt.savefig("pictures/sites connections.png")
    plt.close()


# progress printer
def print_progress(iteration_num, total_num):
    progress_percentage = math.floor(100 * (iteration_num / total_num))
    text_to_print = "In progress..." + str(progress_percentage) + "%"
    sys.stdout.write('\r' + text_to_print)


# performing new PPR
def another_PPR(temp_G, k, selected_trusted_sites, without_spammers_flag):
    ppr = dict()
    if k == 0:
        ppr = pagerank(temp_G, alpha=0.99, personalization=None, max_iter=100, tol=1.0e-6,
                       nstart=None, weight='weight', dangling=None)
    else:
        # setting the reset vector to probability of 1/k for k trusted sites
        personalization = dict.fromkeys(graph_set, 0)

        ## use when want different randomly seleted trusted site for K-min ann K-centers
        # selected_trusted_sites = random.sample(trusted, k)

        for site in selected_trusted_sites:
            personalization.update({site: (1.0 / k)})

        # if needed graph without spammers this part delete the spammers slot in reset vector dictionary
        if without_spammers_flag:
            for spammer in spammers:
                del personalization[spammer]

        ppr = pagerank(temp_G, alpha=0.99, personalization=personalization, max_iter=100, tol=1.0e-6,
                       nstart=None, weight='weight', dangling=None)
    return ppr


# performing researched PPR
#def researched_PPR(temp_G, k, selected_trusted_sites, without_spammers_flag):
#    min_pr = dict()
#    already_used = []
#    if k == 0:
#        min_pr = pagerank(temp_G, alpha=0.99, personalization=None, max_iter=100, tol=1.0e-6,
#                          nstart=None, weight='weight', dangling=None)
#    else:
#        # calculating the min PR from k page rank runs,
#        # each run is using reset vector with probability 1 for one trusted site
#        for i in range(0, k):
#            personalization = dict.fromkeys(graph_set, 0)
#            personalization.update({selected_trusted_sites[i]: 1})
#
#            ## use when want different randomly seleted trusted site for K-min ann K-centers
#            # personalization = set_personalization(already_used, without_spammers_flag)
#
#            temp_pr = pagerank(temp_G, alpha=0.99, personalization=personalization, max_iter=100, tol=1.0e-6,
#                               nstart=None, weight='weight', dangling=None)
#            if not min_pr:
#                min_pr = temp_pr
#            else:
#                for page in temp_pr:
#                    if temp_pr.get(page) < min_pr.get(page):
#                        min_pr.update({page: temp_pr.get(page)})
#
#        # normalizing the page rank
#        sum_pr = sum(min_pr.values())
#        #  saving the normlization factor
#        normlization_factor[num_trusted_vector.index(k)].append(sum_pr)
#
#        for page in min_pr:
#            min_pr.update({page: (min_pr.get(page) / sum_pr)})
#
#    return min_pr


# calculating the cost
#def cost_calculation(K_centers_PPR):
#    # sum of spmmers rank in K-centers
#    spammers_ranks = [rank for node, rank in K_centers_PPR.items() if node in spammers]
#    sum_of_spammers_ranks = sum(spammers_ranks)
#    # sum of non trusted rank in K-centers
#    non_trusted_ranks = [rank for node, rank in K_centers_PPR.items() if node not in trusted]
#    sum_of_non_trusted_ranks = sum(non_trusted_ranks)
#    return sum_of_spammers_ranks / sum_of_non_trusted_ranks


# challenge 1 - Distance L1
def distance_challenge():
    ppr_diff = []
    another_ppr_diff = []
    pr_diff = []
    iteration_num = 0.0

    # this part creating the distance l1 of regular PR
    pr_without_spammers = pagerank(G_without_spammers, alpha=0.99, personalization=None, max_iter=100, tol=1.0e-6,
                                   nstart=None, weight='weight', dangling=None)
    pr_with_spammers = pagerank(G, alpha=0.99, personalization=None, max_iter=100, tol=1.0e-6,
                                nstart=None, weight='weight', dangling=None)
    # clearing the spammers from rank
    clear_spamers(pr_with_spammers)
    pr_dist_l1 = calculate_dist_l1(pr_without_spammers, pr_with_spammers)

    for num in num_trusted_vector:
        # printing the progress of the loop
        print_progress(iteration_num, len(num_trusted_vector))
        iteration_num += 1.0

        # researched PPR
        # performing researched PPR on graph without spammers
        ppr_without_spammers = researched_PPR(G_without_spammers, num, 1)

        # performing researched PPR on graph with spammers
        ppr_with_spammers = researched_PPR(G, num, 0)

        # clearing the spammers from rank
        clear_spamers(ppr_with_spammers)
        # calculating the distance L1 between PPR without Spammers and PPR with Spammers
        ppr_diff.append(calculate_dist_l1(ppr_without_spammers, ppr_with_spammers))

        # another PPR
        # performing researched PPR on graph without spammers
        another_ppr_without_spammers = another_PPR(G_without_spammers, num, 1)

        # performing researched PPR on graph with spammers
        another_ppr_with_spammers = another_PPR(G, num, 0)

        # clearing the spammers from rank
        clear_spamers(another_ppr_with_spammers)

        # calculating the distance L1 between PPR without Spammers and PPR with Spammers
        ppr_diff.append(calculate_dist_l1(ppr_without_spammers, ppr_with_spammers))
        another_ppr_diff.append(calculate_dist_l1(another_ppr_without_spammers, another_ppr_with_spammers))
        pr_diff.append(pr_dist_l1)

    plt.figure(figsize=(8.7, 5.9))
    plt.plot(num_trusted_vector, ppr_diff, 'b', label="k-min with epsilon=0.99")
    plt.plot(num_trusted_vector, another_ppr_diff, 'g', label="k-centers with epsilon=0.99")
    plt.plot(num_trusted_vector, pr_diff, 'r', label="PR")
    plt.legend(loc="upper right")
    plt.xlabel("Number Of Trusted Sites")
    plt.ylabel("Distance L1")
    plt.title("Distance L1 between PPR with spam and PPR without spam Vs num of trusted sites")
    plt.savefig("pictures/Distance L1 between PPR with spam and PPR without spam Vs num of trusted sites.png")
    # plt.show()
    plt.close()

def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

def spam_challenge_prelim():
    #store trusted site sample
    trusted_site_set = random.sample(trusted, TRUSTED_SITES_NUM)
    with open('trusted_sites_store', 'w') as sites_file:
        json.dump(trusted_site_set, sites_file)

    store_pprs_85 = {site: {} for site in trusted_site_set}
    store_pprs_95 = {site: {} for site in trusted_site_set}
    store_pprs_99 = {site: {} for site in trusted_site_set}
    store_pprs_999 = {site: {} for site in trusted_site_set}

    #calculate ppr for each trusted site and store in store_pprs
    for selected_trusted_site in trusted_site_set:
        personalization = dict.fromkeys(graph_set, 0)
        personalization.update({selected_trusted_site: 1})

        temp_ppr_85 = pagerank(G, alpha=0.85, personalization=personalization,
                max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
                dangling=None)
        #temp_sorted = [node for node, rank in sorted(temp_ppr_85.items(), key=lambda item: item[1])]
        print(selected_trusted_site, ",", temp_ppr_85.get(selected_trusted_site))
        #print(sorted(temp_ppr_85.items(), key=lambda item: item[1])[-20:])
        temp_ppr_95 = pagerank(G, alpha=0.95, personalization=personalization,
                max_iter=100, tol=1.0e-6, nstart=temp_ppr_85, weight='weight',
                dangling=None)
        temp_ppr_99 = pagerank(G, alpha=0.99, personalization=personalization,
                max_iter=100, tol=1.0e-6, nstart=temp_ppr_95, weight='weight',
                dangling=None)
        temp_ppr_999 = pagerank(G, alpha=0.999, personalization=personalization,
                max_iter=100, tol=1.0e-6, nstart=temp_ppr_99, weight='weight',
                dangling=None)

        store_pprs_85.update({selected_trusted_site: temp_ppr_85})
        store_pprs_95.update({selected_trusted_site: temp_ppr_95})
        store_pprs_99.update({selected_trusted_site: temp_ppr_99})
        store_pprs_999.update({selected_trusted_site: temp_ppr_999})

    with open('store_pprs_85_file', 'w') as store_pprs_85_file:
      json.dump(store_pprs_85, store_pprs_85_file)
    with open('store_pprs_95_file', 'w') as store_pprs_95_file:
      json.dump(store_pprs_95, store_pprs_95_file)
    with open('store_pprs_99_file', 'w') as store_pprs_99_file:
      json.dump(store_pprs_99, store_pprs_99_file)
    with open('store_pprs_999_file', 'w') as store_pprs_999_file:
      json.dump(store_pprs_999, store_pprs_999_file)

# challenge 2 - Spam Resistance
def spam_post_compute(epsilon, selected_indices_store):
    spammers_researched_ppr = []
    spammers_another_ppr = []
    spammers_pr = []
    iteration_num = 0.0
    #avg_researched_ppr = {}
    #avg_another_ppr = {}
    another_ppr_spammers_mean_ordinal_postion = []
    researched_ppr_spammers_mean_ordinal_postion = []
    pr_spammers_mean_ordinal_postion = []

    print(epsilon)
    # this part creating the spammers sum of regular PR
    pr = pagerank(G, alpha=epsilon, personalization=None, max_iter=100, tol=1.0e-6,
                  nstart=None, weight='weight', dangling=None)
    print("computed regular pr")
    sum_spammers_pr = calculate_norm_l1(get_spam_ranks(pr))
    print("spammers rank for regular pr: {}".format(sum_spammers_pr))

    # calculating the ordinal position of the spammers in regular PR
    pr_sorted = [node for node, rank in sorted(pr.items(), key=lambda item: item[1])]
    pr_spammers_ordinal_postions = {node: position for position, node in enumerate(reversed(pr_sorted)) if node in spammers}

    store_researched_ppr = {num: [] for num in num_trusted_vector} #dict.fromkeys(num_trusted_vector, [])
    store_avg_ppr = {num: [] for num in num_trusted_vector}
    store_researched_ppr_spammers = {num: [] for num in num_trusted_vector}
    store_another_ppr_spammers = {num: [] for num in num_trusted_vector}
    store_spammers_ordinal_researched = {num: [] for num in num_trusted_vector}
    store_spammers_ordinal_another = {num: [] for num in num_trusted_vector}
    store_trusted_selections = {num: [] for num in num_trusted_vector}
    
    epsilon_string = ""
    if epsilon == 0.999:
        epsilon_string = "999"
    else:
        epsilon_string = str(int(epsilon * 100))
    #print("epsilon_string = {}".format(epsilon_string))
    #load the stored pprs and trusted sites
    with open('trusted_sites_store_500') as sites_file:
        trusted_site_set = json.load(sites_file, object_hook=jsonKeys2int)
    with open('store_pprs_{}_500'.format(epsilon_string)) as file:
        store_pprs_old = json.load(file, object_hook=jsonKeys2int)

    print("loaded files")
    store_pprs = { page: [] for page in graph_set }
    for trusted_site in trusted_site_set:
        for page in graph_set:
            store_pprs[page].append(store_pprs_old[trusted_site][page])
    print("transposed pr vectors")

    #start = time.time()
    for num in num_trusted_vector:
        #researched_ppr = {}
        #another_ppr = {}

        # printing the progress of the loop
        print_progress(iteration_num, len(num_trusted_vector))
        iteration_num += 1.0

        # averaging loop
        for run in range(AVERAGE_RUN_NUM):
            print("k = {}, run = {}, epsilon = {}".format(num, run, epsilon))
            min_pr = {}
            avg_pr = {}
            selected_trusted_sites = []
            selected_site_indices = selected_indices_store.get(num)[run]
            for i in selected_site_indices:
                selected_trusted_sites.append(trusted_site_set[i])
            store_trusted_selections[num].append(selected_trusted_sites)
            #print(selected_site_indices)
            #print(selected_trusted_sites)
            for page in graph_set:
                page_pprs = store_pprs[page]
                min_pr[page] = min([page_pprs[i] for i in selected_site_indices])
                avg_pr[page] = statistics.mean([page_pprs[i] for i in selected_site_indices])
            #print(min_pr[0])
            #print(avg_pr[0])

            # normalizing the page rank
            sum_pr = sum(min_pr.values())
            # saving the normlization factor
            #normlization_factor[num_trusted_vector.index(num)].append(sum_pr)

            for page in min_pr:
                min_pr.update({page: (min_pr.get(page) / sum_pr)})
            #sum_avg = sum(avg_pr_99.values())
            #sum_min = sum(min_pr_99.values())
            #print(sum_avg)
            #print(sum_min)

            #try rank before average
            researched_sorted = [node for node, rank in sorted(min_pr.items(), key=lambda item: item[1])]
            researched_spammers_ordinal_positions = {node: position for position, node in enumerate(reversed(researched_sorted)) if node in spammers}
            store_spammers_ordinal_researched[num].append(statistics.mean(researched_spammers_ordinal_positions.values()))
            another_sorted = [node for node, rank in sorted(avg_pr.items(), key=lambda item: item[1])]
            another_spammers_ordinal_positions = {node: position for position, node in enumerate(reversed(another_sorted)) if node in spammers}
            store_spammers_ordinal_another[num].append(statistics.mean(another_spammers_ordinal_positions.values()))

            #if not researched_ppr:
            #    for key, value in min_pr.items():
            #        researched_ppr[key] = [value]
            #    for key, value in avg_pr.items():
            #        another_ppr[key] = [value]
            #else:
            #    for key, value in min_pr.items():
            #        researched_ppr[key].append(value)
            #    for key, value in avg_pr.items():
            #        another_ppr[key].append(value)

            #store_researched_ppr[num].append(researched_ppr)
            #store_avg_ppr[num].append(another_ppr)

            store_researched_ppr_spammers[num].append(calculate_norm_l1(get_spam_ranks(min_pr)))
            store_another_ppr_spammers[num].append(calculate_norm_l1(get_spam_ranks(avg_pr)))
            #end = time.time()
            #print(end - start)
            #for spammer in spammers:
            #    print(spammer ,":", min_pr.get(spammer))
            #for spammer in spammers:
            #    print(spammer ,":", avg_pr.get(spammer))

        #for key, value in researched_ppr.items():
        #    avg_researched_ppr[key] = statistics.mean(value)
        #for key, value in another_ppr.items():
        #    avg_another_ppr[key] = statistics.mean(value)

        #hk edits
        researched_ppr_spammers_mean_ordinal_postion.append(statistics.mean(store_spammers_ordinal_researched[num]))
        another_ppr_spammers_mean_ordinal_postion.append(statistics.mean(store_spammers_ordinal_another[num]))

        # adding the sum of spammers rank to list
        #spammers_another_ppr.append(calculate_norm_l1(get_spam_ranks(avg_another_ppr)))
        #spammers_researched_ppr.append(calculate_norm_l1(get_spam_ranks(avg_researched_ppr)))

        spammers_researched_ppr.append(statistics.mean(store_researched_ppr_spammers[num]))
        spammers_another_ppr.append(statistics.mean(store_another_ppr_spammers[num]))

        # appending regualr PR vectors
        pr_spammers_mean_ordinal_postion.append(statistics.mean(pr_spammers_ordinal_postions.values()))
        spammers_pr.append(sum_spammers_pr)
    
    print("\ncomputed ranks")
    # calculate the mean of the normlization vector of each k
    #mean_normlization_factor = [statistics.mean(k_values_list) for k_values_list in normlization_factor]

    # saving the results in file
    now=datetime.now()
    dt_string = now.strftime("%d-%m-%y_%H:%M")
    with open('spam_res_{}_{}'.format(dt_string, epsilon_string), 'w') as spam_res_file:
      # Test parameters
      spam_res_file.write('Test Results parameters:\n')
      spam_res_file.write("Number of samples = %s\n" % SAMPLES_NUM)
      spam_res_file.write("Max K = %s\n" % MAX_K)
      spam_res_file.write("Averaging run number = %s\n" % AVERAGE_RUN_NUM)
      #spam_res_file.write("Epsilon = %s\n" % EPSILON)
      # saving K vector
      spam_res_file.write('\n')
      spam_res_file.write('num_trusted_vector\n')
      for item in num_trusted_vector:
        spam_res_file.write("%s " % item)
      #
      # saving K-min sum of spammers
      spam_res_file.write('\n')
      spam_res_file.write('\nK-min_{}\n'.format(epsilon_string))
      for item in spammers_researched_ppr:
        spam_res_file.write("%s " % item)
        #
      # saving K-centers sum of spammers
      spam_res_file.write('\n')
      spam_res_file.write('\nK-centers_{}\n'.format(epsilon_string))
      for item in spammers_another_ppr:
        spam_res_file.write("%s " % item)
        #
      # saving regular pr sum of spammers
      spam_res_file.write('\n')
      spam_res_file.write('\nregular_pr_{}\n'.format(epsilon_string))
      for item in spammers_pr:
        spam_res_file.write("%s " % item)
        #
      # saving the normalization vector
      #spam_res_file.write('\n')
      #spam_res_file.write('K-min normlization vector\n')
      #for item in mean_normlization_factor:
      #  spam_res_file.write("%s " % item)
        #
      # saving the ordinal postions vectors
      spam_res_file.write('\n')
      spam_res_file.write('\nK-min_spammers_mean_ordinal_postions_{}\n'.format(epsilon_string))
      for item in researched_ppr_spammers_mean_ordinal_postion:
        spam_res_file.write("%s " % item)
        #
      spam_res_file.write('\n')
      spam_res_file.write('\nK-centers-85-spammers-mean-ordinal-postions_{}\n'.format(epsilon_string))
      for item in another_ppr_spammers_mean_ordinal_postion:
        spam_res_file.write("%s " % item)
      # 
      spam_res_file.write('\n')
      spam_res_file.write('\nregular-PR-85-spammers mean-ordinal-postions_{}\n'.format(epsilon_string))
      for item in pr_spammers_mean_ordinal_postion:
        spam_res_file.write("%s " % item)
     #
     # saving the individual runs for spam ranks
      spam_res_file.write('\n')
      spam_res_file.write('\n')
      spam_res_file.write('K-min_spammer_ranks_each_run_{}\n'.format(epsilon_string))
      for key, value in store_researched_ppr_spammers.items():
        spam_res_file.write("%s: %s\n" % (key, ' '.join(map(str,value))))
      #
      spam_res_file.write('\n')
      spam_res_file.write('K-centers_spammer_ranks_each_run_{}\n'.format(epsilon_string))
      for key,value in store_another_ppr_spammers.items():
        spam_res_file.write("%s: %s\n" % (key, ' '.join(map(str,value))))
      #
      # saving the individual runs for spam ordinals
      spam_res_file.write('\n')
      spam_res_file.write('K-min_spammer_ordinals_each_run_{}\n'.format(epsilon_string))
      for key, value in store_spammers_ordinal_researched.items():
        spam_res_file.write("%s: %s\n" % (key, ' '.join(map(str,value))))
      #
      spam_res_file.write('\n')
      spam_res_file.write('K-centers_spammer_ordinals_each_run_{}\n'.format(epsilon_string))
      for key, value in store_spammers_ordinal_another.items():
        spam_res_file.write("%s: %s\n" % (key, ' '.join(map(str,value))))
        #
      spam_res_file.write('\ntrusted site selections\n')
      for key, value in store_trusted_selections.items():
          spam_res_file.write("%s: %s\n" % (key, ' '.join(map(str,value))))
    print("wrote to spam_res file")


    #plotting ordinal positions vectors
    plt.figure(figsize=(7.7, 5.3))
    plt.plot(num_trusted_vector, researched_ppr_spammers_mean_ordinal_postion, 'b', label="k-min with epsilon={}".format(round(1-epsilon,3)))
    plt.plot(num_trusted_vector, another_ppr_spammers_mean_ordinal_postion, 'g', label="k-centers with epsilon=0{}".format(round(1-epsilon,3)))
    plt.plot(num_trusted_vector, pr_spammers_mean_ordinal_postion, 'r', label="PR")
    plt.legend(loc="upper right")
    plt.xlabel("Number Of Trusted Sites")
    plt.ylabel("Spammers mean ordinal position")
    plt.title("Spammers mean ordinal position Vs Trusted sites number")
    plt.savefig("Spammers mean ordinal position Vs Trusted sites number 7-5 {}.png".format(epsilon_string))

    #plotting the sum of spammers vectors
    plt.figure(figsize=(7.7, 5.3))
    plt.plot(num_trusted_vector, spammers_researched_ppr, 'b', label="k-min with epsilon={}".format(round(1-epsilon,3)))
    plt.plot(num_trusted_vector, spammers_another_ppr, 'g', label="k-centers with epsilon={}".format(round(1-epsilon,3)))
    plt.plot(num_trusted_vector, spammers_pr, 'r', label="PR")
    plt.legend(loc="upper right")
    plt.xlabel("Number Of Trusted Sites")
    plt.ylabel("Sum of Spam sites rank")
    plt.title("Sum of Spam sites rank Vs Trusted sites number")
    plt.savefig("Sum of Spam sites rank Vs Trusted sites number 7-5 {}.png".format(epsilon_string))
    #text_tp_print = "The spam sites rank declined in " + str(abs(100 * ((spammers_researched_ppr[len(num_trusted_vector) - 1] -
    #                                                                     spammers_pr[0]) / spammers_pr[0]))) + "%"
    # printing the normalization vector
    #print(f"\nK-min normlization vector:\n{mean_normlization_factor}\n")
    #return text_tp_print


# challenge 3 - Distortion
def distortion_challenge():
    stdev_researched_ppr_diff = []
    stdev_another_ppr_diff = []
    # personalization.clear()
    # nstart.clear()
    # G = create_graph(graph_file, 0, 1)
    #
    # # deleting the spam sites from the web graph
    # delete_spammers_from_graph(G)

    pr_no_trusted = pagerank(G, alpha=0.99, personalization=None, max_iter=100, tol=1.0e-6,
                             nstart=None, weight='weight', dangling=None)
    pr_no_trusted_stdev = statistics.stdev(pr_no_trusted.values())

    iteration_num = 0.0
    for num in num_trusted_vector:
        # printing the progress of the loop
        print_progress(iteration_num, len(num_trusted_vector))
        iteration_num += 1.0

        researched_ppr = researched_PPR(G, num, 0)
        another_ppr = another_PPR(G, num, 0)

        # calculating the difference between Standard Deviation of PR and PPR as an index for distortion
        stdev_researched_ppr_diff.append(statistics.stdev(researched_ppr.values()) - pr_no_trusted_stdev)
        stdev_another_ppr_diff.append(statistics.stdev(another_ppr.values()) - pr_no_trusted_stdev)

    plt.figure(figsize=(7.7, 5.3))
    plt.plot(num_trusted_vector, stdev_researched_ppr_diff, 'b', label="k-min with epsilon=0.99")
    plt.plot(num_trusted_vector, stdev_another_ppr_diff, 'g', label="k-centers with epsilon=0.99")
    plt.legend(loc="upper right")
    plt.xlabel("Number Of Trusted Sites")
    plt.ylabel("Distortion")
    plt.title("Distortion Vs Trusted sites number")
    plt.savefig("Distortion Vs Trusted sites number.png")
    # plt.show()
    decline_percentage = abs(
        100 * ((stdev_researched_ppr_diff[len(num_trusted_vector) - 1] - max(stdev_researched_ppr_diff)) /
               max(stdev_researched_ppr_diff)))
    text_tp_print = "The Distortion decline from pick to max number of trusted sites is " + str(
        decline_percentage) + "%"
    # print("The Distortion decline from pick to max number of trusted sites is " + str(decline_percentage) + "%")
    return text_tp_print


# gathering network data from files
def gather_network_details(filename):
    f = open(filename, "r")
    line = f.readline()
    hostID_num = int(line)
    in_degree = dict.fromkeys(graph_set, 0)

    # counting links from trusted sites to spammers
    trusted_to_spam = dict.fromkeys(trusted, 0)
    number_of_links_from_strusted_to_spammers = 0

    for node in range(0, hostID_num):
      line = f.readline()
      if node in graph_set:
          num_links = 0
          if line != "\n":
              dests = line.split(" ")
              for dest in dests:
                  dest_parts = dest.split(":")
                  ## non paralel links data
                  if int(dest_parts[0]) in graph_set:
                      # counting links from trusted sites to spammers
                      if node in trusted and int(dest_parts[0]) in spammers:
                          trusted_to_spam[node] += 1
                          number_of_links_from_strusted_to_spammers += 1
                      in_degree.update({int(dest_parts[0]): in_degree.get(int(dest_parts[0])) + 1})
                      num_links += 1
                  ## paralel links data
                  # in_degree.update({int(dest_parts[0]): in_degree.get(int(dest_parts[0])) + int(dest_parts[1])})
                  # num_links += int(dest_parts[1])
          out_degree.append(num_links)

    total_number_of_links = sum(out_degree)
    print ("Total number of links in the web = " + str(total_number_of_links) + "\n")
    print ("Out degree statistics")
    max_number_of_out_degree = max(out_degree)
    print ("Max out degree = " + str(max_number_of_out_degree))
    min_number_of_out_degree = min(out_degree)
    print ("Min out degree = " + str(min_number_of_out_degree))
    avg_number_of_out_degree = statistics.mean(out_degree)
    print ("Average out degree = " + str(avg_number_of_out_degree) + "\n")
    print ("In degree statistics")
    max_number_of_in_degree = max(in_degree.values())
    print ("Max in degree = " + str(max_number_of_in_degree))
    min_number_of_in_degree = min(in_degree.values())
    print ("Min in degree = " + str(min_number_of_in_degree))
    avg_number_of_in_degree = statistics.mean(in_degree.values())
    print ("Average in degree = " + str(avg_number_of_in_degree) + "\n")
    print ("Number of links from trusted sites to spammers = " + str(number_of_links_from_strusted_to_spammers) + "\n")
    trusted_site_with_most_links_to_spam, most_links_to_spam = (sorted(trusted_to_spam.items(), key=lambda item: item[1])[-1])
    print (f"Trusted site with the largest amount of link to spammers is {trusted_site_with_most_links_to_spam} with {most_links_to_spam} links\n")

    return hostID_num


## MAIN
# initializing files names
graph_file = "hostgraph_weighted.txt"
assessments_file_1 = "assessments_1.txt"
assessments_file_2 = "assessments_2.txt"
strongly_connected_component_file = 'largest strongly connected component.txt'

# initializing variables
trusted = set()  # set of trusted sites
spammers = set()  # set of spam sites
strongly_connected_component = set()
color_map = []  # for graph plot
# personalization = dict()  # for PPR
nstart = dict()  # for controlling reset vector
out_degree = []
AVERAGE_RUN_NUM = 50
MAX_K = 30
SAMPLES_NUM = 30
TRUSTED_SITES_NUM = 500
#EPSILON = 0.99


# # plotting an example for web graph
# print("Plotting an example for Webgraph...")
# get_assessments("test_assessment.txt")
# print_network("test_graph.txt")
# trusted.clear()
# spammers.clear()
# print("Done")

# get graph size
graph_set = get_graph_set(graph_file)
print(f"Graph size = {len(graph_set)}")

# gathering strongly connected component details
get_strongly_connected_component(strongly_connected_component_file)
#print(f"Strongly connected component size = {len(strongly_connected_component)}")

# dealing with the real web graph
get_assessments(assessments_file_1)  # get classification for first part of sites
get_assessments(assessments_file_2)  # get classification for second part of sites
num_trusted_vector = list(np.arange(1, MAX_K + 1, (MAX_K // SAMPLES_NUM)))

# initialize normalization factor
normlization_factor = [[] for k in num_trusted_vector]

#print(f"Trusted sites num = {len(trusted)} \nSpam sites num = {len(spammers)}")
print(f"Spammers density is {len(spammers) / len(strongly_connected_component)}")

## Gathering network details
#print("Gathering network details...")
#hostID_num = gather_network_details(graph_file)
#print("\rDone\n")

# creating graph
print("Creating Graph...")
G = create_graph(graph_file)
# creating a graph without spammers
G_without_spammers = create_graph(graph_file)
# deleting the spam sites from the web graph
delete_spammers_from_graph(G_without_spammers)
print("\rDone\n")

# printing the test details
print("The test run parameters:")
print(f"AVERAGE_RUN_NUM = {AVERAGE_RUN_NUM}")
print(f"MAX_K = {MAX_K}")
print(f"SAMPLES_NUM = {SAMPLES_NUM}\n")

#sampling the trusted sites
print("Sampling the trusted sites...")
#with open('trusted_sites_store_500') as sites_file:
#    trusted_site_set = json.load(sites_file, object_hook=jsonKeys2int)
trusted_index_selections = {num: [] for num in num_trusted_vector}
for k in num_trusted_vector:
    for run in range(AVERAGE_RUN_NUM):
        selected_indices = random.sample(range(0,TRUSTED_SITES_NUM), k)
        trusted_index_selections[k].append(selected_indices)

# plotting the ranks of spammers as function of trusted sites number
print("Attacking the spam challenge...")
#spam_challenge_prelim()
spam_rank_decline = spam_post_compute(0.85, trusted_index_selections)
spam_rank_decline = spam_post_compute(0.95, trusted_index_selections)
spam_rank_decline = spam_post_compute(0.99, trusted_index_selections)
spam_rank_decline = spam_post_compute(0.999, trusted_index_selections)
#spam_rank_decline = spam_challenge()
#print("\r" + spam_rank_decline)
print("Done\n")

# # plotting the Distortion as function of trusted sites number
# print("Attacking the distortion challenge...")
# Distortion_Decline = distortion_challenge()
# print("\r" + Distortion_Decline)
# print("Done")

# # plotting the L1 distance between PPR with spam and PPR without spam Vs number of trusted sites
# print("Attacking the distance challenge...")
# distance_challenge()
# print("\rDone")

# # playing a beep to declare the end of the simulation
# frequency = 1500  # Set Frequency To 1500 Hertz
# duration = 500  # Set Duration To 500 ms == 0.5 second
# winsound.Beep(frequency, duration)

