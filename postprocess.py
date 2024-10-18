import networkx as nx
import numpy as np
import statistics 
import collections
from generate_trainset import show_map_slow
import time

import random

import heapq

def merge_clusters_dict(clusters, cluster1, cluster2):
    for key in clusters:
        if clusters[key] == cluster2:
            clusters[key] = cluster1
    pass

def is_endpoint(wire, point):
    counter = 0
    threshold = 0.5
    max_size = 511
    if point[0] > 0 and wire[point[0] - 1, point[1],0] >= threshold:
        counter += 1
    if point[1] > 0 and wire[point[0], point[1] - 1, 0] >= threshold:
        counter += 1
    if point[0] < max_size and wire[point[0] + 1, point[1], 0] >= threshold:
        counter += 1
    if point[1] < max_size and wire[point[0], point[1] + 1, 0] >= threshold:
        counter += 1

    return counter < 2

def manhattan(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    pass

def get_neighbors(wire, design, point):
    neighbors = []

    max_size = 511

    if point[0] > 0 and design[point[0] - 1, point[1], 0] == 0:
        neighbors.append( (point[0] - 1, point[1]) )
    if point[1] > 0 and design[point[0], point[1] - 1, 0] == 0:
        neighbors.append( (point[0], point[1] - 1) )
    if point[0] < max_size and design[point[0] + 1, point[1], 0] == 0:
        neighbors.append( (point[0] + 1, point[1]) )
    if point[1] < max_size and design[point[0], point[1] + 1, 0] == 0:
        neighbors.append( (point[0], point[1] + 1) )

    return neighbors


def connect_endpoints_astar_actual(wire, design, s, t):
    

    distance = {s: 0}

    #heuristics
    h = lambda x: manhattan(x, t)

    open = [(h(s),s)]

    prev = {s: None}

    counter = 0
    flag = False

    while len(open) > 0:
        _, current = heapq.heappop(open)
        for n in get_neighbors(wire, design, current):
            counter += 1
            if n not in distance:
                distance[n] = distance[current] + 1
                heapq.heappush(open, (h(n), n) )
                #open.append(n)

                prev[n] = current

                # as far as it is bfs, break after first detection
                if (n[0] == t[0]) and (n[1] == t[1]):
                    print("Traversed nodes count = ", counter)
                    flag = True

            else:
                if distance[n] > distance[current] + 1:
                    distance[n] = distance[current] + 1
                    prev[n] = current

        if flag:
            break

    #now reproducing the path
    current = t
    path = []
    while current != None:
        path.append(current)
        current = prev[current]

    return path
    pass

def connect_endpoints_astar(wire, design, s, t):

    #ok, no astar today, djikstra instead

    #situation when path is not available IS NOT COVERED

    open = [s]

    distance = {s:0}

    counter = 0

    while len (open) > 0:
        current = open.pop(0)
        for n in get_neighbors(wire, design, current):
            counter += 1
            if n not in distance:
                distance[n] = distance[current] + 1
                open.append(n)

                # as far as it is bfs, break after first detection
                if (n[0] == t[0]) and (n[1] == t[1]):
                    print("Traversed nodes count = ", counter)
                    break

            else:
                distance[n] = max(distance[n], distance[current] + 1)


    #now reproducing the path
    current = t
    path = []
    while current != s:
        path.append(current)
        min_distance = distance[current]
        opt_node = None
        for n in get_neighbors(wire, design, current):
            if distance[n] < min_distance:
                min_distance = distance[n]
                opt_node = n
        current = opt_node

    return path
    pass

def find_closest_endpoints(endpoints, clusters, wires, design):
    #assuming that every cluster has at least one endpoint

    #situation when cluster has no endpoins IS NOT COVERED

    #for each cluster, looking for closest another clusters endpoint
    while len( endpoints.keys() ) > 1:
        
        
        first_key = list(endpoints.keys())[0]

        closest_manhattan = 1000000
        closest_endpoint = None
        closest_endpoint2 = None

        closest_cluster = -1

        for endpoint in endpoints[first_key]:
            #iterating thought other clusters:
            for key in list(endpoints.keys())[1:]:
                for another_endpoint in endpoints[key]:
                    if manhattan(endpoint, another_endpoint) < closest_manhattan:
                        closest_manhattan = manhattan(endpoint, another_endpoint)
                        closest_endpoint = another_endpoint
                        closest_endpoint2 = endpoint
                        closest_cluster = key

        print("Closest endpoints found: ({},{}) and ({},{}), connecting.".format(endpoint[0], endpoint[1], another_endpoint[0], another_endpoint[1]))
        path = connect_endpoints_astar_actual(wires, design, closest_endpoint2, closest_endpoint)

        print("Path found:", path)

        #adding this path to the wire
        for p in path:
            wires[p[0],p[1],0] = 1.0

        #and merging clusters:
        endpoints[first_key] = endpoints[first_key] + endpoints[closest_cluster]
        del endpoints[closest_cluster]


    pass

def add_extra_endpoints(wire, endpoints):
    pass


def cluster_wire_fast(wire):

    only_endpoints = False

    in_cluster = {}
    #dict that maps pixel into clusters

    endpoints = {}
    #dict that maps cluster id to its endopoints

    cluster_counter = 0

    clusters = []
    #array of clusters ids of present clusters

    merged_clusters = {}
    #dictionary that maps initial cluster number to merged cluster number

    threshold = 0.5
    for x in range(wire.shape[0]):
        for y in range(wire.shape[1]):
            if wire[x, y, 0] >= threshold:
                if ((x-1, y) in in_cluster) and ((x, y-1) in in_cluster) and (in_cluster[(x, y - 1)] != in_cluster[(x - 1, y)]):
                    #merging two clusters in one, by assigning the second cluster to first
                    #if clusters are different

                    clusters.remove(in_cluster[(x, y - 1)])

                    merge_clusters_dict(
                        in_cluster, in_cluster[(x-1, y)], in_cluster[(x, y - 1)])

                    # also merging endpoints
                    endpoints[in_cluster[(x-1, y)]] = endpoints.get(in_cluster[(x-1, y)], []) + endpoints.get(in_cluster[(x, y-1)],[])

                    in_cluster[(x, y)] = in_cluster[(x - 1, y)]

                elif (x-1, y) in in_cluster:
                    in_cluster[(x, y)] = in_cluster[(x - 1, y)]
                elif (x, y-1) in in_cluster:
                    in_cluster[(x, y)] = in_cluster[(x, y - 1)]
                else:
                    in_cluster[(x, y)] = cluster_counter
                    clusters.append(cluster_counter)
                    cluster_counter += 1
                if is_endpoint(wire, (x, y)) or (random.randint(0, 4) == 0):
                    if in_cluster[(x, y)] in endpoints:
                        endpoints[in_cluster[(x, y)]].append((x, y))
                    else:
                        endpoints[in_cluster[(x, y)]] = [(x, y)]



    # dirty hack - if cluster has only one endpoint, it is probably an unwanted noise
    keys = list(endpoints.keys())
    for key in keys:
        if len(endpoints[key]) == 1:
            del endpoints[key]
            if key in clusters:
                clusters.remove(key)

    return endpoints, clusters
    pass

def cluster_wires(wire):

    only_endpoints = False


    in_cluster = {}
    #dict that maps pixel into clusters

    endpoints = {}
    #dict that maps cluster id to its endopoints

    cluster_counter = 0

    clusters = []
    #array of clusters ids of present clusters

    merged_clusters = {}
    #dictionary that maps initial cluster number to merged cluster number 


    threshold = 0.5
    for x in range(wire.shape[0]):
        for y in range(wire.shape[1]):
            if wire[x,y,0] >= threshold:
                if ((x-1, y) in in_cluster) and ((x, y-1) in in_cluster) and (in_cluster[(x, y - 1)] != in_cluster[(x - 1, y)]):
                    #merging two clusters in one, by assigning the second cluster to first
                    #if clusters are different

                    clusters.remove(in_cluster[(x, y - 1)])

                    merge_clusters_dict(in_cluster, in_cluster[(x-1, y)], in_cluster[(x, y - 1)])


                    in_cluster[(x, y)] = in_cluster[(x - 1, y)]

                elif (x-1, y) in in_cluster:
                    in_cluster[(x, y)] = in_cluster[(x - 1, y)]
                elif (x, y-1) in in_cluster:
                    in_cluster[(x, y)] = in_cluster[(x, y - 1)]
                else:
                    in_cluster[(x,y)] = cluster_counter
                    clusters.append(cluster_counter)
                    cluster_counter += 1

    # another cycle to avoid mess with cluster id updates in endpoints
    for x in range(wire.shape[0]):
        for y in range(wire.shape[1]):
            if wire[x, y, 0] >= threshold:

                if only_endpoints is True:
                    if is_endpoint(wire, (x,y)):
                        if in_cluster[(x,y)] in endpoints:
                            endpoints[in_cluster[(x,y)]].append((x,y))
                        else:
                            endpoints[in_cluster[(x, y)]] = [(x,y)]
                else:
                    if is_endpoint(wire, (x, y)) or (random.randint(0,4) == 0):
                        if in_cluster[(x, y)] in endpoints:
                            endpoints[in_cluster[(x, y)]].append((x, y))
                        else:
                            endpoints[in_cluster[(x, y)]] = [(x, y)]


    # dirty hack - if cluster has only one endpoint, it is probably an unwanted noise
    keys = list(endpoints.keys())
    for key in keys:
        if len(endpoints[key]) == 1:
            del endpoints[key]
            clusters.remove(key)

    return endpoints, clusters

    pass

def postprocess_greedy(design, wire):
    endpoints, clusters = cluster_wires(wire)

    print("Clustering finished, {} clusters found.".format(len(clusters)))
    for c in clusters:
        print("Cluster {} has {} endpoints".format(c, len(endpoints[c])))

    find_closest_endpoints(endpoints, clusters, wire, design)

    return wire
    pass

def build_graph_from_map(map):
    # building aa graph, when there is an edge between unobstacled modes and no edge between obstacled
    threshold = 0.5
    G = nx.Graph()
    for x in range(1,map.shape[0]):
        for y in range(1,map.shape[1]):
            if map[x-1][y][0] <= threshold and map[x][y][0] <= threshold:
                G.add_edge((x-1, y), (x,y))
            if map[x][y-1][0] <= threshold and map[x][y][0] <= threshold:
                G.add_edge((x, y-1), (x,y))

    return G


def start_bfs(start_point, g, target_nodes):
    Q = [start_point]
    visited = []
    while len(Q) > 0:
        current = Q.pop(0)
        for n in nx.neighbors(g, current):
            if n in visited: continue
            Q.append(n)
            if n in target_nodes:
                return n
    pass

def connect_with_astar(node_1, node_2, g):
    return nx.shortest_path(g, node_1, node_2)

# def merge_clusters_real(cluster1, cluster2, g):
#
#     #if two clusters intersects, start BFS from center of this intersection
#     cluster1.bb_min
#     intersection_min = (max (cluster1.bb_min[0] , cluster2.bb_min[0]) ,
#                      max (cluster1.bb_min[1], cluster2.bb_min[1]))
#     intersection_max = (min (cluster1.bb_max[0]), cluster2.bb_max[0]),
#                         min (cluster1.bb_max[1], cluster2.bb_max[1])
#
#     #if intersection_max[0] >= intersection_min[0] and intersection_max[1] >= intersection_min[1]:
#         #ok, intersection exists
#     #    pass
#     pass

def merge_clusters(cluster1, cluster2):
    Cluster = collections.namedtuple(
         "Cluster", "graph bb_max bb_min center_of_mass comment", defaults=(None, None, None, None, ""))

    print("merging ({}) and ({})".format(cluster1.comment, cluster2.comment))
    return Cluster(comment = "(" + cluster1.comment + "+" + cluster2.comment + ")", center_of_mass = 
    (cluster1.center_of_mass[0]/2 + cluster2.center_of_mass[0]/2,
     cluster1.center_of_mass[1]/2 + cluster2.center_of_mass[1]/2)
    )
    pass

def get_neighbors_global(node, clusters):
    return clusters

def ward_merge(clusters, get_neighbors = None):
    
   

    if get_neighbors is None:
        get_neighbors = get_neighbors_global
    

    def find_closest(n, c):
        
        def distance(a,b):
            # manhattan?
            return abs(a.center_of_mass[0] - b.center_of_mass[0]) + abs(a.center_of_mass[1] - b.center_of_mass[1])

        neighs = get_neighbors(n, c)
        min_distance = 1000000
        closest = None
        for neigh in neighs:
            if n == neigh:
                continue
            if distance(n, neigh) < min_distance:
                min_distance = distance(n, neigh)
                closest = neigh
        
        return closest
            

    S = []


    while len(S) > 0 or len(clusters) > 1:
        if len (S) == 0:
            c = clusters.pop(0)
            S.insert(0, c)
        c = S[0]
        d = find_closest(c, clusters + S)
        if d in S:
            S.pop(S.index(d))
            S.pop(S.index(c))
            clusters.append(merge_clusters(c, d))
        else:
            clusters.remove(d)
            S.insert(0,d)
    
    return clusters[0]




def get_clusters(predict, threshold = 0.5):
    #building networkx graph
    G = nx.Graph()
    for x in range(1,predict.shape[0]):
        for y in range(1,predict.shape[1]):
            if predict[x-1][y][0] >= threshold and predict[x][y][0] >= threshold:
                G.add_edge((x-1, y), (x,y))
            if predict[x][y-1][0] >= threshold and predict[x][y][0] >= threshold:
                G.add_edge((x, y-1), (x,y))
    #obtaining clusters
    subgraphs = nx.connected_components(G)
    
    Cluster = collections.namedtuple("Cluster", "graph bb_max bb_min center_of_mass comment")
 
    clusters = []

    for subgraph in subgraph:
        #calculating the bounig box and centers of mass
        bb_max = (max([x for x, y in subgraph.nodes]), max([y for x, y in subgraph.nodes]))
        bb_min = (min([x for x, y in subgraph.nodes]),
                  min([y for x, y in subgraph.nodes]))
        center_of_mass = (statistics.mean([[x for x, y in subgraph.nodes]]),
                          statistics.mean([[y for x, y in subgraph.nodes]]))
        
        clusters.append(Cluster(graph = subgraph, bb_max = bb_max, bb_min = bb_min, center_of_mass = center_of_mass))
    
    return clusters
  

    #implementing Wards algorithms

def try_merging():
    clusters = []
    Cluster = collections.namedtuple(
        "Cluster", "graph bb_max bb_min center_of_mass comment", defaults = (None, None, None, None, ""))

    clusters.append(Cluster(center_of_mass = (1,1), comment = "(1,1)"))
    clusters.append(Cluster(center_of_mass = (2,2), comment = "(2,2)"))
    clusters.append(Cluster(center_of_mass = (10,5), comment = "(10,5)"))
    clusters.append(Cluster(center_of_mass = (50,50), comment = "(50,50)"))
    clusters.append(Cluster(center_of_mass = (51,51), comment = "(51,51)"))
    
    ward_merge(clusters)

    pass


def scale_up_export_report(wire):
    wirelength = 0
    output = []

    scale_factor = 2

    for x in range(wire.shape[0] - 1):
        for y in range(wire.shape[1] - 1):
            if wire[x][y] == 1:
                if wire[x+1][y] == 1:
                    if wire[x][y+1] == 1:
                        wirelength += 2*scale_factor - 1
                        #append(L)
                    else:
                        wirelength += scale_factor
                        #append(~)
                else:
                    if wire[x][y+1] == 1:
                        wirelength += scale_factor
                        #append(|)
                    else:
                        wirelength += 1
                        #append(.)

    print("Total wirelength: ", wirelength)

    return wirelength


if __name__ == "__main__":
    
    load = np.load('for_clustering_debug2.npz')
    # X, Y = load['X'], load['Y']
    design = load['X']
    wire = load['Y']
    
    start = time.time()
    
    postprocess_greedy(design, wire)
    
    print("Evaluation time [s] =", time.time() - start)

    scale_up_export_report(wire)

    show_map_slow(design, wire)
    #test_merging()
