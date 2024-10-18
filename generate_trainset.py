import numpy as np
import networkx as nx
import random
import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
import time

import tqdm

import pickle

from networkx.algorithms import approximation

from functools import reduce

from threading import Thread

from scale_design import *

def show_map_slow(map1, map2 = None, threshold = 0.9, file = None):
    colors = ['r', 'g', 'b']

    height, width = map1.shape[:2]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.set_xlim(0, width*10)
    #ax.set_ylim(0, width*10)

    minx, miny = width*10, width*10
    maxx, maxy = 0, 0

    start = time.time()
    for x in range(width):
        for y in range(height):
            if map2 is not None:
                if type(map2[x][y]) is np.ndarray:
                    if map2[x][y][0] >= threshold and map1[x][y][1] >= 0.5:
                        maxx = max(maxx,x*10)
                        maxy = max(maxy,y*10)
                        miny = min(miny, y*10)
                        minx = min(minx, x*10)

                        value = map2[x][y][0]
                        ax.add_patch(patches.Rectangle(
                            (x*10, y*10), 10, 10, linewidth=0, facecolor='yellow'))

                    elif map2[x][y][0] >= threshold:
                        maxx = max(maxx, x*10)
                        maxy = max(maxy, y*10)
                        miny = min(miny, y*10)
                        minx = min(minx, x*10)

                        value = map2[x][y][0]
                        ax.add_patch(patches.Rectangle(
                            (x*10, y*10), 10, 10, linewidth=0, facecolor=(1.0 - value, 1.0 - value,1.0)))
                else:
                    if map2[x][y] >= threshold and map1[x][y][1] >= 0.5:
                        value = map2[x][y]
                        ax.add_patch(patches.Rectangle(
                            (x*10, y*10), 10, 10, linewidth=0, facecolor='yellow'))

                    elif map2[x][y] >= threshold:
                        value = map2[x][y]
                        ax.add_patch(patches.Rectangle(
                            (x*10, y*10), 10, 10, linewidth=0, facecolor=(1.0 - value, 1.0 - value,1.0)))

            if map1 is not None:
                

                if map1[x][y][0] >= 0.5:
                    ax.add_patch(patches.Rectangle((x*10, y*10), 10, 10, linewidth=0, facecolor='red'))
                    maxx = max(maxx,x*10)
                    maxy = max(maxy,y*10)
                    miny = min(miny, y*10)
                    minx = min(minx, x*10)
                if map1[x][y][1] >= 0.5:
                    ax.add_patch(patches.Rectangle((x*10, y*10), 10, 10, linewidth=5, facecolor='green'))
                    maxx = max(maxx,x*10)
                    maxy = max(maxy,y*10)
                    miny = min(miny, y*10)
                    minx = min(minx, x*10)
                    continue

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)       
    
    if file is None:
        plt.show()
    else:
        plt.savefig(file, bbox_inches='tight', dpi=300)
    print('Time: {:.2f}s'.format(time.time() - start))

class Room:

    def __init__(self, X, Y, left_pins = [], top_pins = [], right_route = [],  bottom_route = [], init = True,):
        self.X = X
        self.Y = Y
        self.init = init

    def check_left_neighbor(self, left_n):
        if left_n == None:
            return True
        
        #checking if left_n is empty route
        #if len(np.argwhere(left_n.Y == 1)) == 0:
        #    return True

        resolution = left_n.Y.shape

        #checking if left_n has
        route_last_column = list(
            filter(lambda cell: cell[0] == (
                resolution[0] - 1), np.argwhere(left_n.Y == 1)))
        
        route_first_column = list(
            filter(lambda cell: cell[0] == 0, np.argwhere(self.Y == 1)))


        pins_first_column = list(
            filter(lambda cell: cell[0] == 0, np.argwhere(self.X[:, :, 1] == 1)))


        pins_last_column = list(
                filter(lambda cell: cell[0] == (
                    resolution[0] - 1), np.argwhere(left_n.X[:,:,1] == 1)))
        
        ok = False
        
        for p in pins_first_column:
            if  p[1] in list( [ cell[1] for cell in route_last_column ] ):
                ok = True
                return True

        #overwise too

        for p in pins_last_column:
            if p[1] in list([cell[1] for cell in route_first_column]):
                ok = True
                return True
        
        return ok


    #this works good
    def check_top_neighbor(self, top_n):
        if top_n == None:
            return True

        #checking if left_n is empty route
        #if len(np.argwhere(top_n.Y == 1)) == 0:
        #    return True

        resolution = top_n.Y.shape

        #checking if left_n has
        route_last_row = list(
            filter(lambda cell: cell[1] == (
                resolution[1] - 1), np.argwhere(top_n.Y == 1)))

        route_first_row = list(
            filter(lambda cell: cell[1] == 0, np.argwhere(self.Y == 1)))

        pins_first_row = list(
            filter(lambda cell: cell[1] == 0, np.argwhere(self.X[:, :, 1] == 1)))

        pins_last_row = list(
            filter(lambda cell: cell[1] == (
                resolution[1] - 1), np.argwhere(top_n.X[:, :, 1] == 1)))

        ok = False

        for p in pins_first_row:
            if p[0] in list([cell[0] for cell in route_last_row]):
                ok = True
                return True

        #overwise too

        for p in pins_last_row:
            if p[0] in list([cell[0] for cell in route_first_row]):
                ok = True
                return True

        return ok

        

def generate_map(resolution=(32, 32), pins=5, connection_pins = 4,  obstacles=10, fraction=0.5, routed = True):
    m = np.zeros(resolution)
    
    G = nx.grid_2d_graph(resolution[0], resolution[1])

    for r in range(obstacles):
        obstacle_size = resolution[0]*resolution[1]*fraction // obstacles
        # random obstacle size from with mu = obstacle_size and 3sigma = obstacle_size
        obstacle_size = int( obstacle_size + np.random.randn() * (obstacle_size//3) )
        if obstacle_size <= 0: continue
            
        #random diagonal angle of rectangular obstacle
        angle = random.random() * 90.0
        
        height = math.sin(math.radians(angle))
        width = math.cos(math.radians(angle))
        #how much times height and width are smaller than needed
        scale = math.sqrt(obstacle_size / (height*width))
        height = int(height * scale)
        if height >= resolution[1] - 1: height = resolution[1] - 2
        width =  int(width * scale)
        if width >= resolution[0] - 1: width = resolution[0] - 2
        #now placing this rectangle on map
        x = random.randint(1, resolution[0] - width)
        y = random.randint(1, resolution[1] - height)
        
        #if height or width == 0: continue
        if height == 0 or width == 0:
            continue

        #removing nodes from G
        nodes_to_remove = reduce(lambda x,y: x+y, [[(i,j) for i in range (x, x+ width)] for j in range(y, y+height)] )
        G.remove_nodes_from( nodes_to_remove )

        #filling map with obstacle

        m[x:x+width, y:y+height] = 1

    #delete unconnected components
    components = list(nx.connected_components(G))
    components.sort(reverse = True , key = lambda x: len(x))
    for component in components[1:]:
        G.remove_nodes_from(component)

    empty_cells = list(components[0])        
    cells_with_pins = random.choices(empty_cells, k=pins)

    # random free cells in the last and column
    empty_last_column = list(
        filter( (lambda cell: cell[0] == (resolution[0] - 1)), empty_cells)
        )
    empty_first_column = list(filter(lambda cell: cell[0] == 0, empty_cells))
    
    empty_first_row = list(filter(lambda cell: cell[1] == 0, empty_cells))
    empty_last_row = list(filter(lambda cell: cell[1] == (resolution[1] - 1), empty_cells))

    service_pins = random.choices(empty_last_column + empty_first_column + empty_first_row + empty_last_row, k = connection_pins)

    cells_with_pins = cells_with_pins + service_pins

    empty_last_row = list(
        filter((lambda cell: cell[1] == (resolution[0] - 1)), empty_cells)
    )

    

    
    left_column_pins = list(filter(lambda pin: pin[0] == 0, cells_with_pins))
    top_row_pins = list(filter(lambda pin: pin[1] == 0, cells_with_pins))


    # #ok, obstacles settled now we need to add pins and make sure that pins may be connected
    # empty_cells = np.argwhere(m == 0)    
    # cells_with_pins = np.random.choice(len(empty_cells), pins)
    # cells_with_pins = empty_cells[cells_with_pins]

    rows, cols = zip(*cells_with_pins)
    #pins - map with pin lotaions, another channel
    pins_map = np.zeros(resolution)
    pins_map[rows, cols] = np.ones((pins + connection_pins))

    #next, we must build networkx graph
    #G = nx.Graph()
    #nodes are called by a integer number - for m[x][y] id=x*resolution[1]+y

    # #first, building a map of connections by passing 2 1-d convolutions:
    # #if both ajacent pixels are free [ ][ ] - there is a link
    # #if at least one of them is not free [x][ ] [][x] [x][x] - there is no link
    # for i in range(resolution[0]):
    #     for j in range(resolution[1]):
    #         if j < resolution[1] - 1:    
    #             if m[i][j] == 0 and m[i][j+1] == 0:
    #                 #adding edge from m[i][j] to m[i][j+1]
    #                 G.add_edge(i*resolution[1] + j, i*resolution[1]+ j + 1)
    #         if i < resolution[0] - 1:
    #             if m[i][j] == 0 and m[i+1][j] == 0:
    #                 #adding edge from m[i][j] to m[i+1][j]
    #                 G.add_edge(i*resolution[1] + j, (i+1)*resolution[1] + j)
    
    # # fine, now we have to make sure that all pins are connectible
    # reference_pin = None
    # for p in cells_with_pins:
    #     if reference_pin is None:
    #         reference_pin = p
    #         continue
    #     if nx.has_path(G, reference_pin[0]*resolution[1] + reference_pin[1], p[0]*resolution[1] + p[1]) == False:
    #         # this design is generated without path
    #         # we have to make path manually between p and reference_path
            
    #         #making the horizontal line from p to reference pin
    #         for dx in range(min(reference_pin[0], p[0]), max(reference_pin[0], p[0])):
    #             G.add_edge(dx*resolution[1] + p[1], (dx+1)*resolution[1] + p[1])
    #         #now making vertical line xplicitely from reference pin to p
    #         for dy in range(min(reference_pin[1], p[1]), max(reference_pin[1], p[1])):
    #             G.add_edge(dy*resolution[1] + reference_pin[1],
    #                        (dy+1)*resolution[1] + reference_pin[1])

    #     else:
    #         reference_pin = p


    

    #building a list of nodes to be connected
    nodes = cells_with_pins
    # nodes = [x*resolution[1] + y for (x,y) in cells_with_pins]
    
    #building a steiner tree    
    if routed is True:
        steiner = nx.algorithms.approximation.steinertree.steiner_tree(G, nodes)
        steiner_nodes = steiner

        

        #we have steiner nodes, now converting them to 2d map
        output = np.zeros(resolution)
        rows, cols = zip(*steiner_nodes)
        output[rows, cols] = np.ones((len(steiner_nodes)))

        #and finding cells with the path in last row/column

        route_last_column = list(filter(lambda cell: cell[0] == (resolution[0] - 1), steiner_nodes))
        route_last_row = list(
            filter(lambda cell: cell[1] == (resolution[1] - 1), steiner_nodes))


        # X - map and pins merged, Y - steiner map


        return np.stack((m, pins_map), axis = -1), output, left_column_pins, top_row_pins, route_last_column, route_last_row
    
    else:
        return np.stack((m, pins_map), axis=-1)




def check_room_to_be_appended(level, level_size, room_position, room):
    #if room_position is (0,x) - it is approved from the left side
    if room_position[0] > 0:
        left_neighbor = level[room_position[0]-1][room_position[1]]
        
        checked = False

        for p in [y for (x,y) in room.left_pins]:
            checked = False
            if p in [y for (x, y) in left_neighbor.right_route]:
                checked = True
                break
        
        if checked is False:
            return False
        
    #now checking upper neighbor:
    if room_position[1] > 0:
        top_neighbor = level[room_position[0]][room_position[1]-1]

        checked = False

        for p in [x for (x, y) in room.top_pins]:
            checked = False
            if p in [x for (x, y) in top_neighbor.bottom_route]:
                checked = True
                break

        if checked is False:
            return False
                
    return True

def merge_level(level):
    room_size = (level[0][0].X.shape[0], level[0][0].X.shape[1])
    map_size = (level[0][0].X.shape[0] * len(level),
                level[0][0].X.shape[1] * len(level[0]))
    X = np.zeros((map_size[0], map_size[1], 2))
    Y = np.zeros((map_size[0], map_size[1]))

    for x in range(len(level)):
        for y in range(len(level[0])):
            X[x*room_size[0]:(x+1)*room_size[1], y*room_size[1]              :(y+1)*room_size[1]] = level[x][y].X
            Y[x*room_size[0]:(x+1)*room_size[1], y*room_size[1]:(y+1)*room_size[1]] = level[x][y].Y

    return X, Y

def show_level(level):
    room_size = (level[0][0].X.shape[0], level[0][0].X.shape[1])
    map_size = (level[0][0].X.shape[0] * len(level), level[0][0].X.shape[1] * len(level[0]))
    X = np.zeros( (map_size[0], map_size[1], 2 ))
    Y = np.zeros( (map_size[0], map_size[1]) )
    
    for x in range(len(level)):
        for y in range(len(level[0])):
            X[x*room_size[0]:(x+1)*room_size[1], y*room_size[1]:(y+1)*room_size[1]] = level[x][y].X
            Y[x*room_size[0]:(x+1)*room_size[1], y*room_size[1]
                              :(y+1)*room_size[1]] = level[x][y].Y

    show_map_slow(X,Y)


    

 
def make_level(rooms, level_size):
    
    #random.shuffle(rooms)
    level = []

    #how do we deal with multi-tile rooms:
    #break rooms to 32x32 and making tiles, other than first tile to have only one possible neighbor
    for x in range(level_size[0]):
        line = []
        for y in range(level_size[1]):
            line.append(None)
        level.append(line)

    for x in range(level_size[0]):
        for y in range(level_size[1]):
            

            legal_rooms_1 = []
            legal_rooms_2 = []
            legal_default_1 = False
            legal_default_2 = False
            if x > 0:
                #check if other tile requires this tile to be specific
                if level[x-1][y] in rooms.neccesary_l:
                    level[x][y] = rooms.neccesary_l[level[x-1][y]]
                    print('Room {},{} added as neccesary l'.format(x, y))
                    continue

                legal_rooms_1 = set(rooms.valid_l_neighbors[level[x-1][y]])
                

            else:
                legal_rooms_1 = set(filter(lambda r: r.init == True, rooms.rooms))
                legal_default_1 = True
            
            

            if y > 0:
                #check if other tile requires this tile to be specific
                if level[x][y-1] in rooms.neccesary_t:
                    level[x][y] = rooms.neccesary_t[level[x][y-1]]
                    print('Room {},{} added as neccesary t'.format(x, y))
                    continue

                legal_rooms_2 = set(
                    rooms.valid_t_neighbors[level[x][y-1]])
                
                
            else:
                legal_rooms_2 = set(filter(lambda r: r.init == True, rooms.rooms))
                legal_default_2 = True
            
            legal_rooms = set([])

            if legal_default_1 and legal_default_2:
                legal_rooms = legal_rooms_1.union(legal_rooms_2)
            else:
                if legal_default_1 is True and legal_default_2 is False:
                    legal_rooms = legal_rooms_2
                    
                if legal_default_1 is False and legal_default_2 is True:
                    legal_rooms = legal_rooms_1
                
                if legal_default_1 is False and legal_default_2 is False:
                    legal_rooms = legal_rooms_1.union(legal_rooms_2)

            #legal_rooms = legal_rooms_1.union(legal_rooms_2)
            if len(legal_rooms) > 0:
                legal_rooms = list(legal_rooms)
                # room is weighted random from the allowed rooms list
                weights = [rooms.room_weights[r] + 5  for r in legal_rooms]
                s = sum(weights)
                for i in range(len(weights)):

                    weights[i] = weights[i]/s
                
                room = np.random.choice(legal_rooms, p = weights)
                # random.choice(list(legal_rooms))
                level[x][y] = room

                print('Room {},{} added'.format(x, y))
            else:         
                level[x][y] = rooms.rooms[0]
                print('No any good rooms, empty {},{} added'.format(x, y))
                

        
    return level

rooms = []

def split_room(r):

    rooms = []
    first = True
    for row_X, row_Y in zip(np.split(r.X, r.Y.shape[0] // MIN_TILE_X), np.split(r.Y, r.Y.shape[0] // MIN_TILE_Y)):
        line = []
        for cell_X, cell_Y in zip(np.split(row_X, r.Y.shape[1] // MIN_TILE_X, 1), np.split(row_Y, r.Y.shape[1] // MIN_TILE_Y, 1)):
            if first:
                line.append(Room(cell_X, cell_Y, [], [], [], [], True))
                first = False
            else:
                line.append(Room(cell_X, cell_Y, [],[],[],[], False))
        rooms.append(line)
    
    return rooms

class Rooms:
    def __init__(self, rooms):
        #self.rooms = rooms
        self.valid_l_neighbors = {}
        self.valid_t_neighbors = {}
        self.neccesary_l = {}
        self.neccesary_t = {}

        self.room_weights = {}

        self.rooms = []
        for r in rooms:
            if r.Y.shape == (MIN_TILE_X, MIN_TILE_Y):
                self.rooms.append(r)
            if r.Y.shape != (MIN_TILE_X, MIN_TILE_Y):
                weight = 0
                subrooms = split_room(r)
                size_x = len(subrooms)
                size_y = len(subrooms[0])
                for x in range(size_x):
                    for y in range(size_y):
                        if x < size_x - 1:
                            #only next piece of puzzle is a tile to be fit
                            self.neccesary_l[subrooms[x][y]] = subrooms[x+1][y]
                            self.valid_l_neighbors[subrooms[x][y]] = [
                                subrooms[x+1][y]]
                        else:
                            # if it is a last piece, same story as for single tile rooms
                            valid_horiz = []
                            for another in rooms:
                                if another.init and another.check_left_neighbor(subrooms[x][y]):
                                    valid_horiz.append(another)
                            self.valid_l_neighbors[subrooms[x]
                                                   [y]] = valid_horiz
                            weight += len(valid_horiz)

                        if y < size_y - 1:
                            self.neccesary_t[subrooms[x][y]] = subrooms[x][y+1]
                            self.valid_t_neighbors[subrooms[x][y]] = [
                                subrooms[x][y+1]]
                        else:
                            # if it is a last piece, same story as for single tile rooms
                            valid_vert = []
                            for another in rooms:
                                if another.init and another.check_top_neighbor(subrooms[x][y]):
                                    valid_vert.append(another)
                            self.valid_t_neighbors[subrooms[x][y]] = valid_vert
                            weight += len(valid_vert)

                        if subrooms[x][y] not in self.valid_t_neighbors:
                            self.valid_t_neighbors[subrooms[x][y]] = []

                        if subrooms[x][y] not in self.valid_l_neighbors:
                            self.valid_l_neighbors[subrooms[x][y]] = []

                        # #debug
                        # for v in self.valid_l_neighbors[subrooms[x][y]]:
                        #     assert v in self.valid_t_neighbors

                        # for v in self.valid_t_neighbors[subrooms[x][y]]:
                        #     assert v in self.valid_t_neighbors
                        # #

                        self.rooms.append(subrooms[x][y])
                self.room_weights[subrooms[0][0]] = weight


        for r in tqdm.tqdm(self.rooms):
            if r.Y.shape == (MIN_TILE_X, MIN_TILE_Y):
                if r in self.neccesary_l:
                    self.valid_l_neighbors[r] = self.neccesary_l[r]

                if r in self.neccesary_t:
                    self.valid_t_neighbors[r] = self.neccesary_t[r]

                pass
                valid_horiz = []
                valid_vert = []
                for another in self.rooms:
                    if another.init and another.check_left_neighbor(r):
                        valid_horiz.append(another)
                    if another.init and another.check_top_neighbor(r):
                        valid_vert.append(another)
                self.valid_l_neighbors[r] = valid_horiz
                self.valid_t_neighbors[r] = valid_vert
                if r not in self.room_weights: 
                    self.room_weights[r] = len(valid_horiz) + len (valid_vert)

                


                

            
                

        
        
        # for r in self.rooms:
        #     for v in self.valid_l_neighbors[r]:
        #                 assert v in self.valid_t_neighbors
        #     for v in self.valid_t_neighbors[r]:
        #                 assert v in self.valid_t_neighbors       
                     
        # #checking hashes:
        # for r in self.valid_l_neighbors.keys():
        #     if r not in self.rooms:
        #         print("L_NEIGHBORS KEY NOT IN ROOMS LIST!") 
        # for r in self.valid_t_neighbors.keys():
        #     if r not in self.rooms:
        #         print("T_NEIGHBORS KEY NOT IN ROOMS LIST!") 
        # for r in self.rooms:
        #     if r not in self.valid_l_neighbors.keys():
        #         print("ROOM NOT IN T_NEIGHBORS")

        # print("Dicts checked")


def generate_bunch_or_rooms(n, size, rooms, main_thread = False):
    if main_thread is False:
        for i in range(n):
            X, Y, l_pins, t_pins, r_cells, b_cells = generate_map(
                (size, size), random.randint(0, 4), random.randint(2, 8), random.randint(0, 15))
            room1 = Room(X, Y, l_pins, t_pins, r_cells, b_cells)
            rooms.append(room1)
    else:
        for i in tqdm.tqdm(range(n)):
            X, Y, l_pins, t_pins, r_cells, b_cells = generate_map(
                (size, size), random.randint(0, 4), random.randint(2, 8), random.randint(0, 15))
            room1 = Room(X, Y, l_pins, t_pins, r_cells, b_cells)
            rooms.append(room1)

def generate_small_and_large_tiles(n = 50):
    size = 32
    rooms = []

    room = []

    #6 cores
    num_threads = 2
    
    threads = []
    for t in range(num_threads - 1):
        t = Thread(target=generate_bunch_or_rooms, args=(n // num_threads, 32, rooms))
        t.start()
        threads.append(t)
    
    generate_bunch_or_rooms(n//num_threads, 32, rooms, True)
    for t in threads:
        t.join()


    service_room_X = np.zeros((size, size, 2))

    service_room_Y = np.zeros((size, size))
    
    service_room_X[(0, 0)] = 1
    service_room_X[(0, size-1)] = 1
    service_room_X[(size-1, 0)] = 1
    service_room_X[(size-1, size-1)] = 1

    for x in range(size):
        for y in range(size):
            if x == 0 or x == size-1 or y == 0 or y == size-1:
                service_room_Y[x, y, 1] = 1

    empty_room_X = np.zeros((size, size, 2))
    empty_room_Y = np.zeros((size, size))
    empty_room = Room(empty_room_X, empty_room_Y, [], [], [], [])
    
    size = 64

    n = n//2

    rooms = [empty_room, ] + rooms

    rooms.append(Room(service_room_X, service_room_Y, [], [], [], []))

    threads = []
    for t in range(num_threads - 1):
        t = Thread(target=generate_bunch_or_rooms,
                   args=(n // num_threads, 64, rooms))
        t.start()
        threads.append(t)

    generate_bunch_or_rooms(n//num_threads, 64, rooms, True)
    for t in threads:
        t.join()


    rooms = Rooms(rooms)

    with open('rooms_object_different_{}.pickle'.format(random.randint(0, 1000)), 'wb') as f:
        pickle.dump(rooms, f)

def generate_rooms(n, size = 32, service_room = False, sizes = []):

    rooms = []

    

    for i in tqdm.tqdm(range(n)):
        if len(sizes) > 0:
            current_size = random.choice(sizes)
            while True:
                try:
                    X, Y, l_pins, t_pins, r_cells, b_cells = generate_map( (current_size[0], current_size[1]), random.randint(0, 4), random.randint(2, 8), random.randint(0, 15)  )
                    break
                except ValueError:
                    continue
        else:    
            while True:
                try:
                    X, Y, l_pins, t_pins, r_cells, b_cells = generate_map((size,size), random.randint(0,4), random.randint(2,8), random.randint(0,15))
                    break
                except ValueError:
                    continue
                
        room1 = Room(X, Y, l_pins, t_pins, r_cells, b_cells)
        rooms.append(room1)
        
        if random.randint(0, 3) != 0:
            scaledX, scaledY = scale(X, Y)
            if random.randint(0,2) != 0:
                scaledX, scaledY = scale(scaledX, scaledY)
            room_scaled = Room(scaledX, scaledY, [],[],[],[])
            rooms.append(room_scaled)


    #EMPTY ROOMS EXPERIMENT
    for i in tqdm.tqdm(range(n//4)):
        if len(sizes) > 0:
            current_size = random.choice(sizes)
            X, Y, l_pins, t_pins, r_cells, b_cells = generate_map( (current_size[0], current_size[1]), random.randint(0, 4), random.randint(2, 8), 0, 0.0  )
        else:    
            X, Y, l_pins, t_pins, r_cells, b_cells = generate_map((size,size), random.randint(0,4), random.randint(2,8), random.randint(0,15))
        room1 = Room(X, Y, l_pins, t_pins, r_cells, b_cells)
        rooms.append(room1)

        if random.randint(0, 1) == 0:
            scaledX, scaledY = scale(X, Y)
            room_scaled = Room(scaledX, scaledY, [], [], [], [])
            rooms.append(room_scaled)

    if service_room:
        service_room_X = np.zeros((size, size, 2))

        service_room_X = np.zeros((size, size))
        service_room_X[0, 0, 1] = 1
        service_room_X[0, size-1, 1] = 1
        service_room_X[size-1, 0, 1] = 1
        service_room_X[size-1, size-1, 1] = 1

        for x in range(size):
            for y in range(size):
                if x == 0 or x == size-1 or y == 0 or y == size-1:
                    service_room_Y[x, y, 1] = 1

        rooms.append(Room(service_room_X, service_room_Y, [], [], [], []))


    empty_room_X = np.zeros((size, size, 2))
    empty_room_Y = np.zeros((size, size))

    empty_room = Room(empty_room_X, empty_room_Y, [], [], [], [])

    rooms = [empty_room, ] + rooms

    
    rooms = Rooms(rooms)

    

    with open('rooms_object_{}.pickle'.format(random.randint(0, 1000)), 'wb') as f:
        pickle.dump(rooms, f)

    return rooms


def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, cmax, rmin, rmax


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


MIN_TILE_X = 8
MIN_TILE_Y = 8

class DesignGenerator():

    def __init__(self, rooms, batch_size = 10, target_resolution = 512, content_resolution = 256, levels_batch_size = 15, repeat = False):
        self.rooms = rooms
        self.levels_batch_size = levels_batch_size
        self.levels = []
        self.levels_modified_X = []
        self.levels_modified_Y = []
        self.number_of_shifts = 5
        self.target_resolution = target_resolution
        self.content_resolution = content_resolution 
        self.batch_size = batch_size
        self.repeat = repeat
        #self.on_epoch_end()

    def clear(self):
        self.levels = []
        self.levels_modified_X = []
        self.levels_modified_Y = []

    def on_epoch_end(self):
        
        if len(self.levels) != 0 and self.repeat is True:
            return

        self.levels = []
        self.levels_modified_X = []
        self.levels_modified_Y = []


        for i in range(self.levels_batch_size): 
            
            X, Y = [], [] 

            while True:
                try:
                    X, Y = merge_level(make_level(
                         self.rooms, (self.content_resolution // MIN_TILE_X, self.content_resolution // MIN_TILE_Y)))
                    break
                except TypeError:
                    print("Generation was faulty.  Try again...")

            
            
            self.levels.append( (X, Y) )
            
            cmin, cmax, rmin, rmax = bbox2(Y)
            content_X = X[cmin:cmax, rmin:rmax]
            content_Y = Y[cmin:cmax, rmin:rmax]
            content_Y = np.expand_dims(content_Y, axis = -1) 
            
            max_random_x = X.shape[0] - content_X.shape[0]
            max_random_y = X.shape[1] - content_X.shape[1]

            for j in range(self.number_of_shifts):
                shifted_X = np.zeros(
                    (self.target_resolution, self.target_resolution, 2))
                shifted_Y = np.zeros((self.target_resolution, self.target_resolution, 1))
                random_x = random.randint(0, max_random_x)
                random_y = random.randint(0, max_random_x)
                
                shifted_X[random_x:random_x + content_X.shape[0],
                          random_y:random_y + content_X.shape[1]] = content_X
                
                shifted_Y[random_x:random_x + content_X.shape[0],
                          random_y:random_y + content_X.shape[1]] = content_Y
                
                self.levels_modified_X.append(shifted_X)
                self.levels_modified_Y.append(shifted_Y)

                for k in range(1,4):
                    shifted_X = np.rot90(shifted_X, k=k)
                    shifted_Y = np.rot90(shifted_Y, k=k)
                    self.levels_modified_X.append(shifted_X)
                    self.levels_modified_Y.append(shifted_Y)
                    
                    #show_map_slow(shifted_X,shifted_Y)
                    pass
        
        shuffler = np.random.permutation(len(self.levels_modified_X))
       
        self.levels_modified_X = np.array(self.levels_modified_X)
        self.levels_modified_Y = np.array(self.levels_modified_Y)

        self.levels_modified_X, self.levels_modified_Y = unison_shuffled_copies(
            self.levels_modified_X, self.levels_modified_Y)

        
            
    def __len__(self):
        return len(self.levels_modified_X) // self.batch_size
            
    def __getitem__(self, index):
        X = []
        Y = []
        return self.levels_modified_X[index * self.batch_size: self.batch_size:(index + 1) * self.batch_size], self.levels_modified_Y[index * self.batch_size: self.batch_size: (index + 1) * self.batch_size]


def patch_empty_room():
    rooms = []
    with open('rooms_object_different_299.pickle', 'rb') as f:
        rooms = pickle.load(f)
    
    size = 32

    service_room_X = np.zeros((32,32,2))
    service_room_Y = np.zeros((32,32))
    

    service_room_X[(0, 0, 1)] = 1
    service_room_X[(0, size-1, 1)] = 1
    service_room_X[(size-1, 0, 1)] = 1
    service_room_X[(size-1, size-1, 1)] = 1

    for x in range(size):
        for y in range(size):
            if x == 0 or x == size-1 or y == 0 or y == size-1:
                service_room_Y[x, y] = 1

    service_room = Room(service_room_X, service_room_Y, [], [], [], [])
    
    bad_room = rooms.rooms[250]
    bad_rooms_valid_l_neighbors = rooms.valid_l_neighbors[bad_room]
    bad_rooms_valid_t_neighbors = rooms.valid_t_neighbors[bad_room]

    rooms.rooms[250].X = service_room_X
    rooms.rooms[250].Y = service_room_Y
    rooms.valid_l_neighbors[rooms.rooms[250]] = bad_rooms_valid_l_neighbors
    rooms.valid_t_neighbors[rooms.rooms[250]] = bad_rooms_valid_t_neighbors
    rooms.neccesary_l[rooms.rooms[250]] = []
    rooms.neccesary_t[rooms.rooms[250]] = []
    

    with open('rooms_object_300.pickle', 'wb') as f:
        pickle.dump(rooms, f)



#TESTING!!!!
if __name__ == '__main__':
    rooms = generate_rooms(600, service_room = False, sizes = ((8,8), (16,16), (32,32), (16,64), (64,16), (128,32), (32,128), (64,64), )  )
    
    #gen = DesignGenerator(rooms)
    #gen.on_epoch_end()
    #X, Y = gen.__getitem__(0)
    #show_map_slow(X[0], Y[0])

#patch_empty_room()

# rooms = []

# with open('rooms_object_300.pickle', 'rb') as f:
#     rooms = pickle.load(f)
    
# gen = DesignGenerator(rooms)
# gen.on_epoch_end()
# gen.__getitem__(5)
   #generate_small_and_large_tiles(n=400)





#### this works! ####
#generate_rooms(200, service_room = False, sizes = (32,64,))

# #with open('rooms_object_large_709.pickle', 'rb') as f:
# with open('rooms_object_240.pickle', 'rb') as f:

#     # The protocol version used is detected automatically, so we do not
#     # have to specify it.
#     rooms = pickle.load(f)

#     for r in rooms.rooms:
#         r.init = True
#     rooms.neccesary_t = {}
#     rooms.neccesary_l = {}
    
# rooms.rooms = rooms.rooms[:100]

# for r in rooms.rooms[50:]:
#      r.init = True
#      r.X = np.repeat(r.X, 2, axis = 0)
#      r.X = np.repeat(r.X, 2, axis = 1)

#      r.Y = np.repeat(r.Y, 2, axis = 0)
#      r.Y = np.repeat(r.Y, 2, axis = 1)
    
# rooms = Rooms(rooms.rooms)

# # with open('rooms_object_large_{}.pickle'.format(random.randint(0, 1000)), 'wb') as f:
# #         pickle.dump(rooms, f)

# start = time.time()

# level = make_level(rooms, level_size = (8,8))

# print('Level generation time is {:.2f}s'.format(time.time() - start))
# #level = [[room1, room2],[room3,room4]]
# show_level(level)


# #show_map_slow(X,Y)
    


