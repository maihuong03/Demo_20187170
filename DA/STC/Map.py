import math
import numpy as np

class Map():
    def __init__(self):
        self.min_x = -1
        self.min_y = -1
        self.max_x = -1
        self.max_y = -1
        self.og_width = -1
        self.og_height = -1
        self.width = -1
        self.height = -1
        self.robot_len = 0.3

        self.vis = {}
        self.blocked = {}
        self.graph = {}

        self.block_num = 0
        # TODO: recorder now stores the number of block visited 
        # because depth() isn't working at this moment, 
        # should change back to depth() once it's fixed
        self.recorder = 1
    
    # shows a graph of borders(blocked) and spanning tree(vis)
    def show(self):
        import matplotlib.pyplot as plt
        blockedx = []
        blockedy = []
        for i in range(0, self.width+1):
            for j in range(0, self.height+1):
                if (self.blocked[(i,j)]):
                    # print (i,j)
                    blockedx.append(i * self.robot_len + self.min_x)
                    blockedy.append(j * self.robot_len + self.min_y)
        plt.plot(blockedx, blockedy, '.')

        visx = []
        visy = []
        for i in range(0, self.width+1):
            for j in range(0, self.height+1):
                if (self.vis[(i,j)]):
                    visx.append(i * self.robot_len + self.min_x)
                    visy.append(j * self.robot_len + self.min_y)
        plt.plot(visx, visy, '*')

        plt.show()


    # initialize most attributes in Map, including map.height, width, vis, blocked
    # @params [filename] is the .smap file we are reading in
    def initialize_map(self, filename):
        from MapReader import MapReader
        
        import sys
        import os

        map_reader = MapReader(filename)
        map_reader.run()

        map_x = map_reader.map_x
        map_y = map_reader.map_y

        self.min_x = math.floor(min(map_reader.map_x))
        self.max_x = math.ceil(max(map_reader.map_x))
        self.og_width = self.max_x - self.min_x
        self.min_y = math.floor(min(map_reader.map_y))
        self.max_y = math.ceil(max(map_reader.map_y))
        self.og_height = self.max_y - self.min_y

        self.width = math.floor(self.og_width / self.robot_len)
        self.height = math.floor(self.og_height / self.robot_len)

        # initialize all the blocks in range
        for i in range(1,self.width+1):
            for j in range(1,self.height+1):
                self.vis[(i,j)] = 0
                self.blocked[(i,j)] = 0

        # mark all blocks with obstacles as blocked
        for i in range(len(map_reader.map_x)):
            self.blocked[self.to_point_coor(map_reader.map_x[i], map_reader.map_y[i])] = 1

        # initializethe borders to avoid array out of bound 
        for i in range(-10,self.width+10):
            for j in range(-10, self.height+10):
                if(self.is_valid_point(i,j)):
                    self.vis[(i,j)] = 0
                    self.blocked[(i,j)] = 0
                else :
                    self.vis[(i,j)] = 0
                    self.blocked[(i,j)] = 1
        
        # and count valid block numbers
        for i in range(0,self.width+2, 2):
            for j in range(0, self.height+2, 2):
                if(self.is_valid_block(i,j)):
                    self.block_num = self.block_num + 1


    def to_point_coor(self, mapx, mapy):
        return ( 
            math.floor((mapx-self.min_x)/self.robot_len), 
            math.floor((mapy-self.min_y)/self.robot_len))


    # @return true when position (x,y) is within bound of the map and is not blocked
    def is_valid_point(self, x, y):
        if (x,y) in self.blocked:
            return (not self.blocked[(x,y)]) \
                and x>0 and x<self.width \
                and y>0 and y<self.height
        else :
            return False

    # @return true if all four points in this block are valid
    def is_valid_block(self, x, y):
        return x % 2 == 0 and y % 2 == 0 and \
            self.is_valid_point(x-1,y-1) and \
            self.is_valid_point(x-1,y) and \
            self.is_valid_point(x,y-1) and \
            self.is_valid_point(x,y)

class Block:
    def __init__(self, ix, iy):
        self.x = ix
        self.y = iy
        self.parent = None
        self.children = np.array([None, None, None, None])


class Point:
    def __init__(self,*args):
        if len(args)==0:
            self.x = -1
            self.y = -1
        elif len(args) == 2:
            self.x = args[0]
            self.y = args[1]
        self.last = None
        self.next = None

# gets the block that point (ptx, pty) belongs to
def point_to_block(ptx, pty):
    return ((ptx+1) //2 *2, (pty+1)//2 *2)