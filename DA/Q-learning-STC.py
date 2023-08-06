import os
import sys
import math
import cv2 

import random
from random import randint as r
import numpy as np
import matplotlib.pyplot as plt

from numba import cuda

# config
do_animation = True
target = None
path_value = -0.1 # for coverd path

gamma = 0.9
epsilon = 0.5

reward = None
obstacles = []

Q = None
actions = {"up": 0, "down": 1, "left": 2, "right": 3}
states = {}
sumOfRewards = []
episodeViaStep = []
step = 0
temp = 0
highestReward = 0
highestReward_counter = 0
shortest_path = []
repeatLimit = 50

class SpiralSpanningTreeCoveragePlanner:
    def __init__(self, occ_map):
        self.origin_map_height = occ_map.shape[0]
        self.origin_map_width = occ_map.shape[1]

        # original map resolution must be even
        if self.origin_map_height % 2 == 1 or self.origin_map_width % 2 == 1:
            sys.exit('original map width/height must be even \
                in grayscale .png format')

        self.occ_map = occ_map
        self.merged_map_height = self.origin_map_height // 2
        self.merged_map_width = self.origin_map_width // 2

        self.edge = []

    def plan(self, start):
        visit_times = np.zeros(
            (self.merged_map_height, self.merged_map_width), dtype=int)
        visit_times[start[0]][start[1]] = 1

        # generate route by
        # recusively call perform_spanning_tree_coverage() from start node
        route = []
        self.perform_spanning_tree_coverage(start, visit_times, route)

        path = []
        # generate path from route
        for idx in range(len(route)-1):
            dp = abs(route[idx][0] - route[idx+1][0]) + \
                abs(route[idx][1] - route[idx+1][1])
            if dp == 0:
                # special handle for round-trip path
                path.append(self.get_round_trip_path(route[idx-1], route[idx]))
            elif dp == 1:
                path.append(self.move(route[idx], route[idx+1]))
            elif dp == 2:
                # special handle for non-adjacent route nodes
                mid_node = self.get_intermediate_node(route[idx], route[idx+1])
                path.append(self.move(route[idx], mid_node))
                path.append(self.move(mid_node, route[idx+1]))
            else:
                sys.exit('adjacent path node distance larger than 2')

        return self.edge, route, path

    def perform_spanning_tree_coverage(self, current_node, visit_times, route):
        def is_valid_node(i, j):
            is_i_valid_bounded = 0 <= i < self.merged_map_height
            is_j_valid_bounded = 0 <= j < self.merged_map_width
            if is_i_valid_bounded and is_j_valid_bounded:
                # free only when the 4 sub-cells are all free
                return bool(
                    self.occ_map[2*i][2*j]
                    and self.occ_map[2*i+1][2*j]
                    and self.occ_map[2*i][2*j+1]
                    and self.occ_map[2*i+1][2*j+1])

            return False

        # counter-clockwise neighbor finding order
        order = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        found = False
        route.append(current_node)
        for inc in order:
            ni, nj = current_node[0] + inc[0], current_node[1] + inc[1]
            if is_valid_node(ni, nj) and visit_times[ni][nj] == 0:
                neighbor_node = (ni, nj)
                self.edge.append((current_node, neighbor_node))
                found = True
                visit_times[ni][nj] += 1
                self.perform_spanning_tree_coverage(
                    neighbor_node, visit_times, route)

        # backtrace route from node with neighbors all visited
        # to first node with unvisited neighbor
        if not found:
            has_node_with_unvisited_ngb = False
            for node in reversed(route):
                # drop nodes that have been visited twice
                if visit_times[node[0]][node[1]] == 2:
                    continue

                visit_times[node[0]][node[1]] += 1
                route.append(node)

                for inc in order:
                    ni, nj = node[0] + inc[0], node[1] + inc[1]
                    if is_valid_node(ni, nj) and visit_times[ni][nj] == 0:
                        has_node_with_unvisited_ngb = True
                        break

                if has_node_with_unvisited_ngb:
                    break

        return route

    def move(self, p, q):
        direction = self.get_vector_direction(p, q)
        # move east
        if direction == 'E':
            p = self.get_sub_node(p, 'SE')
            q = self.get_sub_node(q, 'SW')
        # move west
        elif direction == 'W':
            p = self.get_sub_node(p, 'NW')
            q = self.get_sub_node(q, 'NE')
        # move south
        elif direction == 'S':
            p = self.get_sub_node(p, 'SW')
            q = self.get_sub_node(q, 'NW')
        # move north
        elif direction == 'N':
            p = self.get_sub_node(p, 'NE')
            q = self.get_sub_node(q, 'SE')
        else:
            sys.exit('move direction error...')
        return [p, q]

    def get_round_trip_path(self, last, pivot):
        direction = self.get_vector_direction(last, pivot)
        if direction == 'E':
            return [self.get_sub_node(pivot, 'SE'),
                    self.get_sub_node(pivot, 'NE')]
        elif direction == 'S':
            return [self.get_sub_node(pivot, 'SW'),
                    self.get_sub_node(pivot, 'SE')]
        elif direction == 'W':
            return [self.get_sub_node(pivot, 'NW'),
                    self.get_sub_node(pivot, 'SW')]
        elif direction == 'N':
            return [self.get_sub_node(pivot, 'NE'),
                    self.get_sub_node(pivot, 'NW')]
        else:
            sys.exit('get_round_trip_path: last->pivot direction error.')

    def get_vector_direction(self, p, q):
        # east
        if p[0] == q[0] and p[1] < q[1]:
            return 'E'
        # west
        elif p[0] == q[0] and p[1] > q[1]:
            return 'W'
        # south
        elif p[0] < q[0] and p[1] == q[1]:
            return 'S'
        # north
        elif p[0] > q[0] and p[1] == q[1]:
            return 'N'
        else:
            sys.exit('get_vector_direction: Only E/W/S/N direction supported.')

    def get_sub_node(self, node, direction):
        if direction == 'SE':
            return [2*node[0]+1, 2*node[1]+1]
        elif direction == 'SW':
            return [2*node[0]+1, 2*node[1]]
        elif direction == 'NE':
            return [2*node[0], 2*node[1]+1]
        elif direction == 'NW':
            return [2*node[0], 2*node[1]]
        else:
            sys.exit('get_sub_node: sub-node direction error.')

    def get_interpolated_path(self, p, q):
        # direction p->q: southwest / northeast
        if (p[0] < q[0]) ^ (p[1] < q[1]):
            ipx = [p[0], p[0], q[0]]
            ipy = [p[1], q[1], q[1]]
        # direction p->q: southeast / northwest
        else:
            ipx = [p[0], q[0], q[0]]
            ipy = [p[1], p[1], q[1]]
        return ipx, ipy

    def get_intermediate_node(self, p, q):
        p_ngb, q_ngb = set(), set()

        for m, n in self.edge:
            if m == p:
                p_ngb.add(n)
            if n == p:
                p_ngb.add(m)
            if m == q:
                q_ngb.add(n)
            if n == q:
                q_ngb.add(m)

        itsc = p_ngb.intersection(q_ngb)
        if len(itsc) == 0:
            sys.exit('get_intermediate_node: \
                 no intermediate node between', p, q)
        elif len(itsc) == 1:
            return list(itsc)[0]
        else:
            sys.exit('get_intermediate_node: \
                more than 1 intermediate node between', p, q)

    def visualize_path(self, edge, path, start):
        def coord_transform(p):
            return [2*p[1] + 0.5, 2*p[0] + 0.5]

        if do_animation:
            last = path[0][0]
            trajectory = [[last[1]], [last[0]]]
            for p, q in path:
                distance = math.hypot(p[0]-last[0], p[1]-last[1])
                if distance <= 1.0:
                    trajectory[0].append(p[1])
                    trajectory[1].append(p[0])
                else:
                    ipx, ipy = self.get_interpolated_path(last, p)
                    trajectory[0].extend(ipy)
                    trajectory[1].extend(ipx)

                last = q

            trajectory[0].append(last[1])
            trajectory[1].append(last[0])

            for idx, state in enumerate(np.transpose(trajectory)):
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

                # draw spanning tree
                plt.imshow(self.occ_map, 'gray')
                for p, q in edge:
                    p = coord_transform(p)
                    q = coord_transform(q)
                    plt.plot([p[0], q[0]], [p[1], q[1]], '-oc')
                sx, sy = coord_transform(start)
                plt.plot([sx], [sy], 'pr', markersize=10)

                # draw move path
                plt.plot(trajectory[0][:idx+1], trajectory[1][:idx+1], '-k')
                plt.plot(state[0], state[1], 'or')
                plt.axis('equal')
                plt.grid(True)
                plt.pause(0.01)

        else:
            # draw spanning tree
            plt.imshow(self.occ_map, 'gray')
            for p, q in edge:
                p = coord_transform(p)
                q = coord_transform(q)
                plt.plot([p[0], q[0]], [p[1], q[1]], '-oc')
            sx, sy = coord_transform(start)
            plt.plot([sx], [sy], 'pr', markersize=10)

            # draw move path
            last = path[0][0]
            for p, q in path:
                distance = math.hypot(p[0]-last[0], p[1]-last[1])
                if distance == 1.0:
                    plt.plot([last[1], p[1]], [last[0], p[0]], '-k')
                else:
                    ipx, ipy = self.get_interpolated_path(last, p)
                    plt.plot(ipy, ipx, '-k')
                plt.arrow(p[1], p[0], q[1]-p[1], q[0]-p[0], head_width=0.2)
                last = q

            plt.show()

def settings(route):
    global target, states, n, path_value, Q, reward
    Q = np.zeros((n**2, 4))
    reward = np.zeros((n, n))
    reward[target[0], target[1]] = 5

    for i in range(n):
        for j in range(n):
            if (i, j) not in route:
                reward[i, j] = -5
            elif (i, j) in route and reward[i, j] == 0:
                reward[i, j] = path_value 
    k = 0
    for i in range(n):
        for j in range(n):
            states[(i,j)] = k
            k += 1

cuda.jit()

def select_action(current_state):
    global current_pos, epsilon, actions
    possible_actions = []
    if np.random.uniform() <= epsilon:
        if current_pos[1] != 0:
            possible_actions.append("left")
        if current_pos[1] != n - 1:
            possible_actions.append("right")
        if current_pos[0] != 0:
            possible_actions.append("up")
        if current_pos[0] != n - 1:
            possible_actions.append("down")
        action = actions[possible_actions[r(0, len(possible_actions) - 1)]]
    else:
        m = np.min(Q[current_state])
        if current_pos[0] != 0:
            possible_actions.append(Q[current_state, 0])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != n - 1:
            possible_actions.append(Q[current_state, 1])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != 0:
            possible_actions.append(Q[current_state, 2])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != n - 1:
            possible_actions.append(Q[current_state, 3])
        else:
            possible_actions.append(m - 100)

        action = random.choice(
            [i for i, a in enumerate(possible_actions)
             if a == max(possible_actions)]
        )
        return action 

cuda.jit()

def episode():
    global current_pos, epsilon, agent_starting_position, temp, sumOfRewards, highestReward, highestReward_counter, shortest_path, episodeViaStep, step, repeatLimit
    current_state = states[(current_pos[0], current_pos[1])]
    action = select_action(current_state)

    if action == 0:
        current_pos[0] -= 1
    elif action == 1:
        current_pos[0] += 1
    elif action == 2:
        current_pos[1] -= 1
    elif action == 3:
        current_pos[1] += 1

    new_state = states[(current_pos[0], current_pos[1])]
    if new_state not in obstacles:
        Q[current_state, action] = reward[current_pos[0],
                                          current_pos[1]] + gamma * np.max(Q[new_state])
        temp += reward[current_pos[0], current_pos[1]]
        step += 1
        if highestReward_counter >= (repeatLimit*4/5):
            pos = [current_pos[0], current_pos[1]]
            shortest_path.append(pos)
    else:
        Q[current_state, action] = reward[current_pos[0],
                                          current_pos[1]] + gamma * np.max(Q[new_state])
        temp += reward[current_pos[0], current_pos[1]]
        if temp > highestReward:
            highestReward = temp
        if temp == highestReward:
            highestReward_counter += 1
        sumOfRewards.append(temp)
        episodeViaStep.append(step)
        step = 0
        temp = 0
        if highestReward_counter < repeatLimit:
            shortest_path = []
        current_pos = [agent_starting_position[0], agent_starting_position[1]]
        epsilon -= 1e-4

cuda.jit()

dir_path = os.path.dirname(os.path.realpath(__file__))
img = cv2.imread(os.path.join(dir_path, 'map', 'test_2.png'), cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, dsize=(40, 40))
n = img.shape[0]
agent_starting_position = (10, 0)
current_pos = [agent_starting_position[0], agent_starting_position[1]]

STC_planner = SpiralSpanningTreeCoveragePlanner(img)
edge, route, path = STC_planner.plan(agent_starting_position)
route = list(set(route)) # get unique point
target = route[r(0, len(route)-1)]
settings(route)
while True:
    print(highestReward_counter)
    if highestReward_counter < repeatLimit:
        episode()
    else:
        # draw shortest path
        print(shortest_path)
        break
# print(shortest_path)
# STC_planner.visualize_path(edge, path, agent_starting_position)