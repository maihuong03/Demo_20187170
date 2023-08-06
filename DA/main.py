import numpy as np
import pygame
from time import time, sleep
import matplotlib.pyplot as plt
from random import randint as r
import random
import pickle
import sys
from PyQt5 import QtWidgets
from numba import jit, cuda

"""
import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

do_animation = True


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
        plan

        performing Spiral Spanning Tree Coverage path planning

        :param start: the start node of Spiral Spanning Tree Coverage
        

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
        perform_spanning_tree_coverage

        recursive function for function <plan>

        :param current_node: current node

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


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img = plt.imread(os.path.join(dir_path, 'map_stc', 'test_3.png'))
    STC_planner = SpiralSpanningTreeCoveragePlanner(img)
    start = (10, 0)
    edge, route, path = STC_planner.plan(start)
    STC_planner.visualize_path(edge, path, start)


if __name__ == "__main__":
    main()
"""
agent_starting_position = [2, 2]
n = 10
target = [5, 5]
penalities = 20
path_value = -0.1

gamma = 0.9
epsilon = 0.5
csize = 15
scrx = n * csize
scry = n * csize
background = (51, 51, 51)
colors = [
    (51, 51, 51),  # gri
    (255, 0, 0),  # kırmızı
    (0, 255, 0),  # yeşil
    (143, 255, 240),  # turkuaz
]

reward = np.zeros((n, n))
obstacles = []

Q = np.zeros((n ** 2, 4))
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


class QLearningSettings(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(QLearningSettings, self).__init__(parent)

        layout = QtWidgets.QFormLayout()

        self.btn = QtWidgets.QPushButton("Matrix Size")
        self.btn.clicked.connect(self.matrixSize)
        self.le = QtWidgets.QLineEdit()
        self.le.setPlaceholderText("15")
        layout.addRow(self.btn, self.le)

        self.btn1 = QtWidgets.QPushButton("Agent Starting Position")
        self.btn1.clicked.connect(self.starting_position)
        self.le1 = QtWidgets.QLineEdit()
        self.le1.setPlaceholderText("5,5")
        layout.addRow(self.btn1, self.le1)

        self.btn2 = QtWidgets.QPushButton("Target Position")
        self.btn2.clicked.connect(self.target_position)
        self.le2 = QtWidgets.QLineEdit()
        self.le2.setPlaceholderText("10,10")
        layout.addRow(self.btn2, self.le2)

        self.btn3 = QtWidgets.QPushButton("Obstacle Percentage")
        self.btn3.clicked.connect(self.obstacle_percentage)
        self.le3 = QtWidgets.QLineEdit()
        self.le3.setPlaceholderText("40")
        layout.addRow(self.btn3, self.le3)

        self.btn4 = QtWidgets.QPushButton("Epsilon Value")
        self.btn4.clicked.connect(self.epsilon_value)
        self.le4 = QtWidgets.QLineEdit()
        self.le4.setPlaceholderText("0.50")
        layout.addRow(self.btn4, self.le4)

        self.btn5 = QtWidgets.QPushButton("Path Value")
        self.btn5.clicked.connect(self.path_value)
        self.le5 = QtWidgets.QLineEdit()
        self.le5.setPlaceholderText("-0.1")
        layout.addRow(self.btn5, self.le5)

        self.btn3 = QtWidgets.QPushButton("OK")
        self.btn3.clicked.connect(self.ex)
        layout.addRow(self.btn3)

        self.setLayout(layout)
        self.setWindowTitle("Q Learning Settings")

    def matrixSize(self):
        num, ok = QtWidgets.QInputDialog.getInt(
            self, "Matrix Size", "Enter Matrix Size:")

        if ok:
            self.le.setText(str(num))

    def starting_position(self):

        text, ok = QtWidgets.QInputDialog.getText(
            self, 'Agent Starting Position', 'Enter Agent Starting Position:')

        if ok:
            self.le1.setText(str(text))

    def target_position(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self, 'Target Position', 'Enter Target Position:')

        if ok:
            self.le2.setText(str(text))

    def obstacle_percentage(self):
        num, ok = QtWidgets.QInputDialog.getInt(
            self, "Obstacle Percentage", "Enter Obstacle Percentage:")

        if ok:
            self.le3.setText(str(num))

    def epsilon_value(self):
        num, ok = QtWidgets.QInputDialog.getInt(
            self, "Epsilon Value", "Enter Epsilon Value:")

        if ok:
            self.le4.setText(str(num))

    def path_value(self):
        num, ok = QtWidgets.QInputDialog.getInt(
            self, "Path Value", "Enter Path Value:")

        if ok:
            self.le5.setText(str(num))

    def ex(self):
        global agent_starting_position, n, target, penalities, path_value, epsilon
        n = int(self.le.text()) if self.le.text() != "" else 15
        agent_starting_position = list(
            map(int, (str(self.le1.text())).split(","))) if self.le1.text() != "" else [5, 5]
        target = list(map(int, (str(self.le2.text())).split(","))
                      ) if self.le2.text() != "" else [10, 10]
        penalities = int((n*n*int(self.le3.text())/100)
                         ) if self.le3.text() != "" else 40
        epsilon = float(self.le4.text()) if self.le4.text() != "" else 0.3
        path_value = float(self.le5.text()) if self.le5.text() != "" else -0.1
        self.close()


def settingsWindow():
    app = QtWidgets.QApplication(sys.argv)
    ex = QLearningSettings()
    ex.show()
    app.exec_()


settingsWindow()


def settings():
    global reward, penalities, target, obstacles, agent_starting_position, n, path_value, Q, scrx, scry, csize

    scrx = n * csize
    scry = n * csize
    Q = np.zeros((n ** 2, 4))
    reward = np.zeros((n, n))
    reward[target[0], target[1]] = 5

    while penalities != 0:
        i = r(0, n - 1)
        j = r(0, n - 1)
        if reward[i, j] == 0 and [i, j] != agent_starting_position and [i, j] != target:
            reward[i, j] = -5
            penalities -= 1
            obstacles.append(n * i + j)

    obstacles.append(n * target[0] + target[1])

    for i in range(n):
        for j in range(n):
            if reward[i, j] == 0:
                reward[i, j] = path_value

    k = 0
    for i in range(n):
        for j in range(n):
            states[(i, j)] = k
            k += 1


cuda.jit()


def layout():
    for i in range(n):
        for j in range(n):
            pygame.draw.rect(
                screen,
                (255, 255, 255),
                (j * csize, i * csize, (j * csize) + csize, (i * csize) + csize),
                0,
            )

            pygame.draw.rect(
                screen,
                colors[0],
                (
                    (j * csize) + 3,
                    (i * csize) + 3,
                    (j * csize) + 11,
                    (i * csize) + 11,
                ),
                0,
            )

            if reward[i, j] == -5:
                pygame.draw.rect(
                    screen,
                    colors[1],
                    (
                        (j * csize) + 3,
                        (i * csize) + 3,
                        (j * csize) + 11,
                        (i * csize) + 11,
                    ),
                    0,
                )
            if reward[i, j] == 5:
                pygame.draw.rect(
                    screen,
                    colors[2],
                    (
                        (j * csize) + 3,
                        (i * csize) + 3,
                        (j * csize) + 11,
                        (i * csize) + 11,
                    ),
                    0,
                )


cuda.jit()


def select_action(current_state):
    global current_pos, epsilon
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


def draw_shortest_path():
    global n, shortest_path
    for i in range(n):
        for j in range(n):
            pygame.draw.rect(
                screen,
                (255, 255, 255),
                (j * csize, i * csize, (j * csize) + csize, (i * csize) + csize),
                0,
            )

            pygame.draw.rect(
                screen,
                colors[0],
                (
                    (j * csize) + 3,
                    (i * csize) + 3,
                    (j * csize) + 11,
                    (i * csize) + 11,
                ),
                0,
            )

            if reward[i, j] == -5:
                pygame.draw.rect(
                    screen,
                    colors[1],
                    (
                        (j * csize) + 3,
                        (i * csize) + 3,
                        (j * csize) + 11,
                        (i * csize) + 11,
                    ),
                    0,
                )
            if [i, j] == agent_starting_position:
                pygame.draw.rect(
                    screen,
                    (25, 129, 230),
                    (
                        (j * csize) + 3,
                        (i * csize) + 3,
                        (j * csize) + 11,
                        (i * csize) + 11,
                    ),
                    0,
                )
            if reward[i, j] == 5:
                pygame.draw.rect(
                    screen,
                    colors[2],
                    (
                        (j * csize) + 3,
                        (i * csize) + 3,
                        (j * csize) + 11,
                        (i * csize) + 11,
                    ),
                    0,
                )

            if [i, j] in shortest_path:
                pygame.draw.rect(
                    screen,
                    colors[3],
                    (
                        (j * csize) + 3,
                        (i * csize) + 3,
                        (j * csize) + 11,
                        (i * csize) + 11,
                    ),
                    0,
                )

    pygame.display.flip()

    plt.subplot(1, 2, 1)
    plt.plot(sumOfRewards)
    plt.xlabel("Episodes")
    plt.ylabel("Episode via cost")

    plt.subplot(1, 2, 2)
    plt.plot(episodeViaStep)
    plt.xlabel("Episodes")
    plt.ylabel("Episode via step")
    plt.show()


def map2txt():
    global path_value
    f = open("engel.txt", "w")
    for i in range(n):
        for j in range(n):
            if reward[i, j] == -5:
                f.write("({}, {}, RED)\n".format(i, j))
            if reward[i, j] == 5:
                f.write("({}, {}, GREEN)\n".format(i, j))
            if reward[i, j] == path_value and [i, j] != agent_starting_position:
                f.write("({}, {}, GRAY)\n".format(i, j))
            if [i, j] == agent_starting_position:
                f.write("({}, {}, BLUE)\n".format(
                    agent_starting_position[0], agent_starting_position[1]))
    f.close()


current_pos = [agent_starting_position[0], agent_starting_position[1]]
settings()
screen = pygame.display.set_mode((scrx, scry))
map2txt()
while True:
    screen.fill(background)
    layout()
    pygame.draw.circle(
        screen,
        (25, 129, 230),
        (current_pos[1] * csize + 8, current_pos[0] * csize + 8),
        4,
        0,
    )

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            plt.plot(sumOfRewards)
            plt.xlabel("Episodes")
            plt.ylabel("Episode via cost")
            plt.show()

    if highestReward_counter < repeatLimit:
        episode()
    else:
        map2txt()
        draw_shortest_path()

    pygame.display.flip()
