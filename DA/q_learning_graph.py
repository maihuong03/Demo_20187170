import pylab as pl
import random
import networkx as nx
import numpy as np
import pickle as pkl

#this program print out shortest path to terminal

random.seed(100)
np.random.seed(50)

class PathFinder(object):
    
    def __init__(self,graph):
        self.graph = graph
        self.adjacent_mat = nx.adjacency_matrix(graph).todense()
        self.num_nodes = len(self.adjacent_mat)
        self.adjacent_mat = nx.adjacency_matrix(graph, nodelist=range(self.num_nodes)).toarray()#:D


    def q_learning(self,start_state=10,aim_state = 13, num_epoch=200, gamma=0.8, epsilon=0.05, alpha=0.1):

        len_of_paths = []
        q_table = np.zeros((self.num_nodes, self.num_nodes))  # num_states * num_actions
        
        for _ in range(1, num_epoch + 1):
            current_state = start_state
            path = [current_state]
            len_of_path = 0
            while True:
                next_state = self.epsilon_greedy(current_state, q_table, epsilon=epsilon)
                s_next_next = self.epsilon_greedy(next_state, q_table, epsilon=-0.2)  # epsilon<0, greedy policy
                # update q_table
                reward = -self.adjacent_mat[current_state][next_state]
                print("reward: ",reward)
                delta = reward + gamma * q_table[next_state, s_next_next] - q_table[current_state, next_state]
                
                q_table[current_state, next_state] = q_table[current_state, next_state] + alpha * delta
                # update current state
                current_state = next_state
                len_of_path += -reward
                path.append(current_state)
                print("current_state: ",current_state)
                print("next_state: ",next_state)

                if current_state == aim_state:
                    break
            len_of_paths.append(len_of_path)
            print("--------")
            
        return path

    def epsilon_greedy(self,s_curr, q, epsilon):#exploraiton vs exploitation 
        potential_next_states = np.where(np.array(self.adjacent_mat[s_curr]) > 0)[0]
        if random.random() > epsilon:  
            q_of_next_states = q[s_curr][potential_next_states]
            s_next = potential_next_states[np.argmax(q_of_next_states)]
        else:  
            s_next = random.choice(potential_next_states)
        return s_next


if __name__ == '__main__':
    # adjacent matrix
    # the target node is 0
    # edges = [(0,4), (4, 0), (0,3),(3,0),(1,2), (2, 1),
    #      (1,4), (4, 1), (1,8),(8,1),(1,9), (9, 1),
    #      (2,3), (3, 2), (2,6),(6,2),(1,5), (5, 1),
    #      (2,5), (5, 2), (5,6),(6,5),(7,8), (8, 7),
    #      (7,5), (5, 7), (8,9),(9,8),(8,10), (10, 8),
    #      (9,10), (10, 9)]
    with open("env.pkl", "rb") as file:
        edges = pkl.load(file)
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)

    pl.show()

    
    rl = PathFinder(G)
    res = rl.q_learning()#start_state=0,aim_state = 10
    print("path: ",res)
