from PathPlanner import *

# shows a graph of borders(blocked), spanning tree(vis), and robot's path(path)
def show_graph(map, path):
    import matplotlib.pyplot as plt
    blockedx = []
    blockedy = []
    for i in range(0, map.width+1):
        for j in range(0, map.height+1):
            if (map.blocked[(i,j)]):
                # print (i,j)
                blockedx.append(i * map.robot_len + map.min_x)
                blockedy.append(j * map.robot_len + map.min_y)
    plt.plot(blockedx, blockedy, '.')

    # visx = []
    # visy = []
    # for i in range(0, map.width+1):
    #     for j in range(0, map.height+1):
    #         if (map.vis[(i,j)]):
    #             visx.append(i * map.robot_len + map.min_x)
    #             visy.append(j * map.robot_len + map.min_y)
    # plt.plot(visx, visy, '*')
    
    pathx = []
    pathy = []
    for i in path:
        pathx.append(i[0] * map.robot_len + map.min_x)
        pathy.append(i[1] * map.robot_len + map.min_y)
    plt.plot(pathx, pathy, '.-')

    plt.show()

def main():

    # initialize a map object
    map_path = "Path to initializing map"
    my_map = Map()
    my_map.initialize_map(map_path)
    
    # initialize a path planner by giving it map object and the coordinate of 
    # the robot in map's original unit. Then start generating spanning tree,
    # generating path, and drawing out resulted path graph
    planner = PathPlanner(my_map, 20, 10)
    planner.spanning_tree()
    # planner.show_sp(root)
    planner.draw_path()
    result = planner.get_path()
    show_graph(my_map, result)




if __name__ == "__main__":
    main()