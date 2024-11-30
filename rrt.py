import numpy as np
import random
from lib.detectCollision import detectCollision, isRobotCollided, isPathCollided
from lib.loadmap import loadmap
from copy import deepcopy

class node():
    def __init__(self, parent, q):
        self.parent = parent
        self.q = q

def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (1x7).
    :param goal:        goal pose of the robot (1x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """
    # initialize path
    #boundary = map.boundary
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
    newNode = np.zeros(7)
    origin = node("", start)
    nodes = [origin]
    if isRobotCollided(start, map) | isRobotCollided(goal, map):
        print("Path Endpoint Positions are Collided")
        return []
    while isPathCollided(nodes[-1].q, goal, map):
        for ii in range(7):
            newNode[ii] = random.uniform(lowerLim[ii], upperLim[ii])
        minDist = np.linalg.norm(newNode - start)
        nearest = deepcopy(origin)
        for config in nodes:
            if np.linalg.norm(newNode - config.q) < minDist:
                nearest = deepcopy(config)
                minDist = np.linalg.norm(newNode - nearest.q)

        if isPathCollided(nearest.q, newNode, map):
            continue
        if isRobotCollided(newNode, map):
            continue
        nodes.append(node(deepcopy(nearest), deepcopy(newNode)))
        print("newest", nodes[-1].q)
        print("parent", nodes[-1].parent.q)
    pathBack = deepcopy(nodes[-1])
    path = [pathBack.q, goal]

    while (pathBack.parent != ""):
        path.insert(0, pathBack.parent.q)
        pathBack = pathBack.parent


    print("Path", np.array(path))
    return np.array(path)

if __name__ == '__main__':
    map_struct = loadmap("../maps/map2.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
