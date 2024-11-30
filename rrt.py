### Dont need for VM ##############
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
########################
import numpy as np
import random
from lib.detectCollision import detectCollision, isRobotCollided
from lib.loadmap import loadmap
from copy import deepcopy

def random_sample(lowerLim, upperLim, goal=None, goal_bias=0.1):
    """
    Samples a random configuration that is within the joint limits. 
    Goal bias can be adjusted so that it samples the goal more or less frequently.
    If no goal is passed, just finds a random configuration

    return: A random 1x7 configuration for the joints (node)
    """
    if goal is not None and random.random() < goal_bias:
        return goal
    else:
        return np.random.uniform(low=lowerLim, high=upperLim)
    
def nearest_node(q, nodes):
    """
    Finds the nearest node from the lost of nodes to configuration q.
    Uses Euclidean distance.

    return: The 1x7 configuration of the nearest node
    """
    distances = [np.linalg.norm(q - node) for node in nodes]
    nearest_index = np.argmin(distances) 

    return nodes[nearest_index]

def is_path_collision_free(start, end, map, step_size=0.05):
    """
    Checks if the straigt line connecting the start and end nodes is collision free.
    Samples along the line based on step size.
    
    returns: True if collsion free, false if collision
    """
    direction = (end - start) / np.linalg.norm(end - start)
    num_samples = int(np.linalg.norm(end - start) / step_size)

    for i in range(num_samples + 1):
        point = start + i * step_size * direction
        if isRobotCollided(point, map):
            return False
    return True

def construct_path(tree, goal):
    """
    Contructs the path from the start to the goal using the tree. 
    Works backwards from goal.
    
    return: List of nodes representing the path
    """
    # If there is no possible path
    if tuple(goal) not in tree:
        print("No path found!")
        return np.array([])
    
    path = []
    current_node = goal

    while current_node is not None:
        path.append(current_node)
        current_node = tree[tuple(current_node)]
    
    path.reverse()
    return np.array(path)

def rrt(map, start, goal, max_iterations=1000, step_size=0.1):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :param max_iterations: maximum number of iterations to try
    :param step_size: distance to extend each step in the RRT
    :param goal_threshold: distance threshold to consider the goal reached
    
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """
    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    # Ensure start config is within joint limits
    if not np.all((start >= lowerLim) & (start <= upperLim)):
        print("Starting configuration is not within joint limits!")
        return np.array([])
    
    # First check the edge case if the start = goal
    if np.array_equal(start, goal):
        return np.array([start]) 
    
    # initialize path
    path = []

    # Initialize tree and set of nodes
    nodes = [start]
    tree = {tuple(start): None} # Key is the node in the tree, the value is the parent node

    # Check for start configuration collision
    if isRobotCollided(start, map):
        print("Starting configuration is in collision!")
        return np.array([])

    # RRT main loop
    iterations = 0
    for i in range(max_iterations):

        # Sample random point within joint limits -- uses default goal bias (change for tuning)
        q_rand = random_sample(lowerLim, upperLim, goal)

        # Ensure this point is not in a collision, if it is move to next iteration
        if isRobotCollided(q_rand, map):
            continue

        # Find the nearest node to random point
        q_near = nearest_node(q_rand, nodes)

        # Compute a new node by moving in the direction of the random node
        direction = (q_rand - q_near) / np.linalg.norm(q_rand - q_near)
        q_new = q_near + step_size * direction

        # If path from node to new node is collision free, add it to the tree and set of nodes
        if is_path_collision_free(q_near, q_new, map, step_size):
            nodes.append(q_new)
            tree[tuple(q_new)] = tuple(q_near)  # Add the new node to the tree, with the nearest node as the parent
        else:
            # If not collsiion free, move to next iteration
            continue

        # Check if we can connect to the goal from the new node without a collision
        if is_path_collision_free(q_new, goal, map, step_size):
            nodes.append(goal)
            tree[tuple(goal)] = q_new
            print("Goal reached!")
            break
        
        iterations += 1
    # Extract the path
    path = construct_path(tree, goal)
    print("Path found in " + str(iterations) + " iterations")
        
    return path

if __name__ == '__main__':
    map_struct = loadmap(r"C:\Users\Zach\MEAM_5200\meam520_labs\maps\map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    print(np.round(path,2))

