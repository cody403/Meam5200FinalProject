import numpy as np
from lib.detectCollision import detectCollision
from lib.calculateFKJac import FK_Jac


BUFFER_RADIUS = 0.05
HEIGHT_LIMIT = 0.2
LOWER = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
UPPER = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

fk = FK_Jac()

class Node:
    def __init__(self, q, parent=None):
        self.q = q
        self.parent = parent
        self.step_size = 1.0

    def extend(self, q_target):
        direction = (q_target.q - self.q)
        direction = direction / np.linalg.norm(direction)
        new_q = self.q + self.step_size * direction
        return new_q  

    def distance(self, other: 'Node') -> float:
        return np.linalg.norm(self.q - other.q)
    
class Tree:
    def __init__(self, obstacles, start):
        self.nodes = [start]
        self.obstacles = obstacles
        self.neighbor_radius = 0.1

    def extend(self, new_node):
        min_dist = np.inf
        for node in self.nodes:
            dist = new_node.distance(node)
            if dist < min_dist:
                min_dist = dist
                q_near = node

        new_q = q_near.extend(new_node)
        q_new = Node(new_q, q_near)

        if self.is_path_valid(q_near.q, new_q):
            self.nodes.append(q_new)
            return q_new
        else:
            return None
        

    def rewire(self, new = 'Node') -> None:
        for neighbor in self.nodes:
            distance = neighbor.distance(new)
            if distance < self.neighbor_radius:
                pass
                #something
                #
        

    def limits_exceeded(self, q) -> bool:
        if any((LOWER-q) > 0) or any((q-UPPER) > 0):
            return True
        return False
        

    def is_configuration_valid(self, q, joint_positions = None) -> bool:
        #checks joint limits
        if self.limits_exceeded(q):
            return False
        
        #saves time not computing joint locations if they've been computed
        if joint_positions is None:
            joint_positions, _ = fk.forward_expanded(q)

        #goes under the table
        if any(joint_positions[:, 2] <= HEIGHT_LIMIT):
            return False
        
        #hits a box
        for box in self.obstacles:
            if any(detectCollision(joint_positions[:-1, :], joint_positions[1:,:], box)):
                return False
        return True
    

    def create_joints_along_links(self, joint_positions, num_steps = 3):
        points = []
        for joint_number in range(len(joint_positions) - 1):
            curr_joint = joint_positions[joint_number]
            next_joint = joint_positions[joint_number + 1]
            for i in range(num_steps):
                points.append(curr_joint + (next_joint - curr_joint)*(i/num_steps))
        return np.array(points)
    

    def is_path_valid(self, start, end):
        start_joint_positions, _ = fk.forward_expanded(start)
        end_joint_positions, _ = fk.forward_expanded(end)
        if not (self.is_configuration_valid(start, start_joint_positions) and self.is_configuration_valid(end, end_joint_positions)):
            return False
        
        start_points = self.create_joints_along_links(start_joint_positions)
        end_points = self.create_joints_along_links(end_joint_positions)
        for box in self.obstacles:
            if any(detectCollision(start_points, end_points, box)):
                return False
        return True


    def trace_path(self, node: 'Node') -> list:
        path = []
        while node is not None:
            path.append(node.q)
            node = node.parent
        path.reverse()
        return path
    

def buffered_obstacles(map):
    obstacles = map.obstacles
    br = BUFFER_RADIUS
    for obstacle in obstacles:
        obstacle = obstacle + [-br, -br, -br, br, br, br]
    return obstacles



def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """


    # initialize path
    path = []
    obstacles = buffered_obstacles(map)

    # initialize the tree
    tree = Tree(obstacles, Node(start))
    q_goal = Node(goal)

    #If start or end configurations aren't valid
    if not tree.is_configuration_valid(start) or not tree.is_configuration_valid(goal):
        return []

    for _ in range(100):
        q_rand = Node(np.random.uniform(LOWER, UPPER, size=(1,7)).flatten())

        q_new = tree.extend(q_rand)

        #
        #
        #rewire for rrt*
        tree.rewire(q_new)
        
        #if there is a collision skip to the next
        if q_new is None:
            continue

        valid_path_to_end = tree.is_path_valid(q_rand.q, q_goal.q)

        if valid_path_to_end:
            q_goal.parent = q_new
            path = tree.trace_path(q_goal)
            break
        
            
    return np.array(path)

if __name__ == '__main__':
    import time
    from lib.loadmap import loadmap
    map_struct = loadmap("./maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])

    start_time = time.time()
    path = rrt(map_struct, start, goal)
    print(f"Path found in {round(time.time() - start_time, 4)} seconds")

    for i in range(path.shape[0]):
        print(f"Iteration {i}: {np.round(path[i], 2)}")
    
