import numpy as np
from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision


alpha = 0.02
zeta = 30
eta = 0.0015
d_0 = 0.12

# JOINT LIMITS
lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
fk = FK_Jac()

class PotentialFieldPlanner:

    def __init__(self, tol=1e-4, max_steps=3000, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """

        ## STUDENT CODE STARTS HERE
        
        #idk

        att_direction = target - current

        dist = np.linalg.norm(att_direction)

        if dist**2 > d_0:
            return att_direction / dist
        else:
            return zeta * att_direction


    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE
        dist, unit = PotentialFieldPlanner.dist_point2box(current.reshape((1, 3)), obstacle)
        dist = dist[0]
        unit = unit[0]

        if dist < 1e-4:
            dist = 1e-4
        if dist > d_0:
            return np.zeros((3,1))

        rep_f = -eta*((1/(dist)) - (1/d_0))*(1/(dist**2))*unit

        ## END STUDENT CODE

        return rep_f.reshape((3,1))

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        if dist[:, np.newaxis] == 0:
            dist[:, np.newaxis] = 0.000001
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE
        joint_forces = np.zeros(shape=target.shape)
        for joint_index in range(target.shape[1]):
            current_xyz = current[:, joint_index].reshape((3,1))
            target_xyz = target[:, joint_index].reshape((3,1))
            F_tot = PotentialFieldPlanner.attractive_force(target_xyz, current_xyz)
            if joint_index > 6:
                F_tot = F_tot * (1/2)
            for box in obstacle:
                F_tot = F_tot + PotentialFieldPlanner.repulsive_force(box, current_xyz)
            joint_forces[:,joint_index] = F_tot.reshape((1,3))
            

        ## END STUDENT CODE

        return joint_forces
    
    @staticmethod
    def linear_jacobian_transpose(q_in, joint_index):

        joint_centers, T_0_is = fk.forward_expanded(q_in)
        d_index_0 = joint_centers[joint_index - 1]

        jacobian = np.zeros(shape = (3, joint_index))
        for i in range(joint_index):
            R_i_0_z = T_0_is[i][0:3, 2]
            d_i_0 = joint_centers[i]

            J_V_i = (np.cross(R_i_0_z, (d_index_0 - d_i_0))).reshape(3, 1)

            jacobian[:, i] = J_V_i.reshape((1,3))

        return jacobian.T
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint 
        """

        ## STUDENT CODE STARTS HERE

        joint_torques = np.zeros(shape=(1, joint_forces.shape[1]))

        for joint_index in range(2, joint_forces.shape[1]):
            J_T = PotentialFieldPlanner.linear_jacobian_transpose(q, joint_index)
            force = joint_forces[:, joint_index]
            torque_i = J_T @ force
            torque_i = torque_i.reshape((1, joint_index))
            zeros = np.zeros(shape=(1, joint_forces.shape[1] - joint_index))
            torques = np.concatenate((torque_i, zeros), axis = 1)
            joint_torques = joint_torques + torques


        ## END STUDENT CODE

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE

        distance = np.linalg.norm(target - current)

        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task. 
        """

        ## STUDENT CODE STARTS HERE

        dq = np.zeros((1, 7))

        current_joints_xyz, _ = fk.forward_expanded(q.flatten())
        target_joints_xyz, _ = fk.forward_expanded(target.flatten())

        
        joint_forces = PotentialFieldPlanner.compute_forces(target_joints_xyz.T, map_struct.obstacles, current_joints_xyz.T)
        torques = PotentialFieldPlanner.compute_torques(joint_forces, q)

        torques = torques[:, :7].flatten()
        
        #nudges the joints towards their goal orientation
        torques = torques + (0.5 * (target - q))
        
        if np.linalg.norm(torques) == 0:
            dq = np.zeros(shape = torques.shape)
        else:
            dq = alpha * (torques / np.linalg.norm(torques))

        #makes the step size smaller if we are close to the goal
        if PotentialFieldPlanner.q_distance(target, q) < 0.01:
            dq = dq * (1/100)

        ## END STUDENT CODE

        return dq
    
    
    @staticmethod
    def limits_exceeded(q):
        if any((lower-q) > 0) or any((q-upper) > 0):
            return True
        return False



    @staticmethod
    def is_configuration_valid(q, obstacles):
        #checks joint limits
        if PotentialFieldPlanner.limits_exceeded(q):
            return False
        
        #checks obstacle collisions
        joint_positions, _ = fk.forward_expanded(q)

        #goes under the table
        if any(joint_positions[:, 2] <= 0):
            return False
        
        #hits a box
        for box in obstacles:
            if any(detectCollision(joint_positions[:-1, :], joint_positions[1:,:], box)):
                return False
        return True
    
    @staticmethod
    def is_path_valid(start, end, obstacles):
        number_of_checks = 100
        for i in range(number_of_checks):
            q = start + (i/number_of_checks) * (end-start)
            if not PotentialFieldPlanner.is_configuration_valid(q, obstacles):
                return False
        return True



    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the starting configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """

        obstacles = map_struct.obstacles

        q_path = np.array([start])

        #invalid start or destination
        if not PotentialFieldPlanner.is_configuration_valid(start, obstacles):
            return np.array([]).reshape(0,7)
        if not PotentialFieldPlanner.is_configuration_valid(goal, obstacles):
            return np.array([]).reshape(0,7)

        step = 0
        q_prev = start
        q_new = np.zeros((1,7))
        while step < self.max_steps:

            #solution found
            if PotentialFieldPlanner.q_distance(goal, q_prev) < self.tol:
                break

            dq = PotentialFieldPlanner.compute_gradient(q_prev, goal, map_struct)
            q_new = q_prev + dq

            # check for collision
            if not PotentialFieldPlanner.is_configuration_valid(q_new, obstacles):
                q_new = q_prev

            #local minimum or collisions
            if np.linalg.norm(dq) < self.min_step_size or PotentialFieldPlanner.q_distance(q_new, q_prev) < self.tol:
                #random walk
                while True:
                    q_new = np.random.uniform(lower, upper)
                    if PotentialFieldPlanner.is_path_valid(q_prev, q_new, obstacles):
                        break

            q_path = np.vstack((q_path, q_new))
            q_prev = q_new
            step += 1           

        return q_path

################################
## Simple Testing Environment ##
################################


if __name__ == "__main__":
    from loadmap import loadmap
    from copy import deepcopy
    from collections import namedtuple
    from numpy import pi

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()

    # inputs 
    map_struct = loadmap("maps/map1.txt")
    #MyStruct = namedtuple("map", "obstacles")
    #map_struct = MyStruct(obstacles = np.array([]))
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])

    #start = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    #goal = np.array([0,0,0,-pi/2,0,pi,pi/4])

    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path) 