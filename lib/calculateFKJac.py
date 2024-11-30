import numpy as np
from math import pi

class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout
        self.transform_manual_values = [[0,       pi/-2,    0.333,  0       ],
                                        [0,       pi/2,     0,      0       ],
                                        [0.0825,  pi/2,     0.316,  0       ],
                                        [0.0825,  pi/2,     0,      pi      ],
                                        [0,       pi/2,     0.384,  pi      ],
                                        [0.088,   pi/2,     0,      0       ],
                                        [0,       0,        0.21,   pi/-4   ],
                                        [0,       pi/2,     -0.105, pi      ],
                                        [0,       pi,       0,     0       ]]
        
        #these are the z offsets for the center of joint i in coordinate frame i-1
        self.joint_offsets = [
            0.141,
            0,
            0.195,
            0,
            0.125,
            -0.015,
            0.051,
            0.0,
            0.1,
            0.1
        ]

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        # Your code starts here
        q = np.concatenate((q, np.array([0,0])), axis = 0)

        #creates all of the transformation matrices based on the given theta values
        transformations = self.compute_Ai(q)

        #creates T0e by multiplying all of the matrices
        jointPositions, T_0_e = self.compute_joint_centers(transformations)

        return jointPositions, np.array(T_0_e)

    
    def transform_matrix(self, theta : float, a : float, alpha : float, d : float, theta_offset : float):
        """creates the transformation matrix from DH values"""
        c = np.cos
        s = np.sin
        theta = theta + theta_offset
        return  np.array([[c(theta), -1*s(theta)*c(alpha), s(theta)*s(alpha), a*c(theta)],
                [s(theta), c(theta)*c(alpha), -1*c(theta)*s(alpha), a*s(theta)],
                [0, s(alpha), c(alpha), d],
                [0, 0, 0, 1]])
    

    def compute_joint_centers(self, transformations):
        center_matrix = np.empty([1, 3])
        T_0_is = []
        for i, offset in enumerate(self.joint_offsets):
            T_0_i = self.transform_i_to_base(i, transformations)
            T_0_is.append(T_0_i)
            joint_center = np.matmul(T_0_i, np.array([[0],[0], [offset], [1]]))
            joint_center = np.transpose(joint_center[0:3])
            center_matrix = np.concatenate([center_matrix, joint_center])

        return center_matrix[1:, :], T_0_is


    def transform_i_to_base(self, coord_frame : int, transformations):
        """this gives the 4x4 that defines the transformation from coordinate base i to 0"""

        #base case/error handling
        if coord_frame < 0 or coord_frame > transformations.shape[0]:
            raise ValueError(f"coordinate frame {coord_frame} does not exist in the transformation matrix")
        if coord_frame == 0:
            return np.identity(4)
        if coord_frame == 1:
            return transformations[0]
        
        #calculates value
        T_0_i = transformations[coord_frame - 1]
        for matrix in transformations[coord_frame - 2::-1]:
            T_0_i = np.matmul(matrix, T_0_i)
        return T_0_i
    

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        transformations = []
        for i, theta in enumerate(q):
            joint_vals = self.transform_manual_values[i]
            T_i1_i= self.transform_matrix(theta, joint_vals[0], joint_vals[1], joint_vals[2], joint_vals[3])
            transformations.append(T_i1_i)
        
        return np.array(transformations)
    
if __name__ == "__main__":

    fk = FK_Jac()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward_expanded(q)
    
    print("Joint Positions:\n",np.round(joint_positions, 2))
    print("End Effector Pose:\n",np.round(T0e, 2))
