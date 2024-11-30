import numpy as np


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation 
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 0x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    R_curr_0 = R_curr
    R_0_curr = R_curr_0.T
    R_des_0 = R_des

    R_curr_des = np.matmul(R_0_curr, R_des_0)

    skew = 0.5*(R_curr_des - R_curr_des.T)

    omega = np.array([skew[2, 1],skew[0, 2],skew[1, 0]])


    return np.matmul(R_curr_0, omega)



if "__main__" in __name__:
    angle1 = np.pi/2
    angle2 = np.pi/2
    R_curr_0 = np.array([[0, 1, 0],
                           [-1, 0, 0],
                           [0, 0, 1]])
    R_des_0 = np.array([[0, 0, 1],
                   [-1, 0, 0],
                   [0, -1, 0]])
    omega = calcAngDiff(R_des_0, R_curr_0)
    print(omega)

