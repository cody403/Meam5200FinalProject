import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    fk = FK()

    transformations = fk.compute_Ai(q_in)
    d_7_0 = fk.transform_i_to_base(7, transformations)[0:3, 3]

    jacobian = np.zeros(shape = (6, 1))
    for i in range(len(q_in)):
        T_i_0 = fk.transform_i_to_base(i, transformations)
        R_i_0_z = T_i_0[0:3, 2]
        d_i_0 = T_i_0[0:3, 3]

        J_V_i = (np.cross(R_i_0_z, (d_7_0 - d_i_0))).reshape(3, 1)
        J_W_i = R_i_0_z.reshape(3, 1)

        J_i = np.concatenate((J_V_i, J_W_i), axis = 0)
        jacobian = np.concatenate((jacobian, J_i), axis = 1)

    jacobian = jacobian[:, 1:8]

    return jacobian

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
