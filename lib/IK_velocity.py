import numpy as np 
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    # STUDENT CODE GOES HERE
    J = calcJacobian(q_in)
    v_vector = np.concatenate((v_in.reshape(3,1), omega_in.reshape(3,1)))

    real_indices = np.where(~np.isnan(v_vector))[0].tolist()

    J_masked = J[real_indices]
    v_masked = v_vector[real_indices]

    dq = np.linalg.lstsq(J_masked, v_masked, rcond=None)[0]

    return dq.reshape((1, 7))

if "__main__" in __name__:
    q_in = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    v_in = np.array([1, 2, 4])
    omega_in = np.array([np.nan, np.nan, np.nan])

    dq = IK_velocity(q_in, v_in, omega_in)
    print(dq)
