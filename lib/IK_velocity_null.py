import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE

    J = calcJacobian(q_in)
    v_vector = np.concatenate((v_in.reshape(3,1), omega_in.reshape(3,1)))

    real_indices = np.where(~np.isnan(v_vector))[0].tolist()

    J_masked = J[real_indices]
    v_masked = v_vector[real_indices]

    dq = (np.linalg.lstsq(J_masked, v_masked, rcond=None)[0]).reshape((1,7))

    pseudo_J = np.linalg.pinv(J_masked)
    z = np.eye(7) - (pseudo_J @ J_masked)

    null = (z @ b)

    return dq + null


if "__main__" in __name__:
    q_in = np.array([-0.01779158, -0.76012297,  0.01978317, -2.34204913,  0.02984069,  1.54119363, 0.75344874])
    v_in = np.array([ 0.00311232, -0.03668384,  0.37611519])
    omega_in = np.array([ 3.44811838, -0.13677721, -0.19047782])

    b = np.array([ 0.01779158,  0.76012297, -0.01978317,  0.77124913, -0.02984069,  0.32630637, -0.75344874])

    dq = IK_velocity_null(q_in, v_in, omega_in, b)
    print(dq)
