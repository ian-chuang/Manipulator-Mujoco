import numpy as np
from manipulator_mujoco.utils.transform_utils import (
    quat_multiply,
    quat2mat,
    axisangle2quat,
    quat2axisangle
)

EPS = np.finfo(float).eps * 1e8

def task_space_inertia_matrix(M, J, threshold=1e-3):
    """Generate the task-space inertia matrix

    Parameters
    ----------
    M: np.array
        the generalized coordinates inertia matrix
    J: np.array
        the task space Jacobian
    threshold: scalar, optional (Default: 1e-3)
        singular value threshold, if the detminant of Mx_inv is less than
        this value then Mx is calculated using the pseudo-inverse function
        and all singular values < threshold * .1 are set = 0
    """

    # calculate the inertia matrix in task space
    M_inv = np.linalg.inv(M)
    Mx_inv = np.dot(J, np.dot(M_inv, J.T))
    if abs(np.linalg.det(Mx_inv)) >= threshold:
        # do the linalg inverse if matrix is non-singular
        # because it's faster and more accurate
        Mx = np.linalg.inv(Mx_inv)
    else:
        # using the rcond to set singular values < thresh to 0
        # singular values < (rcond * max(singular_values)) set to 0
        Mx = np.linalg.pinv(Mx_inv, rcond=threshold * 0.1)

    return Mx, M_inv

def orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (np.array): 2d array representing target orientation matrix
        current (np.array): 2d array representing current orientation matrix

    Returns:
        np.array: 2d array representing orientation error as a matrix
    """
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

    return error

def pose_error(target_pose, ee_pose) -> np.ndarray:
    """
    Calculate the rotational error (orientation difference) between the target and current orientation.

    Parameters:
        target_ori_mat (numpy.ndarray): The target orientation matrix.
        current_ori_mat (numpy.ndarray): The current orientation matrix.

    Returns:
        numpy.ndarray: The rotational error in axis-angle representation.
    """
    target_pos = target_pose[:3]
    target_quat = target_pose[3:]
    ee_pos = ee_pose[:3]
    ee_quat = ee_pose[3:]

    err_pos = target_pos - ee_pos
    err_ori = orientation_error(quat2mat(target_quat), quat2mat(ee_quat))

    return np.concatenate([err_pos, err_ori])

    # err_quat = quat_multiply(target_quat, quat_conjugate(ee_quat))
    # angle, axis = get_rot_angle(quat2mat(err_quat))
    # return np.concatenate([err_pos, angle * axis])

def clip_orientation(orientation, orientation_limit):
    # Create a rotation matrix from the provided quaternion
    axis_angle = quat2axisangle(orientation)

    axis_angle = np.clip(axis_angle, orientation_limit[0], orientation_limit[1])
    
    clipped_orientation = axisangle2quat(axis_angle)
    
    return clipped_orientation

def offset_pose(delta, current_pose, position_limit, orientation_limit):
    current_pos = current_pose[:3]
    current_ori = current_pose[3:]

    delta_pos = delta[:3]
    delta_rot = delta[3:]

    new_pos = current_pos + delta_pos
    new_ori = quat_multiply(current_ori, axisangle2quat(delta_rot))

    new_pos = np.clip(new_pos, position_limit[0], position_limit[1])
    new_ori = clip_orientation(new_ori, orientation_limit)

    return np.concatenate([new_pos, new_ori])