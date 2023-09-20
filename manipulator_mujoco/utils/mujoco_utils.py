from dm_control.mujoco.wrapper import mjbindings
import numpy as np

mjlib = mjbindings.mjlib

def get_site_jac(model, data, site_id):
    """Return the Jacobian' translational component of the end-effector of
    the corresponding site id.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mjlib.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack([jacp, jacr])

    return jac

def get_fullM(model, data):
    M = np.zeros((model.nv, model.nv))
    mjlib.mj_fullM(model, M, data.qM)
    return M