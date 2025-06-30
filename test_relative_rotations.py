import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# def apply_relative_rotation_transform(Rx, Ry, Rz, r):
    
#     R_obj_rel = Rotation.from_euler('xyz', [Rx, Ry, Rz], degrees=True)
#     R_ref_rot = Rotation.from_euler('z', r, degrees=True)
#     R_world = R_ref_rot * R_obj_rel
#     R_new_rel = R_ref_rot.inv() * R_world
#     new_euler = R_new_rel.as_euler('xyz', degrees=True)

#     return new_euler


# # initial_rots = np.array([-318, 4, 6])
# initial_rots = np.array([42, 4, 6])

# for rot in range(-180, 181, 30):
#     print("initial rots: ", initial_rots)
#     # convert to radians
#     # apply the relative rotation transform
#     Rx_new, Ry_new, Rz_new = apply_relative_rotation_transform(
#         initial_rots[0], initial_rots[1], initial_rots[2], rot
#     )
    
#     print(f"Rotation: {rot} degrees -> New Rots: {np.degrees([Rx_new, Ry_new, Rz_new])}")


def rotate_reference_frame_about_z(Rx, Ry, Rz, r_degrees):
    # 1. The object's orientation is fixed in the world.
    #    The initial reference frame is assumed to be the world frame.
    R_obj_in_world = R.from_euler('zyx', [Rx, Ry, Rz], degrees=True)

    # 2. The new reference frame's orientation in the world is a rotation
    #    around the Z-axis.
    R_newframe_in_world = R.from_euler('z', r_degrees, degrees=True)

    # 3. To find the object's orientation relative to the new frame, we use
    #    the change of basis formula: R_obj_in_new = R_new_in_world⁻¹ @ R_obj_in_world
    R_obj_in_newframe = R_newframe_in_world.inv() * R_obj_in_world

    # 4. Extract the new Euler angles from the resulting rotation matrix.
    #    The new Rz will be Rz_old - r_degrees, plus coupling effects.
    Rz_new, Ry_new, Rx_new = R_obj_in_newframe.as_euler('zyx', degrees=True)

    return Rx_new, Ry_new, Rz_new

Rx, Ry, Rz = 42, 4, 6  # Initial relative pose (degrees)

for r in range(-180, 181, 30):
    new_rx, new_ry, new_rz = rotate_reference_frame_about_z(Rx, Ry, Rz, r)
    print(f"Rotation: {r:>4} degrees -> New Rots: [{new_rx:6.2f}, {new_ry:6.2f}, {new_rz:6.2f}]")