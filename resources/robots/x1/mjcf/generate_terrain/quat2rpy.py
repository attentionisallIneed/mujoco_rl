import math


def quat_to_rpy(q):
    """
    Convert a quaternion to roll, pitch, yaw (RPY) angles.

    Parameters:
    q -- A tuple or list of quaternion components (w, x, y, z)

    Returns:
    roll, pitch, yaw -- The corresponding RPY angles in radians.
    """
    w, x, y, z = q

    # Calculate roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Calculate pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi / 2, sinp)

    # Calculate yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


quat = (0.707107, -0.707107, 0, 0)
roll, pitch, yaw = quat_to_rpy(quat)
print(roll, pitch, yaw)
