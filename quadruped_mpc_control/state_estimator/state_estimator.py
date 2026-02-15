import numpy as np
import mujoco
from quadruped_mpc_control.state_estimator.moving_window_filter import MovingWindowFilter
from quadruped_mpc_control.robot_model.go1 import Go1

def quat_to_rpy(q) -> np.ndarray:
    """
    Convert a quaternion to RPY. Return
    angles in (roll, pitch, yaw).
    """
    qx, qy, qz, qw = q
    rpy = np.zeros((3,1), dtype=np.float32)
    s = np.min([2.*(qx*qz-qw*qy),1.0])
    # yaw
    rpy[0] = np.arctan2(2.*(qy*qz+qw*qx), qw**2 - qx**2 - qy**2 + qz**2)
    # pitch
    rpy[1] = np.arcsin(s)
    # roll
    rpy[0] = np.arctan2(2.*(qx*qy+qw*qz), qw**2 + qx**2 - qy**2 - qz**2)
    return rpy

class StateEstimator():
  """Get the CoM states:
      - position
      - roll, pitch, yaw
      - translational velocity
      - angular velocity
  """

  def __init__(self, robot: Go1):
    self._robot = robot
    self.reset(0)

  @property
  def com_pos(self):
    """Base position in world frame
    """
    return self._com_position

  @property
  def com_rpy(self):
    """Base roll-pitch-yaw in world frame
    """
    return self._com_rpy

  @property
  def com_linvel_body(self):
    """The base velocity projected in the body-aligned world inertial frame.
    """
    return self._com_velocity_body_frame
  
  @property
  def com_linvel_world(self):
    """The base velocity projected in the world inertial frame.
    """
    return self._com_velocity_world_frame
  
  @property
  def com_angvel_body(self):
    """The base angular velocity projected in the body-aligned world inertial frame.
    """
    return self._com_angular_velocity_body_frame

  def reset(self, current_time):
    """Reset data
    """
    del current_time
    self._com_position = np.array([0., 0., self._robot._bodyHeight])
    self._com_rpy = np.array([0., 0., 0.])
    self._com_velocity_world_frame = np.array([0., 0., 0.])
    self._com_velocity_body_frame = np.array([0., 0., 0.])
    self._com_angular_velocity_world_frame = np.array([0., 0., 0.])
    self._com_angular_velocity_body_frame = np.array([0., 0., 0.])

  def update(self):
    """Update state from simulation
    """
    # Get CoM position
    self._com_position = self._robot.getTrueBasePosition()

    # Get CoM RPY
    """
    Note: RPY consist yaw, followed by pitch, then roll
    Therefore, a yaw-aligned RPY will be [roll, pitch, 0]
    """
    base_quat = self._robot.getTrueBaseOrientation()
    base_rpy = quat_to_rpy(base_quat)
    self._com_rpy = base_rpy.copy()
    self._com_rpy[2] = 0.

    # Get CoM linear velocity in world frame
    self._com_velocity_world_frame = self._robot.getTrueBaseVelocity()

    # Get transform from world to base frame
    base_rot_mat = self._robot.getBaseRotMat()

    # Get CoM linear and angular velocity in body frame
    self._com_velocity_body_frame = base_rot_mat.T @ self._com_velocity_world_frame
    self._com_angular_velocity_world_frame = self._robot.getTrueBaseAngularVelocity()
