from quadruped_mpc_control.leg_control.gait import trotting
from quadruped_mpc_control.robot_model.go1 import Go1
from quadruped_mpc_control.state_estimator.state_estimator import StateEstimator
import numpy as np
import copy

DTYPE = np.float32

def cubicBezier(y0, yf, x):
    # Phase x is between 0 to 1
    assert x >= 0. and x <= 1.
    yDiff = yf - y0
    bezier = x**3 + 3*(x**2 - x**3)
    return y0 + yDiff*bezier

def cubicBezierFirstDerivative(y0, yf, x):
    # Phase x is between 0 and 1
    assert x >= 0. and x <= 1.
    yDiff = yf - y0
    bezier = 6.0 * x * (1.0 - x)
        
    return yDiff * bezier

def compute_foot_trajectory(p0, pf, height, phase, swing_time):
    """
    :param p0: start position (x,y,z)
    :param pf: end position
    :param height: swing height
    :param phase: what part of the swing is it
    :param swing_time: total time from liftoff to touchdown

    return: desired position and velocity at this phase
    """
    # phase = t / swing_time
    # dp/dt = dp/dphase * dphase/dt = deriv / swing_time
    pDes = cubicBezier(p0, pf, phase)
    vDes = cubicBezierFirstDerivative(p0, pf, phase) / swing_time

    # phase = 2*t / swing_time
    # dp/dt = dp/dphase * dphase/dt = deriv * 2/swing_time
    if phase < 0.5:
        # goes from bottom to top, phase = 1 at top and 0 at bottom
        zp = cubicBezier(p0[2], p0[2] + height, phase*2)
        zv = cubicBezierFirstDerivative(p0[2], p0[2] + height, phase*2) * 2/swing_time
    else:
        # goes from top to bottom, phase = 1 at bottom and 0 at top
        zp = cubicBezier(p0[2] + height, pf[2], phase*2 - 1)
        zv = cubicBezierFirstDerivative(p0[2] + height, pf[2], phase*2 - 1) * 2/swing_time

    pDes[2] = zp
    vDes[2] = zv

    return pDes, vDes

Kp = np.diag([700., 700., 150.])
Kd = np.diag([10., 10., 10.])
class SwingLegControlRaibert():
    def __init__(self, robot: Go1, dtMPC: float = 0.03, dt: float = 0.01,
                 gait = trotting, state_estimator : StateEstimator = None):
        self._robot = robot
        self._state_estimator = state_estimator
        self._dt = dt
        self._dtMPC = dtMPC

        # Commands
        self.desired_speed = np.zeros(2, dtype=DTYPE)
        self.desired_yaw_rate = 0.0

        # Initialize gait
        self._gait = gait
        self._iterationsBetweenMPC = int(dtMPC / dt)
        self._gait.setIterations(self._iterationsBetweenMPC, 0)
        self._swing_time = self._gait.getSwingTime(dtMPC)
        self._swing_time_remaining = np.array([self._swing_time]*4)

        # desired leg height
        self._desired_height = self._robot._bodyHeight / 3

        # for initializing feet trajectory
        self._last_leg_state = self._gait.getLegStates()
        self._pI = {} # initial location

        # if leg is in swing during initialization, assume it's the beginning of the swing
        feet_pos = self._robot.getGlobalFeetPosition()
        for leg_id, state in enumerate(self._last_leg_state):
            if state == 0:
                self._pI[leg_id] = feet_pos[leg_id]

        self._joint_angles = {}
        self._iterations = 0

    def updateCommand(self, command):
        """Update desired velocity and yaw rate (body frame)
        """
        self.desired_speed = np.array([command[0], command[1]])
        self.desired_yaw_rate = command[2]

    def gait_update(self):
        """Called at each control step.
        This is for querying the gait so that we know what to do next with the legs,
        This is NOT for actually updating the legs.
        """
        # See what leg state we are supposed to have next iteration
        self._iterations += 1
        self._gait.setIterations(self._iterationsBetweenMPC, self._iterations)
        new_leg_state = self._gait.getLegStates()

        # Detects phase switch for each leg so we can remember the feet position at
        # the beginning of the swing phase.
        feet_pos = self._robot.getGlobalFeetPosition()

        for leg_id, state in enumerate(new_leg_state):
            # if leg initiates swing now, remember its location, and reset swing_time_remaining
            if (state == 0 and state != self._last_leg_state[leg_id]):
                self._pI[leg_id] = feet_pos[leg_id]
                self._swing_time_remaining[leg_id] = self._swing_time

            # check if leg is still in swing to decrease swing_time_remaining
            if (state == 0 and state == self._last_leg_state[leg_id]):
                self._swing_time_remaining[leg_id] -= self._dt

        self._last_leg_state = copy.deepcopy(new_leg_state)

    def get_feet_pos(self, ground_normal_vec):
        # Get data
        com_pos = self._state_estimator.com_pos
        com_vel_world = self._state_estimator.com_linvel_world
        com_ang_vel_world = self._state_estimator.com_angvel_world
        base_rot_mat = self._robot.getBaseRotMat()

        # Set swing leg final position
        pF_world = {}
        sideSign = [-1, 1, -1, 1]

        # Raibert Heuristics
        v_des_world = base_rot_mat @ np.hstack((self.desired_speed, 0.))
        desired_ang_vel = np.array([0., 0., self.desired_yaw_rate])

        for i in self._pI.keys():
            # make sure robot foot is in the plane of hip angle = 0
            offset = np.array([0., sideSign[i]*self._robot._abadLinkLength, 0])
            rHipRobotFrame = self._robot.getHipLocation(i) + offset
            # hip displacement from CoM in ground frame
            rHipWorld = base_rot_mat @ rHipRobotFrame
            
            # hip velocity in world frame
            vHipWorld = com_vel_world + np.cross(com_ang_vel_world, rHipWorld)
            # target hip velocity
            wHipWorldDes = base_rot_mat @ desired_ang_vel
            vHipWorldDes = v_des_world + np.cross(wHipWorldDes, rHipWorld)

            # Offset from hip, not from CoM
            pF_world[i] = com_pos + rHipWorld + vHipWorld * self._swing_time_remaining[i] + 0.03 * (vHipWorld - vHipWorldDes)
            pF_world[i][2] = 0.0
            # find intersection of pF with the ground plane
            t = (ground_normal_vec @ (self._pI[i] - pF_world[i])) / (ground_normal_vec @ np.array([0., 0., 1.]))
            pF_world[i] += t * np.array([0., 0., 1.])

        return pF_world

    def get_action(self, ground_normal_vec = np.array([0., 0., 1.], dtype=DTYPE)):
        """Get motor torques
        """
        # get leg data
        swing_phases = self._gait.getSwingProgress()
        leg_state = self._gait.getLegStates()
        feet_pos = self._robot.getGlobalFeetPosition()
        feet_vel = self._robot.getGlobalFeetVelocity()

        # Delete pI for feet in stance
        for i, leg_in_contact in enumerate(leg_state):
            if leg_in_contact:
                if i in self._pI:
                    self._pI.pop(i)

        pF_world = self.get_feet_pos(ground_normal_vec)

        # compute force for each swing foot to follow desired trajectory
        spatial_forces = {}
        for i, pI in self._pI.items():
            pDes, vDes = compute_foot_trajectory(pI, pF_world[i], self._desired_height, swing_phases[i], self._swing_time)
            spatial_forces[i] = Kp@(pDes - feet_pos[i]) + Kd@(vDes - feet_vel[i])

        # print(f"Swing: {spatial_forces}")

        # Get motor torque for each motor
        action = np.zeros(self._robot._num_motors, dtype=DTYPE)
        for leg_id, force in spatial_forces.items():
            # get motor torque for each leg
            motor_torques = self._robot.computeTorquefromForce(leg_id, force)
            # append to action array, motor_torques already includes the motor index
            for joint_id, torque in motor_torques.items():
                action[joint_id] = torque

        # -- IMPORTANT -- update gait 
        self.gait_update()

        return action
