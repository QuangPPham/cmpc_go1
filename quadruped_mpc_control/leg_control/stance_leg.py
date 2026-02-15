from quadruped_mpc_control.leg_control.gait import trotting
from quadruped_mpc_control.robot_model.go1 import Go1
from quadruped_mpc_control.state_estimator.state_estimator import StateEstimator
import numpy as np
import sys

try:
    import mpc_osqp as mpc
except:
    print("Run 'pip install -e .' in the root folder")
    sys.exit()

DTYPE = np.float32

class StanceLegControlMPC():
    def __init__(self, robot: Go1, horizonLength: int = 10, dtMpc: float = 0.03, gait = trotting,
                 state_estimator: StateEstimator = None, qp_solver = mpc.QPOASES):
        self._robot = robot
        self._stateEstimator = state_estimator
        self.desired_speed = np.zeros(2, dtype=DTYPE)
        self.desired_yaw_rate = 0.0
        self._desired_body_height = robot._bodyHeight
        self._friction_coeffs = robot._friction_coeffs
        self._gait = gait

        self._mpc = mpc.ConvexMpc(
            robot._bodyMass,            # body mass
            list(robot._bodyInertia),   # body inertia
            4,                          # num legs
            horizonLength,              # planning horizon
            dtMpc,                      # planning rate
            list(robot._mpc_weights),   # mpc_weights for cost
            1e-5,                       # alpha, see paper for details
            qp_solver                   # solver
        )
    
    def updateCommand(self, command):
        """Update desired velocity and yaw rate
        """
        self.desired_speed = np.array([command[0], command[1]])
        self.desired_yaw_rate = command[2]
    
    def get_action(self):
        """Computes the torque for stance legs
        """
        # Update data
        self._stateEstimator.update()
        # com_pos doesn't matter
        desired_com_position = np.array([0., 0., self._desired_body_height],dtype=DTYPE)
        # want base at constant height
        desired_com_velocity = np.array([self.desired_speed[0], self.desired_speed[1], 0.], dtype=DTYPE)
        # want base to be parallel to ground. Also yaw in body-aligned frame is zero
        desired_com_roll_pitch_yaw = np.array([0., 0., 0.], dtype=DTYPE)
        # only want to change twisting speed
        desired_com_angular_velocity = np.array([0., 0., self.desired_yaw_rate], dtype=DTYPE)

        # print(self._stateEstimator.com_pos,
        #       self._stateEstimator.com_linvel_body,
        #       self._stateEstimator.com_rpy,
        #       self._stateEstimator.com_angvel_body,
        #       desired_com_position,
        #       desired_com_velocity,
        #       desired_com_roll_pitch_yaw,
        #       desired_com_angular_velocity,
        #       sep="\n")

        # Run MPC
        predicted_contact_forces = self._mpc.compute_contact_forces(
            np.asarray(self._stateEstimator.com_pos, dtype=DTYPE),          # com_position
            np.asarray(self._stateEstimator.com_linvel_body, dtype=DTYPE),  # com_velocity
            np.asarray(self._stateEstimator.com_rpy, dtype=DTYPE),          # com_roll_pitch_yaw
            np.asarray(self._stateEstimator.com_angvel_body, dtype=DTYPE),  # com_angular_velocity
            np.asarray(self._gait.getMPCtable(), dtype=DTYPE),              # foot_contact_states
            np.asarray(self._robot.getLocalFeetPosition().flatten(), dtype=DTYPE),  #foot_positions_base_frame
            self._friction_coeffs,                                          # foot_friction_coeffs
            desired_com_position,                                           # desired_com_position
            desired_com_velocity,                                           # desired_com_velocity
            desired_com_roll_pitch_yaw,                                     # desired_com_roll_pitch_yaw
            desired_com_angular_velocity                                    # desired_com_angular_velocity
        )

        """ Convert contact force to joint torques """
        # Get contact force at each leg
        contact_forces = {}
        for i in range(4):
            contact_forces[i] = np.array(predicted_contact_forces[i*3 : (i+1)*3])

        #print(f"Stance: {contact_forces}")
        # Get motor torque for each motor
        action = np.zeros(self._robot._num_motors, dtype=DTYPE)
        for leg_id, force in contact_forces.items():
            # get motor torque for each leg
            motor_torques = self._robot.computeTorquefromForce(leg_id, force)
            # append to action array, motor_torques already includes the motor index
            for joint_id, torque in motor_torques.items():
                action[joint_id] = torque

        # which leg should output force is already accounted for by the gaitTable passed
        # to the MPC solver
        return action, contact_forces
