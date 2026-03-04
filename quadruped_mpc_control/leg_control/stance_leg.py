from quadruped_mpc_control.leg_control.gait import trotting
from quadruped_mpc_control.robot_model.go1 import Go1
from quadruped_mpc_control.state_estimator.state_estimator import StateEstimator
import numpy as np
from scipy.linalg import expm
import sys

try:
    import mpc_osqp as mpc
    import rf_mpc_osqp as rf_mpc
except Exception as error:
    print(error)
    sys.exit()

DTYPE = np.float32

qp_solver_map = {
    "QPOASES": mpc.QPOASES,
    "OSQP": mpc.OSQP,
}

rf_qp_solver_map = {
    "QPOASES": rf_mpc.QPOASES,
    "OSQP": rf_mpc.OSQP,
}

class StanceLegControlMPC():
    def __init__(self, robot: Go1, horizonLength: int = 10, dtMpc: float = 0.03, gait = trotting,
                 state_estimator: StateEstimator = None, qp_solver = "QPOASES", mpc_weights = None):
        self._robot = robot
        self._stateEstimator = state_estimator
        self.desired_speed = np.zeros(2, dtype=DTYPE)
        self.desired_yaw_rate = 0.0
        self._desired_body_height = robot._bodyHeight
        self._friction_coeffs = robot._friction_coeffs
        self._gait = gait

        self.init_mpc(dtMpc, horizonLength, qp_solver, mpc_weights)

    def init_mpc(self, dtMpc, horizonLength, qp_solver, _mpc_weights):
        self._mpc = mpc.ConvexMpc(
            self._robot._bodyMass,            # body mass
            list(self._robot._bodyInertia),   # body inertia
            4,                          # num legs
            horizonLength,              # planning horizon
            dtMpc,                      # planning rate
            list(_mpc_weights),         # mpc_weights for state cost
            1e-5,                       # alpha for control cost
            qp_solver_map[qp_solver]    # solver
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
        desired_com_velocity_body = np.array([self.desired_speed[0], self.desired_speed[1], 0.], dtype=DTYPE)
        desired_com_velocity = self._robot.getBaseRotMat() @ desired_com_velocity_body
        # want base to be parallel to ground. Also yaw in body-aligned frame is zero
        desired_com_roll_pitch_yaw = np.array([0., 0., 0.], dtype=DTYPE)
        # only want to change twisting speed
        desired_com_angular_velocity = np.array([0., 0., self.desired_yaw_rate], dtype=DTYPE)

        # Run MPC
        predicted_contact_forces = self._mpc.compute_contact_forces(
            np.asarray(self._stateEstimator.com_pos, dtype=DTYPE),          # com_position
            np.asarray(self._stateEstimator.com_linvel_world, dtype=DTYPE), # com_velocity
            np.asarray(self._stateEstimator.com_rpy, dtype=DTYPE),          # com_roll_pitch_yaw
            np.asarray(self._stateEstimator.com_angvel_world, dtype=DTYPE), # com_angular_velocity
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

        # print(f"Stance: {contact_forces}")
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

class StanceLegControlRFMPC(StanceLegControlMPC):
    def __init__(self, robot, horizonLength = 10, dtMpc = 0.03, gait=trotting,
                 state_estimator = None, qp_solver="QPOASES", mpc_weights = None):
        super().__init__(robot, horizonLength, dtMpc, gait, state_estimator, qp_solver, mpc_weights)
        self._iter = 0
        self.contact_forces = {}

    def init_mpc(self, dtMpc, horizonLength, qp_solver, mpc_weights):
        self._mpc = rf_mpc.RFConvexMpc(
            self._robot._bodyMass,            # body mass
            list(self._robot._bodyInertia),   # body inertia
            4,                                # num legs
            horizonLength,                    # planning horizon
            dtMpc,                            # planning rate
            list(mpc_weights),                # mpc_weights for cost
            0.1,                              # force cost
            rf_qp_solver_map[qp_solver]       # solver
        )

    def get_action(self):
        # Update data
        self._stateEstimator.update()
        # com_pos doesn't matter
        desired_com_position = np.array([0., 0., self._desired_body_height],dtype=DTYPE)
        # want base at constant height
        desired_com_velocity_body = np.array([self.desired_speed[0], self.desired_speed[1], 0.], dtype=DTYPE)
        desired_com_velocity = self._robot.getBaseRotMat() @ desired_com_velocity_body
        # want base to be parallel to ground. Also yaw in body-aligned frame is zero
        des_yaw = self._stateEstimator.com_rpy[2] + self.desired_yaw_rate * 0.03
        des_hat = np.zeros((3,3))
        des_hat[0, 1] = -des_yaw
        des_hat[1, 0] = des_yaw
        desired_com_R = expm(des_hat)
        # only want to change twisting speed
        desired_com_angular_velocity = np.array([0., 0., self.desired_yaw_rate], dtype=DTYPE)

        # operating point force
        leg_state = self._gait.getLegStates()
        f_op = np.zeros((4, 3))
        f_d = np.zeros((4, 3))
        for i, state in enumerate(leg_state):
            if state == 1:
                f_d[i, 2] = self._robot._bodyMass*9.81 / np.sum(leg_state)
                # if self._iter == 0:
                f_op[i, 2] = self._robot._bodyMass*9.81 / np.sum(leg_state)
                # OSQP does not like the below for some reason, but QPOASES is good
                # else:
                #     # if np.isclose(self.contact_forces[i], [0., 0., 0.]).all():
                #     #     f_op[i, 2] = self._robot._bodyMass*9.81 / np.sum(leg_state)
                #     # else:
                #     #     f_op[i] = - self.contact_forces[i]
                #     f_op[i,:] = self._robot.getFeetForce(i)
        # print(f_op)
        # Run MPC
        predicted_contact_forces = self._mpc.compute_contact_forces(
            np.asarray(self._stateEstimator.com_pos, dtype=DTYPE),          # com_position
            np.asarray(self._stateEstimator.com_linvel_world, dtype=DTYPE), # com_velocity
            np.asarray(self._robot.getBaseRotMat().flatten(), dtype=DTYPE), # com_R
            np.asarray(self._stateEstimator.com_angvel_body, dtype=DTYPE),  # com_angular_velocity
            np.asarray(f_op, dtype=DTYPE).flatten(),                        # operating-point force
            np.asarray(self._gait.getMPCtable(), dtype=DTYPE),              # foot_contact_states
            np.asarray(self._robot.getLocalFeetPosition().flatten(), dtype=DTYPE),  # foot_positions_base_frame
            self._friction_coeffs,                                          # foot_friction_coeffs
            desired_com_position,                                           # desired_com_position
            desired_com_velocity,                                           # desired_com_velocity
            desired_com_R.flatten(),                                        # desired_com_R
            desired_com_angular_velocity,                                   # desired_com_angular_velocity
            np.asarray(f_d, dtype=DTYPE).flatten()                          # desired foot force
        )
        
        """ Convert contact force to joint torques """
        # Get contact force at each leg
        # contact_forces = {}
        for i, state in enumerate(leg_state):
            if state == 1:
                self.contact_forces[i] = -(f_op[i] + np.array(predicted_contact_forces[i*3 : (i+1)*3]))
            else:
                self.contact_forces[i] = np.zeros(3, dtype=DTYPE)

        #print(f"Stance: {contact_forces}")
        # Get motor torque for each motor
        action = np.zeros(self._robot._num_motors, dtype=DTYPE)
        for leg_id, force in self.contact_forces.items():
            # get motor torque for each leg
            motor_torques = self._robot.computeTorquefromForce(leg_id, force)
            # append to action array, motor_torques already includes the motor index
            for joint_id, torque in motor_torques.items():
                action[joint_id] = torque

        self._iter += 1

        # which leg should output force is already accounted for by the gaitTable passed
        # to the MPC solver
        return action, self.contact_forces
