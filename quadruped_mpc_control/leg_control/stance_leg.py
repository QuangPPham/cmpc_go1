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
        """Update desired velocity (body frame) and yaw rate (body frame)
        """
        self.desired_speed = np.array([command[0], command[1]])
        self.desired_yaw_rate = command[2]
    
    def get_action(self, ground_normal_vec = np.array([0., 0., 1.], dtype=DTYPE)):
        """Computes the torque for stance legs
        """
        
        ground_z = ground_normal_vec / np.linalg.norm(ground_normal_vec)
        ground_y = np.cross(ground_z, [1., 0., 0.])
        ground_x = np.cross(ground_y, ground_z)
        ground_rot_mat = np.column_stack((ground_x, ground_y, ground_z))
        base_rot_mat_ground = ground_rot_mat.T @ self._robot.getBaseRotMat()

        # desired com_pos is in world frame, not ground
        com_pos = self._stateEstimator.com_pos

        # Determine body height
        leg_state = self._gait.getLegStates()
        leg_contact = np.where(leg_state == 1)
        leg_id = leg_contact[0][0] if leg_contact else None
        if leg_id is not None:
            feet_body_frame = self._robot.getLocalFeetPosition()
            foot_pos_ground = base_rot_mat_ground @ feet_body_frame[leg_id, :]
            base_height_ground = -foot_pos_ground[2]
        else:
            base_height_ground = self._robot._bodyHeight

        com_pos_ground = ground_rot_mat.T @ com_pos
        com_pos_ground[2] -= base_height_ground
        com_pos_projected = ground_rot_mat @ com_pos_ground
        # com_pos_projected = com_pos - (com_pos @ ground_z) * ground_z # project com_pos into ground plane -> only works if ground plane passes through origin
        desired_com_position = com_pos_projected + ground_rot_mat @ np.array([0., 0., self._robot._bodyHeight], dtype=DTYPE)
        
        # desired com_vel is in world frame, not ground
        desired_com_velocity_body = np.array([self.desired_speed[0], self.desired_speed[1], 0.], dtype=DTYPE)
        desired_com_velocity = self._robot.getBaseRotMat() @ desired_com_velocity_body

        # want base to be parallel to ground. Also yaw in body-aligned frame is zero
        desired_com_roll_pitch_yaw = np.array([0., 0., 0.], dtype=DTYPE)

        # only want to change twisting speed
        desired_com_angular_velocity_body = np.array([0., 0., self.desired_yaw_rate], dtype=DTYPE)
        desired_com_angular_velocity = self._robot.getBaseRotMat() @ desired_com_angular_velocity_body

        # Run MPC
        predicted_contact_forces = self._mpc.compute_contact_forces(
            np.asarray(self._stateEstimator.com_pos, dtype=DTYPE),          # com_position
            np.asarray(self._stateEstimator.com_linvel_world, dtype=DTYPE), # com_velocity
            np.asarray(self._stateEstimator.com_rpy, dtype=DTYPE),          # com_roll_pitch_yaw
            np.asarray(self._stateEstimator.com_angvel_world, dtype=DTYPE), # com_angular_velocity
            np.asarray(self._gait.getMPCtable(), dtype=DTYPE),              # foot_contact_states
            np.asarray(self._robot.getLocalFeetPosition().flatten(), dtype=DTYPE),  #foot_positions_base_frame
            self._friction_coeffs,                                          # foot_friction_coeffs
            np.asarray(ground_normal_vec),                                  # ground normal vector
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
                 state_estimator = None, qp_solver="OSQP", mpc_weights = None):
        """Use OSQP only for now, QPOASES is not working for some reason"""
        super().__init__(robot, horizonLength, dtMpc, gait, state_estimator, qp_solver, mpc_weights)

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

    def get_action(self, ground_normal_vec = np.array([0., 0., 1.], dtype=DTYPE)):

        ground_z = ground_normal_vec / np.linalg.norm(ground_normal_vec)
        ground_y = np.cross(ground_z, [1., 0., 0.])
        ground_x = np.cross(ground_y, ground_z)
        ground_rot_mat = np.column_stack((ground_x, ground_y, ground_z))
        base_rot_mat_ground = ground_rot_mat.T @ self._robot.getBaseRotMat()

        # desired com_pos is in world frame, not ground
        com_pos = self._stateEstimator.com_pos

        # Determine body height
        leg_state = self._gait.getLegStates()
        leg_contact = np.where(leg_state == 1)
        leg_id = leg_contact[0][0] if leg_contact else None
        if leg_id is not None:
            feet_body_frame = self._robot.getLocalFeetPosition()
            foot_pos_ground = base_rot_mat_ground @ feet_body_frame[leg_id, :]
            base_height_ground = -foot_pos_ground[2]
        else:
            base_height_ground = self._robot._bodyHeight

        com_pos_ground = ground_rot_mat.T @ com_pos
        com_pos_ground[2] -= base_height_ground
        com_pos_projected = ground_rot_mat @ com_pos_ground
        # com_pos_projected = com_pos - (com_pos @ ground_z) * ground_z # project com_pos into ground plane -> only works if ground plane passes through origin
        desired_com_position = com_pos_projected + ground_rot_mat @ np.array([0., 0., self._robot._bodyHeight], dtype=DTYPE)
        
        # desired com_vel is in world frame, not ground
        desired_com_velocity_body = np.array([self.desired_speed[0], self.desired_speed[1], 0.], dtype=DTYPE)
        desired_com_velocity = self._robot.getBaseRotMat() @ desired_com_velocity_body

        # want base to be parallel to gravity
        # des_yaw = np.pi
        # des_hat = np.zeros((3,3))
        # des_hat[0, 1] = -des_yaw
        # des_hat[1, 0] = des_yaw
        # des_yaw_R = expm(des_hat)
        desired_com_R = ground_rot_mat # @ des_yaw_R
        #desired_com_R = np.eye(3, dtype=DTYPE)
        # only want to change twisting speed
        desired_com_angular_velocity = np.array([0., 0., self.desired_yaw_rate], dtype=DTYPE)

        # operating point force
        leg_state = self._gait.getLegStates()
        f_d = np.zeros((4, 3))

        # Desired force
        for leg, state in enumerate(leg_state):
            if state == 1:
                f_d[leg, 2] = self._robot._bodyMass*9.81 / np.sum(leg_state)
        
        # Operating force
        f_op = self._robot.getFeetForce()
                
        # print(self.f_op)
        # Run MPC
        predicted_contact_forces = self._mpc.compute_contact_forces(
            np.asarray(com_pos, dtype=DTYPE),                               # com_position
            np.asarray(self._stateEstimator.com_linvel_world, dtype=DTYPE), # com_velocity
            np.asarray(self._robot.getBaseRotMat().flatten(), dtype=DTYPE), # com_R
            np.asarray(self._stateEstimator.com_angvel_body, dtype=DTYPE),  # com_angular_velocity
            np.asarray(f_op, dtype=DTYPE).flatten(),                        # operating-point force
            np.asarray(self._gait.getMPCtable(), dtype=DTYPE),              # foot_contact_states
            np.asarray(self._robot.getLocalFeetPosition().flatten(), dtype=DTYPE),  # foot_positions_base_frame
            self._friction_coeffs,                                          # foot_friction_coeffs
            np.asarray(ground_normal_vec),                                  # ground normal vector
            desired_com_position,                                           # desired_com_position
            desired_com_velocity,                                           # desired_com_velocity
            desired_com_R.flatten(),                                        # desired_com_R - row-major
            desired_com_angular_velocity,                                   # desired_com_angular_velocity
            np.asarray(f_d, dtype=DTYPE).flatten()                          # desired foot force
        )

        # If not solved, just do action previously
        if len(predicted_contact_forces) == 0:
            print("MPC error")
            predicted_contact_forces = np.zeros(12, dtype=DTYPE)
        
        """ Convert contact force to joint torques """
        # Get contact force at each leg
        contact_forces = {}
        for i, state in enumerate(leg_state):
            if state == 1:
                contact_forces[i] = -(f_op[i] + np.array(predicted_contact_forces[i*3 : (i+1)*3]))

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
