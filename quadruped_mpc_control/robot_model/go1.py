import numpy as np
from numpy import sin, cos
import mujoco

class Go1:
    def __init__(self, model, data):
        # MuJoCo model and data
        self.model = model
        self.data = data

        # Constants
        self._abadLinkLength = 0.08
        self._abadLocation = np.array([0.1881, 0.04675, 0], dtype=np.float32) # hip location
        self._feetName = ["FR", "FL", "RR", "RL"]
        self._num_motor_per_leg = 3
        self._num_motors = 12
        self._bodyMass = 12.75 # trunk: 5.204 kg, other parts: 7.539 kg
        self._bodyInertia = np.array([
                                    0.0168128557, -0.0002296769, -0.0002945293, 
                                    -0.0002296769, 0.063009565, -4.18731e-05, 
                                    -0.0002945293, -4.18731e-05, 0.0716547275
                                    ]) * 5 # for scaling due to other body parts
        self._bodyHeight = 0.26
        self._friction_coeffs = np.ones(4, dtype=np.float32) * 0.4
        # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
        self._mpc_weights = np.array([
                                5.0, 5.0, 0.0,
                                0.0, 0.0, 10.,
                                0.0, 0.0, 1.0,
                                1.0, 1.0, 0.0,
                                0.0], dtype=np.float32)
        # self._mpc_weights = np.array([1.0, 1.5, 0.0,
        #                          0.0, 0.0, 5.,
        #                          0.0, 0.0, 0.1,
        #                          1.0, 1.0, 0.1,
        #                          0.0], dtype=np.float32) * 10

    def reset(self):
        """Reset simulation at keyframe position
        """
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def update(self):
        """Step simulation
        """
        mujoco.mj_step(self.model, self.data)

    def apply_action(self, action):
        """Apply motor torques
        """
        self.data.ctrl[:] = action
    
    def getHipLocation(self, leg:int):
        """Get location of the hip for the given leg in robot frame
        """
        assert leg >= 0 and leg < 4
        pHip = np.array([
            self._abadLocation[0] if (leg == 0 or leg == 1) else -self._abadLocation[0],
            self._abadLocation[1] if (leg == 1 or leg == 3) else -self._abadLocation[1],
            self._abadLocation[2]
            ], dtype=np.float32)

        return pHip

    def getTrueBasePosition(self):
        """Get position of base in global frame
        """
        pos = self.data.qpos[:3]
        return pos

    def getTrueBaseOrientation(self):
        """Get orientation of base as a quaternion
        """
        quat = self.data.qpos[3:7]
        return quat
    
    def getBaseRotMat(self):
        """Get orientation of base as a rotation matrix
        """
        base_quat = self.getTrueBaseOrientation()

        base_rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(base_rot_mat, base_quat)
        base_rot_mat = base_rot_mat.reshape((3, 3))

        return base_rot_mat

    def getTrueBaseVelocity(self):
        """Get translational velocity of robot base in world frame
        """
        linVel = self.data.qvel[:3]
        return linVel
    
    def getTrueBaseAngularVelocity(self):
        """Get angular velocity of robot base in local frame
        https://github.com/google-deepmind/mujoco/issues/691
        """
        angVel = self.data.qvel[3:6]
        return angVel

    def getLocalFeetPosition(self):
        """Get local feet position in base frame.
        This is because the "framepos" sensor is
        defined relative to the "imu" site, which
        is at the center of the trunk
        """
        foot_world = np.zeros((4, 3))
        for i, footName in enumerate(self._feetName):
            foot_world[i,:] = self.data.sensor(footName+"_pos").data
        return foot_world

    def getGlobalFeetPosition(self):
        """Get feet location in world frame
        """
        base_world = self.getTrueBasePosition()
        base_rot_mat = self.getBaseRotMat()
        
        # get displacement from base to feet
        rel_world = self.getLocalFeetPosition() @ base_rot_mat.T # (4, 3)
        pos_global = rel_world + base_world

        return pos_global

    def getGlobalFeetVelocity(self):
        """Compute feet velocity in global frame
        """
        vel = np.zeros((4,3))
        
        for i, footName in enumerate(self._feetName):
            vel[i,:] = self.data.sensor(footName+"_global_linvel").data

        return vel

    def getLocalFeetVelocity(self):
        """Compute feet velocity in base frame
        """
        base_rot_mat = self.getBaseRotMat()
        world_vel = self.getGlobalFeetVelocity()
        local_vel = world_vel @ base_rot_mat

        return local_vel

    def computeLegJacobian(self, leg):
        """Get Jacobian of foot for the given leg in world frame
        """
        assert leg >= 0 and leg < 4
        footName = self._feetName[leg]
        foot_id = self.model.site(footName).id

        JFoot = np.zeros((3, self.model.nv))  # Translational Jacobian
        JFootr = np.zeros((3, self.model.nv)) # Rotational Jacobian

        # Compute Jacobian at the center of the geom
        mujoco.mj_jacSite(self.model, self.data, JFoot, JFootr, foot_id)

        return JFoot
    
    def computeTorquefromForce(self, leg, force):
        """Turn contact forces in world-frame to joint torques
        
        Note: tau = J.T @ f <- tau is (N,) and f is (3,)
        So, for matrix of force (c, 3), matrix of tau is (c, N)
            torque = (J.T @ force.T).T => torque = force @ J
        """
        J = self.computeLegJacobian(leg)
        # First 6 is translational and angular velocity, so we only need the rest
        # torques = (force @ J)[6:]
        gen_forces = (J.T @ force)
        torques = gen_forces[6:]

        # we only want the torque for the motors on the specific leg, and each leg has 3 motors
        # make this a dictionary for easy access in the main functioin
        motor_torques = {}
        for joint_id in range(leg*self._num_motor_per_leg, (leg+1)*self._num_motor_per_leg):
            motor_torques[joint_id] = torques[joint_id]

        return motor_torques
