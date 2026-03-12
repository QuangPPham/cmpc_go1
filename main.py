import mujoco
import mujoco_viewer
import numpy as np

from quadruped_mpc_control.leg_control.stance_leg import StanceLegControlMPC, StanceLegControlRFMPC
from quadruped_mpc_control.leg_control.swing_leg import SwingLegControlRaibert
from quadruped_mpc_control.leg_control.gait import trotting, Gait
from quadruped_mpc_control.state_estimator.state_estimator import StateEstimator
from quadruped_mpc_control.robot_model.go1 import Go1

from pathlib import Path

here = Path(__file__).resolve().parent
model_path = here / "descriptions" / "scene.xml"

# Initialize
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Timesteps
sim_dt = model.opt.timestep
ctrl_dt = 0.01
mpc_dt = 0.03

# Iteration rate
sim_iter_per_ctrl = int(ctrl_dt / sim_dt)
ctrl_iter_per_mpc = int(mpc_dt / ctrl_dt)
sim_iter = 0
ctrl_iter = 0

mpc_horizon = 10
go1 = Go1(model, data)

trotting = Gait(mpc_horizon, 10, 
                [0, 5, 5, 0], 
                [5, 5, 5, 5], "Trotting")

mpc_weights = np.array([5.0, 5.0, 0.0,
                        0.0, 0.0, 10.,
                        0.0, 0.0, 5.0,
                        1.0, 1.0, 0.0,
                        0.0], dtype=np.float32)

# (position, velocity, eta, angular_velocity)
# rf_mpc_weights = np.array([0.0, 0.0, 1E6,
#                            1E5, 1E5, 1E2,
#                            1E5, 1E5, 0.0,
#                            1E3, 1E3, 1E5], dtype=np.float32) # this is good for following velocity commands

rf_mpc_weights = np.array([1E3, 1E3, 1E6,
                           1E5, 1E5, 1E3,
                           1E5, 1E5, 0.0,
                           1E3, 1E3, 1E5], dtype=np.float32) # this is good for following velocity commands on slopes

# rf_mpc_weights = np.array([1E4, 1E4, 1E6,
#                            0.0, 0.0, 1E3,
#                            1E5, 1E5, 1E5,
#                            1E4, 1E4, 0.0], dtype=np.float32) # this is good for following desired positions and yaw

# Reset simulation
go1.reset()
go1.update() # need this so qpos and qvel is updated
state_estimator = StateEstimator(go1)
state_estimator.update() # need this so data is updated for leg control initialization

stance_control = StanceLegControlRFMPC(horizonLength=mpc_horizon, robot=go1, dtMpc=mpc_dt, gait=trotting, state_estimator=state_estimator, qp_solver="OSQP", mpc_weights=rf_mpc_weights)
swing_control = SwingLegControlRaibert(robot=go1, dtMPC=mpc_dt, dt=ctrl_dt, gait=trotting, state_estimator=state_estimator)
# stance_control = StanceLegControlMPC(horizonLength=mpc_horizon, robot=go1, dtMpc=mpc_dt, gait=trotting, state_estimator=state_estimator, qp_solver="OSQP", mpc_weights=mpc_weights)

viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=False)
viewer._render_every_frame = False

command = [0.5, 0., 0.]
stance_control.updateCommand(command)
swing_control.updateCommand(command)
# n1 = np.array([-0.1, 0.1, 1.])
n0 = np.array([ 0.0, 0.0, 1.])
n1 = n0
n2 = np.array([-0.2, 0.2, 1.])

while True:
    # Apply ctrl at 100 Hz
    if (sim_iter % sim_iter_per_ctrl) == 0:

        if go1.getFeetContact().sum() > 1:
            n = n2
        else:
            n = n1

        # Update robot states
        state_estimator.update()

        # Apply MPC at 33 Hz
        if (ctrl_iter % ctrl_iter_per_mpc) == 0:
            # print(n)
            mpc_action, mpc_force = stance_control.get_action(n)
        
        # Get swing leg actions - after stance leg to update gait
        # Using default swing leg control works better for some reason
        action = swing_control.get_action(n0)

        action += mpc_action
        ctrl_iter += 1

        # print(go1.getTrueBasePosition())

    # Apply actions
    go1.apply_action(action)

    # Update simulation and render
    go1.update()
    viewer.render()

    sim_iter += 1

    if not viewer.is_alive:
        break

    # if sim_iter > 1000:
    #     break
    
print("Done")
viewer.close()
