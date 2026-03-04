import mujoco
import mujoco_viewer
import numpy as np

from quadruped_mpc_control.leg_control.stance_leg import StanceLegControlMPC, StanceLegControlRFMPC
from quadruped_mpc_control.leg_control.swing_leg import SwingLegControlSimple, SwingLegControlRaibert
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
                        0.0, 0.0, 1.0,
                        1.0, 1.0, 0.0,
                        0.0], dtype=np.float32)

rf_mpc_weights = np.array([0.0, 0.0, 3E5,
                           1E4, 1E4, 1E3,
                           1E4, 1E4, 1E2,
                           1E1, 1E1, 1E5], dtype=np.float32)

# Reset simulation
go1.reset()
go1.update() # need this so qpos and qvel is updated

# Set control stuff
state_estimator = StateEstimator(go1)
stance_control = StanceLegControlRFMPC(horizonLength=mpc_horizon, robot=go1, dtMpc=mpc_dt, gait=trotting, state_estimator=state_estimator, qp_solver="OSQP", mpc_weights=rf_mpc_weights)
swing_control = SwingLegControlRaibert(robot=go1, dtMPC=mpc_dt, dt=ctrl_dt, gait=trotting, state_estimator=state_estimator)
# stance_control = StanceLegControlMPC(horizonLength=mpc_horizon, robot=go1, dtMpc=mpc_dt, gait=trotting, state_estimator=state_estimator, qp_solver="QPOASES", mpc_weights=mpc_weights)

viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=False)
viewer._render_every_frame = False

print("Press ESC to exit")

command = [0., 1., 0.]
stance_control.updateCommand(command)
swing_control.updateCommand(command)

while True:
    # Apply ctrl at 100 Hz
    if (sim_iter % sim_iter_per_ctrl) == 0:
        # Update robot states
        state_estimator.update()

        # Apply MPC at 33 Hz
        if (ctrl_iter % ctrl_iter_per_mpc) == 0:
            mpc_action, mpc_force = stance_control.get_action()
        
        # Get swing leg actions - after stance leg to update gait
        action = swing_control.get_action()

        action += mpc_action
        ctrl_iter += 1

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
