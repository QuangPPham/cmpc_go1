import mujoco
import mujoco_viewer

from quadruped_mpc_control.leg_control.stance_leg import StanceLegControlMPC
from quadruped_mpc_control.leg_control.swing_leg import SwingLegControlSimple
from quadruped_mpc_control.leg_control.gait import trotting, bounding, pacing, pronking
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

go1 = Go1(model, data)
# Reset simulation
go1.reset()
go1.update() # need this so qpos and qvel is updated

# Set control stuff
state_estimator = StateEstimator(go1)
stance_control = StanceLegControlMPC(robot=go1, dtMpc=mpc_dt, gait=trotting, state_estimator=state_estimator)
swing_control = SwingLegControlSimple(robot=go1, dtMPC=mpc_dt, dt=ctrl_dt, gait=trotting, state_estimator=state_estimator)

viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=False)
viewer._render_every_frame = False

print("Press ESC to exit")

command = [1., 0., 0.]
stance_control.updateCommand(command)
swing_control.updateCommand(command)

while True:
    # Apply ctrl at 100 Hz
    if (sim_iter % sim_iter_per_ctrl) == 0:
        action = swing_control.get_action()
        # Apply MPC at 33 Hz
        if (ctrl_iter % ctrl_iter_per_mpc) == 0:
            mpc_action, mpc_force = stance_control.get_action()
        action += mpc_action
        ctrl_iter += 1

    go1.apply_action(action)
    go1.update()
    viewer.render()

    sim_iter += 1

    if not viewer.is_alive:
        break
