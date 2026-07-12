"""Race demo: 3 spider robots on parallel lanes, each with its OWN pretrained
brain, racing from a start line to a finish line. First to cross the finish y
wins. Records a top-down video and a trajectory plot; --live opens a viewer.
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
from rich.console import Console
from torch import nn

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider_with_blocks import (
    body_spider45,
)
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import VideoRecorder

console = Console()

# ------------------------- CLI -----------------------------------------------
DEFAULT_WEIGHTS = str(Path.cwd() / "__data__" /
                     "1_brain_evolution_multiprocessing" / "best_weights.npy")

parser = argparse.ArgumentParser()
parser.add_argument("--weights1", type=str, default=DEFAULT_WEIGHTS,
                    help="Weights .npy for robot in lane 0")
parser.add_argument("--weights2", type=str, default=DEFAULT_WEIGHTS,
                    help="Weights .npy for robot in lane 1")
parser.add_argument("--weights3", type=str, default=DEFAULT_WEIGHTS,
                    help="Weights .npy for robot in lane 2")
parser.add_argument("--dur", type=int, default=15)
parser.add_argument("--lane-spacing", type=float, default=1.0,
                    help="Distance between adjacent lanes (m)")
parser.add_argument("--start-y", type=float, default=0.0)
parser.add_argument("--finish-y", type=float, default=-4.0)
parser.add_argument("--out", type=str, default="racing_demo")
parser.add_argument("--live", action="store_true",
                    help="Open an interactive MuJoCo viewer window during the race")
args = parser.parse_args()

DURATION = args.dur
START_Y = args.start_y
FINISH_Y = args.finish_y
LANE_SPACING = args.lane_spacing
LANE_XS = [-LANE_SPACING, 0.0, LANE_SPACING]
WEIGHTS_PATHS = [args.weights1, args.weights2, args.weights3]

# Visual target sits just past the finish line so the vision-driven nets have
# something green to steer toward (win condition is crossing the y-line).
TARGET_POS = np.array([0.0, FINISH_Y - 0.3, 0.1])

ROBOT_COLORS = [(255, 80, 80), (80, 255, 80), (80, 160, 255)]  # RGB overlay
ROBOT_LABELS = ["Lane0", "Lane1", "Lane2"]

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / args.out
DATA.mkdir(parents=True, exist_ok=True)


def robot_start_positions() -> list[np.ndarray]:
    return [np.array([x, START_Y, 0.1]) for x in LANE_XS]


# ------------------------- Network (must match training) ---------------------
class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, state):
        x = torch.as_tensor(state, dtype=torch.float32)
        x = self.hidden_activation(self.fc1(x))
        x = self.hidden_activation(self.fc2(x))
        x = self.output_activation(self.fc4(x)) * (torch.pi / 2)
        return x.detach().numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: np.ndarray):
    v = torch.as_tensor(vector, dtype=torch.float32)
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = v[address:address + n]
        address += n
    if address != len(v):
        raise IndexError("Weight vector size mismatch")


# ------------------------- Vision helpers ------------------------------------
def isolate_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv,
                       np.array([35, 40, 40]),
                       np.array([85, 255, 255]))


def analyze_sections(mask):
    sections = np.array_split(mask, 3, axis=1)
    out = []
    for s in sections:
        total = s.size
        out.append(0.0 if total == 0 else cv2.countNonZero(s) / total)
    return out


def per_robot_state(qpos_slice: np.ndarray) -> np.ndarray:
    """Same as get_state_from_data but on a per-robot qpos slice."""
    quat = qpos_slice[3:7].copy()
    if quat[0] < 0:
        quat = -quat
    joints = qpos_slice[7:]
    return np.concatenate([quat[1:], joints])


# ------------------------- Build world ---------------------------------------
def build_world():
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()

    # Long, thin green wall along the finish line so vision nets steer to it.
    track_half_width = LANE_SPACING * 1.75
    target_body = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=TARGET_POS.tolist()
    )
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[track_half_width, 0.05, 0.2],
        rgba=[0, 1, 0, 1],
    )

    # Start line (white) and finish line (checkered-ish red) as thin visual strips.
    for name, y, rgba in [
        ("start_line", START_Y, [1.0, 1.0, 1.0, 1.0]),
        ("finish_line", FINISH_Y, [1.0, 0.1, 0.1, 1.0]),
    ]:
        line = world.spec.worldbody.add_body(name=name, pos=[0.0, y, 0.005])
        line.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[track_half_width, 0.03, 0.005],
            rgba=rgba,
            contype=0, conaffinity=0,
        )

    starts = robot_start_positions()
    # Spawn with default rotation to match training-time orientation.
    # The pretrained brain uses body quaternion as input; rotating the spawn
    # feeds out-of-distribution states and the robots steer wrong.
    for pos in starts:
        spider = body_spider45()
        world.spawn(spider.spec, position=pos.tolist())

    model = world.spec.compile()
    data = mujoco.MjData(model)
    return world, model, data, starts


def find_robot_cameras(model, num_robots: int) -> list[str | None]:
    """Return one camera name per robot, in spawn order."""
    cams_per_robot: list[list[tuple[int, str]]] = [[] for _ in range(num_robots)]
    for i in range(model.ncam):
        name = model.camera(i).name
        if "video" in name:
            continue
        # Ariel spawn prefix looks like "robot1_...", "robot2_...", etc.
        for r in range(num_robots):
            token = f"robot{r + 1}_"
            if token in name:
                cams_per_robot[r].append((i, name))
                break
    result: list[str | None] = []
    for r, entries in enumerate(cams_per_robot):
        pick = None
        for _, name in entries:
            if "core" in name or "camera" in name:
                pick = name
                break
        if pick is None and entries:
            pick = entries[0][1]
        result.append(pick)
    return result


def robot_qpos_slices(model, num_robots: int) -> list[slice]:
    """Each spider has one free joint (7 qpos) + N hinges. Assume equal splits."""
    total_qpos = model.nq
    per = total_qpos // num_robots
    if per * num_robots != total_qpos:
        raise RuntimeError(
            f"qpos not evenly divisible: nq={total_qpos}, robots={num_robots}"
        )
    return [slice(r * per, (r + 1) * per) for r in range(num_robots)]


def robot_ctrl_slices(model, num_robots: int) -> list[slice]:
    per = model.nu // num_robots
    if per * num_robots != model.nu:
        raise RuntimeError(
            f"nu not evenly divisible: nu={model.nu}, robots={num_robots}"
        )
    return [slice(r * per, (r + 1) * per) for r in range(num_robots)]


# ------------------------- Race ----------------------------------------------
def main():
    world, model, data, starts = build_world()
    num_robots = len(starts)

    q_slices = robot_qpos_slices(model, num_robots)
    c_slices = robot_ctrl_slices(model, num_robots)
    cams = find_robot_cameras(model, num_robots)
    console.log(f"Robot cameras: {cams}")

    # Network: input dim matches training (3 quat_imag + num_joints + 3 vision + 2 phase)
    num_joints_per_robot = (q_slices[0].stop - q_slices[0].start) - 7
    input_dim = 3 + num_joints_per_robot + 3 + 2
    nu_per_robot = c_slices[0].stop - c_slices[0].start

    # One independent network per lane, each loaded from its own weights file.
    nets: list[Network] = []
    for r, wpath in enumerate(WEIGHTS_PATHS):
        n = Network(input_size=input_dim, output_size=nu_per_robot, hidden_size=32)
        w = np.load(wpath)
        fill_parameters(n, w)
        nets.append(n)
        console.log(f"[green]{ROBOT_LABELS[r]} weights: {wpath}[/green]")

    # Per-robot low-res vision renderers
    vision_renderers = [mujoco.Renderer(model, height=24, width=32)
                        for _ in range(num_robots)]

    # Reset & place target
    mujoco.mj_resetData(model, data)
    target_mocap_id = model.body("green_target").mocapid[0]
    data.mocap_pos[target_mocap_id] = TARGET_POS

    # ---- Video setup: top-down, wide enough to see all robots + target ----
    video_H, video_W = 480, 640
    video_recorder = VideoRecorder(
        file_name="race_topdown",
        output_folder=str(DATA / "videos"),
    )
    os.makedirs(str(DATA / "videos"), exist_ok=True)

    viz = mujoco.MjvOption()
    viz.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    viz.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    viz.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False

    top_cam = mujoco.MjvCamera()
    top_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Center view between target and starts
    all_xy = np.array([s[:2] for s in starts] + [TARGET_POS[:2]])
    center = all_xy.mean(axis=0)
    span = float(np.max(np.ptp(all_xy, axis=0))) + 1.5
    top_cam.lookat = np.array([center[0], center[1], 0.0])
    top_cam.distance = max(3.0, span) * 1.8
    top_cam.azimuth = 90.0
    top_cam.elevation = -90.0

    fps = 30
    dt = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
    control_step_freq = 50
    step_ctr = 0

    current_ctrl = np.zeros(model.nu)
    trajectories: list[list[np.ndarray]] = [[] for _ in range(num_robots)]
    times_to_target: list[float | None] = [None] * num_robots

    fovy_rad = np.deg2rad(float(model.vis.global_.fovy))
    px_per_m = video_H / (2.0 * top_cam.distance * np.tan(fovy_rad / 2.0))

    def to_px(wx, wy):
        u = int(round(video_W / 2 + (wx - center[0]) * px_per_m))
        v = int(round(video_H / 2 - (wy - center[1]) * px_per_m))
        return u, v

    viewer_ctx = (
        mujoco.viewer.launch_passive(model, data)
        if args.live else None
    )
    if viewer_ctx is not None:
        console.log("[cyan]Live viewer opened. Close the window to abort early.[/cyan]")

    wall_start = None
    if args.live:
        import time as _time
        wall_start = _time.time()

    with mujoco.Renderer(model, height=video_H, width=video_W) as renderer:
        while data.time < DURATION:
            if viewer_ctx is not None and not viewer_ctx.is_running():
                console.log("[yellow]Viewer closed early — stopping race.[/yellow]")
                break
            for _ in range(steps_per_frame):
                if step_ctr % control_step_freq == 0:
                    # Compute per-robot action, write into ctrl
                    for r in range(num_robots):
                        qpos_r = data.qpos[q_slices[r]]
                        state_r = per_robot_state(qpos_r)

                        if cams[r] is not None:
                            vision_renderers[r].update_scene(data, camera=cams[r])
                            img = vision_renderers[r].render()
                            vision = analyze_sections(isolate_green(img))
                        else:
                            vision = [0.0, 0.0, 0.0]

                        phase = [
                            2 * np.sin(data.time * 2.0 * np.pi),
                            2 * np.cos(data.time * 2.0 * np.pi),
                        ]
                        state = np.concatenate([state_r, vision, phase]
                                               ).astype(np.float32)
                        current_ctrl[c_slices[r]] = nets[r].forward(state)

                np.copyto(data.ctrl, current_ctrl)
                mujoco.mj_step(model, data)
                step_ctr += 1

                # Track finish-line crossing (y ≤ FINISH_Y when racing south).
                for r in range(num_robots):
                    if times_to_target[r] is not None:
                        continue
                    y = float(data.qpos[q_slices[r].start + 1])
                    crossed = (y <= FINISH_Y) if FINISH_Y < START_Y else (y >= FINISH_Y)
                    if crossed:
                        times_to_target[r] = float(data.time)
                        console.log(
                            f"[bold]{ROBOT_LABELS[r]} crossed finish at "
                            f"t={data.time:.2f}s[/bold]"
                        )

            # Log positions this frame
            for r in range(num_robots):
                xy = data.qpos[q_slices[r].start:q_slices[r].start + 2].copy()
                trajectories[r].append(xy)

            renderer.update_scene(data, scene_option=viz, camera=top_cam)
            frame = renderer.render().copy()

            # Overlay trajectories & markers
            for r in range(num_robots):
                color = ROBOT_COLORS[r]
                traj = trajectories[r]
                if len(traj) >= 2:
                    pts = np.array([to_px(p[0], p[1]) for p in traj],
                                   dtype=np.int32)
                    cv2.polylines(frame, [pts], False, color, 2)
                sx, sy = to_px(starts[r][0], starts[r][1])
                cv2.circle(frame, (sx, sy), 5, color, -1)
                cx_, cy_ = to_px(traj[-1][0], traj[-1][1])
                cv2.putText(frame, ROBOT_LABELS[r], (cx_ + 8, cy_ - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Start line (white) and finish line (red) drawn across the track
            x_left = LANE_XS[0] - LANE_SPACING
            x_right = LANE_XS[-1] + LANE_SPACING
            sl_l = to_px(x_left, START_Y); sl_r = to_px(x_right, START_Y)
            fl_l = to_px(x_left, FINISH_Y); fl_r = to_px(x_right, FINISH_Y)
            cv2.line(frame, sl_l, sl_r, (255, 255, 255), 2)
            cv2.line(frame, fl_l, fl_r, (0, 0, 255), 2)
            cv2.putText(frame, "START", (sl_l[0] + 6, sl_l[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "FINISH", (fl_l[0] + 6, fl_l[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            video_recorder.write(frame=frame)

            if viewer_ctx is not None:
                viewer_ctx.sync()
                # Pace loop to wall-clock so the live view isn't warp-speed.
                target_wall = wall_start + data.time
                lag = target_wall - _time.time()
                if lag > 0:
                    _time.sleep(lag)

        video_recorder.release()
        if viewer_ctx is not None:
            viewer_ctx.close()

    for rndr in vision_renderers:
        rndr.close()

    # ---- Standings ----
    console.rule("Race results")
    finished = [(r, t) for r, t in enumerate(times_to_target) if t is not None]
    finished.sort(key=lambda x: x[1])
    if not finished:
        console.log("No robot crossed the finish line.")
    for rank, (r, t) in enumerate(finished, start=1):
        console.log(f"  {rank}. {ROBOT_LABELS[r]}  t={t:.2f}s")
    unfinished = [r for r in range(num_robots) if times_to_target[r] is None]
    for r in unfinished:
        final_y = float(trajectories[r][-1][1])
        remaining = abs(final_y - FINISH_Y)
        console.log(f"  DNF {ROBOT_LABELS[r]}  y={final_y:.2f}  remaining={remaining:.2f}m")

    # ---- Trajectory plot ----
    plt.figure(figsize=(8, 8))
    for r in range(num_robots):
        xs = [p[0] for p in trajectories[r]]
        ys = [p[1] for p in trajectories[r]]
        c = tuple(v / 255 for v in ROBOT_COLORS[r])
        plt.plot(xs, ys, "-", color=c, linewidth=2, label=ROBOT_LABELS[r])
        plt.plot(starts[r][0], starts[r][1], "o", color=c, markersize=10)
    x_left = LANE_XS[0] - LANE_SPACING
    x_right = LANE_XS[-1] + LANE_SPACING
    plt.plot([x_left, x_right], [START_Y, START_Y], "k--", label="Start")
    plt.plot([x_left, x_right], [FINISH_Y, FINISH_Y], "r-", linewidth=2, label="Finish")
    plt.title("Racing demo trajectories")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.axis("equal"); plt.grid(True); plt.legend()
    plot_path = DATA / "race_trajectories.png"
    plt.savefig(plot_path)
    console.log(f"[green]Saved plot to {plot_path}[/green]")


if __name__ == "__main__":
    main()
