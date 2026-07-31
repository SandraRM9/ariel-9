"""Global MuJoCo compiler / solver / option settings for ARIEL worlds.

Every field here is applied by
:meth:`ariel.simulation.environments._base_world.BaseWorld._init_spec`. If you
add a field, apply it there too — an unapplied setting is worse than no setting,
because it reads as configured while the simulation silently uses a default.

Notes
-----
    * ``timestep`` used to be declared as ``0.02`` here but was never written
      to ``spec.option``, so every simulation actually ran at MuJoCo's 0.002 s
      default. Applying 0.02 s (50 Hz) would have made contacts explode. The
      value below is the one that was really in use.
    * ``balanceinertia`` silently rewrites any inertia that violates the
      triangle inequality instead of raising. That hides modelling errors
      (e.g. a body given a mass but no geom), so it now defaults to off.

References
----------
    [1] https://mujoco.readthedocs.io/en/stable/XMLreference.html#option
    [2] https://mujoco.readthedocs.io/en/stable/computation/index.html#solver-parameters

Todo
----
    [ ] Sweep ``impratio`` / ``friction`` against measured slip on the real
        arena surface.

"""

# Standard library
from pathlib import Path

# Third-party libraries
import mujoco
import numpy as np
from pydantic_settings import BaseSettings

# Local libraries
from ariel.parameters.units import QUANTITY_MODEL_CONFIG, Time, si, u

# Local libraries
# Global constants
# Global functions
# Warning Control
# Type Checking
# Type Aliases
# Type Aliases
type WeightType = float
type DimensionType = tuple[float, float, float]

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)


class MujocoConfig(BaseSettings):
    model_config = QUANTITY_MODEL_CONFIG

    # --- Compiler --- #
    # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjscompiler
    # https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html#compiler
    autolimits: bool = True
    balanceinertia: bool = False  # fail loudly on bad inertias, do not patch
    degree: bool = False

    # --- Option --- #
    # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjoption
    # https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html#option
    timestep: Time = 2 * u.ms  # 500 Hz
    integrator: int = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    solver: int = int(mujoco.mjtSolver.mjSOL_NEWTON)
    iterations: int = 100
    ls_iterations: int = 50

    # Ratio of frictional-to-normal constraint impedance. MuJoCo's default of
    # 1 lets a light robot slip against the floor almost for free, which
    # optimisers reliably discover. 10 was an over-correction: combined with a
    # friction coefficient of 1.0 it made the feet essentially non-slipping,
    # so every gait worked better than it does on a real floor. 3 removes the
    # numerical slip without making grip ideal.
    impratio: float = 3.0
    cone: int = int(mujoco.mjtCone.mjCONE_PYRAMIDAL)

    # --- Contact defaults --- #
    # solref[0] is the contact time constant and must stay above 2 * timestep
    # or the solver goes unstable. Peak interpenetration scales with it: at the
    # 0.02 s default a limb swinging at 1 m/s buries ~2 cm into the body before
    # the contact bites. 0.005 s (2.5 timesteps) roughly halves that again
    # relative to 0.01 s, with no added solver warnings.
    geom_solref_timeconst: Time = 5 * u.ms
    geom_solref_dampratio: float = 1.0
    geom_solimp: tuple[float, float, float, float, float] = (
        0.9,
        0.95,
        0.001,
        0.5,
        2.0,
    )
    # (sliding, torsional, rolling). MuJoCo's 1.0 sliding default is
    # rubber-like; measured PLA on a printed/textured arena floor is nearer
    # 0.45. The torsional term is raised from the 0.005 default because flat
    # printed feet resist spinning in place far more than that implies.
    geom_friction: tuple[float, float, float] = (0.45, 0.05, 0.0001)
    geom_condim: int = 3

    # --- Self-collision --- #
    # Robot geoms are put on their own contype/conaffinity bit so that
    # robot-robot / robot-self contacts can be switched off wholesale without
    # touching the floor. Adjacent (parent-child) links are already filtered
    # by MuJoCo; this controls the rest of the body.
    enable_self_collision: bool = True

    # --- Box-box collider workaround --- #
    # MuJoCo's analytic box<->box collider (mjc_BoxBox) returns a grossly wrong
    # penetration depth for two boxes that share a parallel axis and touch at
    # less than ~0.05 mm. Measured on an ARIEL body: a pair whose true
    # separating-axis overlap was 0.018 mm was reported as 76.458 mm deep. The
    # solver then ejects that "overlap", injecting ~39 J in a single 2 ms step
    # and throwing the robot metres into the air. Reproduced in a bare two-geom
    # model on every MuJoCo from 3.2.7 to 3.8.0, so it is not a regression and
    # upgrading does not help.
    #
    # ARIEL bodies hit this constantly because every module is a cuboid and
    # attachment rotations are multiples of 90 degrees, so limbs routinely rest
    # face-to-face with exactly parallel axes.
    #
    # Only the box<->box code path is affected. Replacing the module boxes with
    # equivalent 8-vertex convex meshes routes the pair through the general
    # convex collider, which returns the correct depth. Mass is unchanged;
    # inertia differs by ~0.2% because MuJoCo integrates it over the mesh.
    #
    # Cost, measured interleaved and best-of-N: simulation stepping +0.8%
    # (27.35 -> 27.56 us/step, inside the per-body noise), model construction
    # +5% (~3 ms per body, one-off). Narrow-phase collision itself is ~30%
    # dearer, but it is a minority of a step and does not surface end to end.
    #
    # Set to False to restore raw box geoms (and the bug).
    convert_boxes_to_meshes: bool = True

    # --- Visual --- #
    offheight: int = 960
    offwidth: int = 1280

    # --- Default Geom --- #
    floor_name: str = "floor"

    # --- Derived SI values (what MuJoCo wants) --- #
    @property
    def timestep_si(self) -> float:
        """Integration timestep in seconds."""
        return si(self.timestep, u.s)

    @property
    def geom_solref(self) -> tuple[float, float]:
        """Contact ``solref`` as MuJoCo wants it: (seconds, dampratio)."""
        return (
            si(self.geom_solref_timeconst, u.s),
            self.geom_solref_dampratio,
        )
