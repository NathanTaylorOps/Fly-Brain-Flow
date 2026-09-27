"""Brains: whatever decides which way an agent wants to go this tick.

Every brain in here (toy connectome, baseline, the real connectome later) exposes the same
shape of thing: given who's walking, where they are, and how big they are, hand back a
`desired_velocities` array that `world.physics.step()` can move them with. `physics.step()`
doesn't know or care which brain produced it -- that's what makes the fly-brained vs. baseline
comparison a comparison of steering decisions, not of two different simulators.

`ConnectomeBrain` (Step 4's real brain, see `connectome.py` -- currently an interface skeleton, not
working yet) is deliberately NOT imported here, unlike `Baseline`/`ToyBrain` below: it imports
`flybrainflow.spiking`, which imports `torch`, an optional `brain` extra (see pyproject.toml) that
M0's tests and everything else in this package run fine without. Importing it here would make
`import flybrainflow.brains` require torch unconditionally for everyone, even code that only ever
touches `Baseline`/`ToyBrain`. Import it directly (`from flybrainflow.brains.connectome import
ConnectomeBrain`) wherever torch is already a given.

    brain = Baseline(walkable_map, target_positions=[t.position for t in scenario.targets])
    # each tick, after spawn/set_positions from the previous step:
    vel = brain.desired_velocities(ids, positions, radii, max_speed_mps=1.3)
    new_pos, actual_vel = physics_step(positions, vel, radii, dt, walkable_map)
"""

from __future__ import annotations

from .baseline import Baseline
from .toy import ToyBrain, placeholder_personality_table

__all__ = ["Baseline", "ToyBrain", "placeholder_personality_table"]
