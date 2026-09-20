"""Fly-Brain Flow — a crowd simulation where every agent runs a real fruit fly brain.

Package layout (grows with the milestones — see docs/PLAN.md):

    flybrainflow.scenario   TOML scenario files -> typed config          (M0)
    flybrainflow.world      maps, odour and wind fields, physics          (M0)
    flybrainflow.agents     spawn / feed / leave, personalities           (M0)
    flybrainflow.brains     toy connectome, baseline, real brain          (M0 / M1)
    flybrainflow.sim        the loop and the recorder                     (M0)
"""

__version__ = "0.0.1"
