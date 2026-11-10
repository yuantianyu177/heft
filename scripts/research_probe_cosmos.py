"""Use the same correspondence probe with Cosmos's verified phase convention."""
import sys

import research_probe
from research_cosmos_rope import cosmos_angles

research_probe.wan_angles = cosmos_angles
research_probe.VARIANTS = [*research_probe.VARIANTS,
    ("post_low10", 1, 1, .1, None),
    ("content_low10", 0, 0, .1, None),
    ("high10_phase_removed", 0, 0, 0, .1),
]
if "--features" not in sys.argv:
    sys.argv.extend(["--features", "cosmos_real_t25"])
if __name__ == "__main__":
    research_probe.main()
