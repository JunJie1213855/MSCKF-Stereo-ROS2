#!/usr/bin/env python3
"""Run an evo command with the mpl_toolkits import repaired.

Workaround for a matplotlib conflict on this machine, needed only for evo's
plotting; the numbers reported by evo are correct without it.

Debian ships /usr/lib/python3/dist-packages/matplotlib-3.5.1-nspkg.pth, a
setuptools namespace .pth that pre-registers `mpl_toolkits` in sys.modules at
interpreter startup and pins its __path__ to the apt matplotlib 3.5.1 tree.
`import matplotlib` meanwhile resolves to the pip matplotlib 3.10.9 in ~/.local.
Mixing 3.5.1 submodules with 3.10.9 internals gives:

    ImportError: cannot import name 'docstring' from 'matplotlib'

Because the pin happens in a .pth, before any user code, reinstalling matplotlib
from pip does not help. Repointing __path__ after startup does, and that is all
this wrapper does before handing over to evo's own CLI.

Usage (same arguments as the evo command, with the command named first):

    evo_run.py ape  tum gt.txt est.txt -a --save_plot ape.png
    evo_run.py rpe  tum gt.txt est.txt -a -d 1 -u m --save_plot rpe.png
    evo_run.py traj tum gt.txt est.txt --save_plot traj.png

Use --save_plot, not --plot: showing the plot needs PyQt6, which is not
installed. evo appends _raw / _map to the filename it is given.
"""
import os
import sys

import mpl_toolkits

# Prefer the pip matplotlib's own mpl_toolkits over the pinned apt 3.5.1 copy.
for _base in (sys.path + [os.path.join(os.path.expanduser("~"), ".local/lib")]):
    _cand = os.path.join(_base, "python3.10/site-packages/mpl_toolkits")
    if os.path.isdir(_cand) and "dist-packages" not in _cand:
        mpl_toolkits.__path__ = [_cand]
        break

from evo.cli import entry_points  # noqa: E402

if len(sys.argv) < 2 or sys.argv[1] not in ("ape", "rpe", "traj"):
    sys.exit(__doc__)

app = sys.argv[1]
sys.argv = ["evo_" + app] + sys.argv[2:]
getattr(entry_points, app)()
