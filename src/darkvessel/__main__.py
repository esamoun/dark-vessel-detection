"""`python -m darkvessel`, the entry point that does not go through a PATH.

`pyproject.toml` also installs a `darkvessel` console script, and on a laptop that is the one
to type. On Kaggle it is not there to type: `pip install -e` writes the script into a directory
the session's shell does not carry, so a notebook whose package imports perfectly answers
`darkvessel: command not found` — at the training cell, with the dataset attached and the wheels
already installed, which is the most expensive place in the run to discover a PATH.

`python -m` is resolved by the interpreter that holds the package rather than by the shell, so
it cannot miss for that reason. It is what `notebooks/kaggle-train.ipynb` calls.
"""

import sys

from darkvessel.cli import main

sys.exit(main())
