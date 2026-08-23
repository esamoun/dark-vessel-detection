"""The command line, reached the way the paid machine reaches it.

`pyproject.toml` declares a `darkvessel` console script, and every instruction in this
repository used to name it. A console script is installed into whichever directory the
installer chose, and whether that directory is on PATH is a property of the machine, not of
this package: on Kaggle it is not, so `pip install -e` succeeded, `import darkvessel` succeeded,
and `!darkvessel train` answered `command not found` at the training cell — after the dataset
was attached and the wheels were built.

`python -m darkvessel` is resolved by the interpreter rather than by the shell, which is why the
notebook now calls it that way and why this file exists: the module has to stay executable, and
it has to stay wired to the same `main` the console script names.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_package_runs_as_a_module_without_a_console_script_on_the_path() -> None:
    """`python -m darkvessel` parses and reports the same command line.

    A subprocess rather than an import, because `__main__.py` ends in `sys.exit`: importing it
    would end the test run. `--help` is chosen for being the one invocation that needs no
    config, no data and no GPU, so this runs on a laptop and in CI.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "darkvessel", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("usage: darkvessel")
    # The subcommand the notebook calls, so that a module that runs but reaches a different
    # parser is not mistaken for this one working.
    assert "train" in completed.stdout


def test_the_module_and_the_console_script_name_the_same_entry_point() -> None:
    """Both routes lead to `darkvessel.cli:main`.

    The module exists so the notebook need not depend on a PATH; it must not become a second,
    quietly divergent command line. Pinning the console script's target as well is what stops
    the two drifting apart — a laptop types one and Kaggle runs the other, so a difference
    between them would show up only on the machine that costs money.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"]["darkvessel"] == "darkvessel.cli:main"

    source = (ROOT / "src" / "darkvessel" / "__main__.py").read_text()
    assert "from darkvessel.cli import main" in source
    assert "sys.exit(main())" in source
