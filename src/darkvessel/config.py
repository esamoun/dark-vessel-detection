"""Reading a config file, and the config file it stands on.

Every stage of this project is defined by one YAML file, and that is deliberate: a run is
reproducible from a file rather than from a sequence of cells executed in the right order. The
ladder in issue #11 puts a strain on it. Five training runs differ from one another by a single
line each, and five standalone copies of a ninety-line file would hide that line in a diff and
would let a second line drift without anyone noticing — which would break the one property the
ladder depends on, that each rung changes exactly one thing.

So a config may name the config it extends, and state only its own difference. Dicts are merged
key by key; anything else replaces. Lists in particular replace rather than concatenate, because
`anchor_sizes` merged with its base would silently mean nine pyramid levels rather than five, and
the run would then be training on something no file states.

Nothing here knows what a config *means*. Which keys exist and what they have to contain is the
business of the `*_request_from` functions in `cli.py`, and keeping that out of here is what lets
this module be read in one sitting.
"""

from pathlib import Path
from typing import Any

import yaml

EXTENDS = "extends"


def load_config(path: Path) -> dict[str, Any]:
    """One config, with everything it inherits already folded in.

    The `extends` key does not survive: whatever reads the result should not have to know that
    the file was assembled from two.
    """
    return _load(path.resolve(), ())


def _load(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    if path in seen:
        chain = " -> ".join(step.name for step in (*seen, path))
        raise ValueError(f"a config extends itself, directly or through a chain: {chain}")

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist, and a config extends it")

    config = yaml.safe_load(path.read_text()) or {}
    base = config.pop(EXTENDS, None)
    if base is None:
        return config

    # Relative to the file that names it, which is the rule this project applies to every other
    # path in every other config.
    return _merge(_load((path.parent / base).resolve(), (*seen, path)), config)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`over` wins, key by key, and two dicts at the same key are merged rather than replaced."""
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
