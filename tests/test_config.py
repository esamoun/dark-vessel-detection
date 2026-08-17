"""Reading a config, and the config it stands on.

The ladder in issue #11 is five runs that differ by one line each. A rung that quietly differs by
two lines measures two things and reports one, so the mechanism that lets a rung state only its
own change is load-bearing rather than a convenience — and its failure modes are a cycle, a base
that is not there, and a merge that drops a key.
"""

from pathlib import Path

import pytest

from darkvessel.config import load_config


def test_a_config_that_extends_nothing_is_read_as_it_stands(tmp_path: Path) -> None:
    path = tmp_path / "base.yaml"
    path.write_text("schedule:\n  epochs: 12\n  learning_rate: 0.005\n")

    assert load_config(path) == {"schedule": {"epochs": 12, "learning_rate": 0.005}}


def test_a_rung_states_only_what_it_changes(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "schedule:\n  epochs: 12\n  learning_rate: 0.005\nmodel:\n  stem: repeat\n"
    )
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\nschedule:\n  lr_schedule: cosine\n")

    assert load_config(rung) == {
        "schedule": {"epochs": 12, "learning_rate": 0.005, "lr_schedule": "cosine"},
        "model": {"stem": "repeat"},
    }


def test_a_rung_overrides_a_value_the_base_already_set(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("model:\n  anchor_sizes: [[32], [64]]\n")
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\nmodel:\n  anchor_sizes: [[4], [8]]\n")

    assert load_config(rung)["model"]["anchor_sizes"] == [[4], [8]]


def test_a_list_is_replaced_and_never_merged(tmp_path: Path) -> None:
    """A merged list would make `anchor_sizes: [[4], [8]]` mean nine levels rather than two, and
    the run would train on something no file states."""
    (tmp_path / "base.yaml").write_text("reporting:\n  thresholds: [0.05, 0.5, 0.9]\n")
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\nreporting:\n  thresholds: [0.75]\n")

    assert load_config(rung)["reporting"]["thresholds"] == [0.75]


def test_a_chain_of_rungs_resolves_through_to_the_base(tmp_path: Path) -> None:
    """The ladder is greedy: rung 2 extends rung 1, which extends the baseline."""
    (tmp_path / "base.yaml").write_text("schedule:\n  epochs: 12\n")
    (tmp_path / "r1.yaml").write_text("extends: base.yaml\nschedule:\n  lr_schedule: cosine\n")
    r2 = tmp_path / "r2.yaml"
    r2.write_text("extends: r1.yaml\nmodel:\n  anchor_sizes: [[4]]\n")

    assert load_config(r2) == {
        "schedule": {"epochs": 12, "lr_schedule": "cosine"},
        "model": {"anchor_sizes": [[4]]},
    }


def test_the_base_is_found_beside_the_file_that_names_it(tmp_path: Path) -> None:
    """Every path in this project is read relative to the config that declares it, and the base a
    rung extends is a path like any other. Rung configs live one directory down from the base."""
    (tmp_path / "train.yaml").write_text("schedule:\n  epochs: 12\n")
    (tmp_path / "ladder").mkdir()
    rung = tmp_path / "ladder" / "r1.yaml"
    rung.write_text("extends: ../train.yaml\nschedule:\n  lr_schedule: cosine\n")

    assert load_config(rung)["schedule"] == {"epochs": 12, "lr_schedule": "cosine"}


def test_a_config_that_extends_itself_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text("extends: loop.yaml\nschedule:\n  epochs: 12\n")

    with pytest.raises(ValueError, match="extends itself"):
        load_config(path)


def test_a_cycle_between_two_configs_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")

    with pytest.raises(ValueError, match="extends itself"):
        load_config(tmp_path / "a.yaml")


def test_a_base_that_is_not_there_is_refused_by_name(tmp_path: Path) -> None:
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: ../nowhere/train.yaml\n")

    with pytest.raises(FileNotFoundError, match="train.yaml"):
        load_config(rung)


def test_the_extends_key_does_not_survive_into_the_config(tmp_path: Path) -> None:
    """Whatever reads the result should not have to know the file was assembled from two."""
    (tmp_path / "base.yaml").write_text("schedule:\n  epochs: 12\n")
    rung = tmp_path / "rung.yaml"
    rung.write_text("extends: base.yaml\n")

    assert "extends" not in load_config(rung)
