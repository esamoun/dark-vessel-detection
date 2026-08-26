"""The evaluation report of issue #12, held against the run it reports on.

A report is prose, and prose drifts: a number is transcribed once, the journal it came from is
re-run, and the document goes on saying what used to be true with no test anywhere going red.
This project already has a rule that a threshold chosen after seeing the numbers is not a
threshold; the same argument applies to a table typed after reading them.

So the table and the figure in `docs/evaluation.md` are pinned to what `darkvessel evaluate`
produces from the committed journal — and from the *right* one: R1 was executed twice and the
report describes the execution whose weights the chain loads, not the sibling the ladder judged.

What is *not* pinned is anything the report says in words — that is the author's job, and a test
that tried would only pin the author's opinion of the day.
"""

import json
import re
from pathlib import Path

from darkvessel.detect.curve import WINDOW, curve, svg
from darkvessel.detect.curve import table as curve_table

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "evaluation.md"
FIGURE = ROOT / "docs" / "figures" / "precision-recall-r1.svg"
JOURNAL = ROOT / "docs" / "runs" / "r1-cosine-rerun.json"


def _points() -> list:
    return curve(json.loads(JOURNAL.read_text())["epochs"], window=WINDOW)


def _table_in(text: str) -> str:
    """The one markdown table in the report, with emphasis stripped.

    Bold marks a row a reader should look at first and is the author's to place; the digits under
    it are not. Stripping the asterisks is what lets both be true of the same table.
    """
    plain = re.sub(r"\*", "", text)
    rows = [line.strip() for line in plain.splitlines() if line.strip().startswith("| 0.")]
    header = next(line for line in plain.splitlines() if line.startswith("| Score threshold"))
    rule = "| --- | --- | --- | --- | --- | --- | --- |"
    return "\n".join([header, rule, *rows])


def test_the_reports_table_is_the_one_the_journal_produces() -> None:
    assert _table_in(REPORT.read_text()) == curve_table(_points())


def test_the_reports_figure_is_the_one_the_journal_produces() -> None:
    """Regenerated rather than compared by eye. An SVG that has fallen a run behind is the most
    convincing wrong number in the repository: it is a picture, and nobody reads a picture twice.
    """
    assert FIGURE.read_text() == svg(_points()) + "\n"


def test_the_report_shows_the_figure_it_ships() -> None:
    assert f"figures/{FIGURE.name}" in REPORT.read_text()
