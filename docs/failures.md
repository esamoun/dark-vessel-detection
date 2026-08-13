# Failure log

What was tried and did not work, and what it cost. Kept deliberately: the dead ends are part of
the method, and a pipeline whose limits are known is more useful than one whose limits are not.

Each entry: what was attempted, what happened, why, and what was done instead.

---

## 2026-08-13 — `.gitignore` silently deleted a source package

**What happened.** `.gitignore` opened with an unanchored `data/`, meant for downloaded imagery
at the repository root. A gitignore pattern containing no slash matches at *every* level, so it
also matched `src/darkvessel/data/` — the package holding scene loading, AIS ingestion, tiling
and the synthetic inputs. Six source files were never tracked. Nothing complained: the working
copy was complete, the tests passed, and the scaffold commit that introduced the rule looked
clean.

**What it would have cost.** A clone of this repository could not have imported `darkvessel.cli`
at all. Confirmed by checking the staged tree out into an empty directory and running the suite
there — `ModuleNotFoundError: No module named 'darkvessel.data'`.

**Why it survived.** Every check ran against the working directory, which had the files. No
check ran against what was actually committed.

**What was done instead.** The rules are anchored to the repository root — `/data/`, `/outputs/`,
`/checkpoints/` — so they cannot reach into `src/`. The lesson generalises past this bug: a test
suite that only ever runs in the working directory cannot see what the repository is missing.
