"""Clustering over the embedding space.

Moved to `structures.py`, which is where the question that wanted it is answered. This file said,
until 2026-08-27, that separating offshore wind turbines from vessels was not implemented because
the archive held no turbines to separate. The archive holds them now, the separation is
implemented, and what it found is that the clustering is not the part that does the work:
`structures.cluster` fits it, `structures.describe` and `structures.separation` measure it, and
the exclusion is built on positional recurrence instead. See docs/decisions.md, 2026-08-27.

Kept as a name rather than deleted, because `docs/decisions.md` and `docs/failures.md` refer to
this module by it and a reader following one of those references should land somewhere that says
where the code went.
"""
