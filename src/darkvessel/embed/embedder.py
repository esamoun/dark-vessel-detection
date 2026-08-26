"""The embedder contract, and how a vector travels with a detection.

The shape `detector.py` has, for the same reason: the pipeline that calls an embedder and the
code that reads the vectors back out of a layer both depend on this, and neither has to depend on
the other. The implementation that satisfies it lives in `contrastive.py`, behind torch; nothing
here imports the framework, so a chain that was never asked for an embedding never learns that
one exists.

A vector travels in the layer, in one column per dimension, rather than in a file beside it. That
is a decision about what a detection *is* by the time the chain has finished with it: an
archive of embeddings kept separately is a second thing to keep in step with the first, and the
join between them would be a row order nobody stated. Sixteen columns is a wide attribute table
and a cheap one — the alternative is a sidecar whose rows can silently stop corresponding to
anything.
"""

from typing import Protocol

import geopandas as gpd
import numpy as np

# What each column of a vector is called in the layer. Zero-padded, so that an attribute table
# and a `sorted()` put e02 before e10 — the same reason a checkpoint is `epoch-007.pt`.
PREFIX = "e"


class Embedder(Protocol):
    """Anything that turns detection crops into vectors. The optional injected dependency.

    It states the geometry it was fitted at rather than being told it. A detector has to be told
    the tile size and `cli.check_tile_size` refuses a run that disagrees with it; here there is
    nothing to disagree about, because the only thing that knows what window an encoder was
    trained to be shown is the encoder.

    Takes the crops as `crops.crops_for` cuts them — margin included, unaugmented — and returns
    one row per crop.
    """

    crop_px: int
    margin_px: int

    def __call__(self, crops: np.ndarray) -> np.ndarray: ...


def columns(dim: int) -> list[str]:
    """The names of the columns a `dim`-dimensional embedding occupies."""
    if dim <= 0:
        raise ValueError(f"an embedding of {dim} dimensions is not an embedding")
    width = max(2, len(str(dim - 1)))
    return [f"{PREFIX}{index:0{width}d}" for index in range(dim)]


def attach(detections: gpd.GeoDataFrame, vectors: np.ndarray) -> gpd.GeoDataFrame:
    """Put one vector on each detection, by position.

    By position because that is the only correspondence there is: the crops were cut from the
    detections in the order the chain reports them and nothing re-sorted in between. The length
    is checked rather than trusted — a mismatch here would not raise on its own, it would attach
    every vector to the wrong vessel and still write a layer that opens.
    """
    if len(vectors) != len(detections):
        raise ValueError(
            f"{len(vectors)} vectors for {len(detections)} detections; attached by position, so "
            "a mismatch would put each vector on the wrong row rather than on none"
        )

    embedded = detections.copy()
    if len(detections) == 0:
        # An empty frame still has to carry the columns, or a scene that found nothing writes a
        # layer with a different schema from the one beside it and the archive stops stacking.
        for name in columns(vectors.shape[1] if vectors.ndim == 2 else 0):
            embedded[name] = np.empty(0, dtype=np.float32)
        return embedded

    for name, values in zip(columns(vectors.shape[1]), vectors.T, strict=True):
        embedded[name] = values.astype(np.float32)
    return embedded


def vectors_of(layer: gpd.GeoDataFrame) -> np.ndarray:
    """The embeddings a layer carries, as one array. Empty of columns where it carries none.

    The columns are taken in name order rather than in the order the frame happens to hold them:
    a GeoPackage read back does not promise the column order it was written with, and a vector
    whose dimensions have been permuted is still a vector — it simply answers a different
    question, silently.
    """
    names = sorted(name for name in layer.columns if _is_dimension(name))
    return layer[names].to_numpy(dtype=np.float32) if names else np.empty((len(layer), 0))


def _is_dimension(name: str) -> bool:
    return name.startswith(PREFIX) and name[len(PREFIX) :].isdigit()
