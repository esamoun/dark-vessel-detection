"""The Danish Maritime Authority's daily AIS archives, as a stream.

One zip per day, published without registration, containing one CSV of every position report
received in Danish waters that day. The day this project's first real scene falls in is 662 MB
compressed and 3.3 GB expanded — two orders of magnitude past what the study area and a quarter
of an hour need, and past what an 8 GB machine can hold either way.

So it is never held. The member is inflated straight off the response and handed to the ingestion
a chunk at a time, which filters each chunk and keeps what survives; what crosses the network is
a day of Danish AIS and what stays is a few thousand reports. This is the same constraint the
Sentinel-1 export is built around, arriving from the other side of the chain.

The archive is a parameter of the ingestion rather than an import inside it — the third use of
the seam that lets the detector and the catalogue be substituted. Everything decidable without a
network is decided on this side of the boundary and tested there; what is left here reads bytes
and decides nothing.
"""

import io
import struct
import urllib.request
import zlib
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import date
from typing import IO, Protocol

# The host `dma.dk` sends a reader to. The archives used to be served from `web.ais.dk`, which is
# still what most published code points at and no longer answers: its certificate expired in June
# 2025 and the connection is reset after the request. Written out here rather than left in a
# comment, because a URL that has moved once will move again and this is the line to change.
ARCHIVE_HOST = "http://aisdata.ais.dk.s3.eu-central-1.amazonaws.com"

# Generous: a day's archive is most of a gigabyte, and a slow link is a slow link rather than a
# fault. What this guards against is a socket that has stopped answering altogether.
READ_TIMEOUT_S = 300

# What is read off the socket at a time before being inflated. Immaterial to the answer; it only
# decides how often the loop goes round.
_CHUNK_BYTES = 1 << 20

_LOCAL_FILE_HEADER = 0x04034B50
_DEFLATED = 8


class Archive(Protocol):
    """What the ingestion needs of the Danish Maritime Authority, and nothing beyond it."""

    def open_day(self, day: date) -> AbstractContextManager[IO[bytes]]:
        """The day's position reports, as a stream of CSV bytes, open for as long as it is held."""
        ...


def archive_url(day: date) -> str:
    """Where the archive for one day lives."""
    return f"{ARCHIVE_HOST}/aisdk-{day.isoformat()}.zip"


def danish_maritime_authority() -> Archive:
    """The real archive, over the network.

    The only code here that opens a socket, and the only part a test cannot reach. It is kept
    this thin for that reason: it builds a URL, checks that what came back is the day that was
    asked for, and hands over the bytes. Everything that decides what the answer is happens in
    `ais.py`, against a fake.
    """
    return _DanishMaritimeAuthority()


class _DanishMaritimeAuthority:
    @contextmanager
    def open_day(self, day: date) -> Iterator[IO[bytes]]:
        url = archive_url(day)
        with urllib.request.urlopen(url, timeout=READ_TIMEOUT_S) as response:  # noqa: S310
            name, member = zip_member(response)
            _refuse_an_archive_for_another_day(name, day, url)
            yield member


def _refuse_an_archive_for_another_day(name: str, day: date, url: str) -> None:
    """Check the file inside the zip is the day the URL asked for.

    A server that answers every path with the same file, or a naming convention that has changed
    under us, produces an archive whose reports all fall outside the window. That comes back as
    an empty slice — a search that ran and found nothing — and every detection in the scene is
    then honestly, and wrongly, dark. The one cheap check against it is the name the archive
    gives itself.
    """
    if day.isoformat() not in name:
        raise ValueError(
            f"{url} contains {name!r}, which does not name {day.isoformat()}; the archive's "
            "layout has changed, and filtering the wrong day to this window would return an "
            "empty slice rather than an error"
        )


def zip_member(stream: IO[bytes]) -> tuple[str, IO[bytes]]:
    """The first member of a zip, named, and inflated as it arrives rather than after.

    `zipfile` reads the central directory at the end of the file, so it needs somewhere seekable
    — which means the whole 662 MB archive on disk before the first row can be parsed. A daily
    archive holds exactly one member, and the local header standing at the front of the stream
    says everything needed to inflate it, so nothing has to be stored to read it.

    The uncompressed size in that header is deliberately not used: on a zip written in streaming
    mode it is zero and the true size lives in a descriptor after the data. Inflation stops when
    the deflate stream says it has ended, which is true in both cases.
    """
    header = _read_exactly(stream, 30)
    signature = struct.unpack("<I", header[0:4])[0]
    method = struct.unpack("<H", header[8:10])[0]
    name_length, extra_length = struct.unpack("<HH", header[26:30])

    if signature != _LOCAL_FILE_HEADER:
        raise ValueError(
            f"expected a zip archive, got something starting {signature:#010x}; the archive host "
            "answers a path it does not have with an error page rather than a 404"
        )
    # Only deflate. A stored member would have to be read by the length in the header, which is
    # the one field here that cannot be trusted — and reading past it hands the central directory
    # to the CSV parser as if it were rows. Refusing is the honest answer to a case that has
    # never occurred: every archive the Danish Maritime Authority publishes is deflated.
    if method != _DEFLATED:
        raise ValueError(
            f"the archive's member is stored with compression method {method} rather than "
            "deflate, which this reader does not handle"
        )

    name = _read_exactly(stream, name_length).decode("utf-8", "replace")
    _read_exactly(stream, extra_length)

    return name, io.BufferedReader(_Inflating(stream))


def _read_exactly(stream: IO[bytes], count: int) -> bytes:
    """`count` bytes, however many reads that takes.

    A socket answers with whatever has arrived, not with what was asked for. A single `read` of
    the header is right almost always, and the time it is not, every field after it is garbage.
    """
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ValueError(f"the archive ended {remaining} bytes into a {count}-byte header")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _Inflating(io.RawIOBase):
    """One deflate stream, decompressed as its bytes arrive.

    Readable once, forwards only, exactly like the response it wraps. Wrapped in a
    `BufferedReader` by the caller so that pandas gets the file-like object it expects.
    """

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        # A negative window size is what says "a raw deflate stream, with no zlib wrapper around
        # it", which is what a zip member is.
        self._inflate = zlib.decompressobj(-zlib.MAX_WBITS)
        self._pending = b""
        self._ended = False

    def readable(self) -> bool:
        return True

    def readinto(self, target: memoryview) -> int:  # type: ignore[override]
        while not self._pending and not self._ended:
            compressed = self._stream.read(_CHUNK_BYTES)
            if not compressed:
                self._pending, self._ended = self._inflate.flush(), True
            else:
                self._pending = self._inflate.decompress(compressed)
                # Past the end of the member lie the descriptor and the central directory. They
                # are not part of the CSV and inflating stops here rather than reading them.
                self._ended = self._inflate.eof

        taken = min(len(target), len(self._pending))
        target[:taken] = self._pending[:taken]
        self._pending = self._pending[taken:]
        return taken
