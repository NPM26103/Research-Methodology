"""One master seed, with deterministic streams for epochs and noise replicates.

Using three copies of exactly the same Gaussian stream would not provide three
replicates. Stream identifiers produce distinct, repeatable draws while the
experiment's only configurable master seed remains 26.
"""

import hashlib
import json

MASTER_SEED = 26


def derived_seed(master_seed: int, *stream_identifiers) -> int:
    """Return a stable 63-bit Torch seed without Python's randomized hash()."""
    payload = json.dumps([int(master_seed), *stream_identifiers], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)
