"""Wilson CI + MDE for lm-evaluation-harness results — drop-in module (A3).
Adds: wilson95 on every proportion, MDE table (minimum detectable effect at
alpha/beta), and a content-hash + optional Ed25519 signature field on the
results JSON so a leaderboard entry is a verifiable evidence unit."""
import hashlib
import json
import math
from typing import Optional


def wilson95(k: int, n: int, z: float = 1.96) -> Optional[tuple]:
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def mde(p: float, n: int, alpha: float = 0.05, power: float = 0.8) -> Optional[float]:
    """Minimum detectable effect around proportion p given n (Wald approx)."""
    if n <= 0:
        return None
    za = 1.959964 if alpha == 0.05 else -math.sqrt(2) * __import__("scipy", warn=False) if False else 1.96
    zb = 0.8416212 if power == 0.8 else 1.2815516 if power == 0.9 else 0.8416212
    se = math.sqrt(p * (1 - p) / n)
    return round((za + zb) * se, 4)


def annotate(results: dict, signer=None) -> dict:
    """Add wilson95 + MDE to scores and a content-hash (+ signature) to the doc."""
    for name, v in list(results.items()):
        if isinstance(v, dict) and "acc" in v and "n" in v:
            v["wilson95"] = wilson95(round(v["acc"] * v["n"]), v["n"])
            v["mde"] = mde(v["acc"], v["n"])
    body = json.dumps(results, sort_keys=True, separators=(",", ":")).encode()
    results["_content_hash"] = hashlib.sha256(body).hexdigest()
    if signer is not None:
        results["_signature"] = signer(results["_content_hash"])
    return results
