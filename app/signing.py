"""Signing the Agent Card - so a caller can check the card came from this deployment.

A2A 1.0 lets a publisher attach `signatures`: each one a detached JWS over the card. The card
without its `signatures` field is canonicalised (keys sorted, no whitespace - RFC 8785's form
for a card like this one, which holds only strings, booleans, lists and objects), and signed:

    protected = base64url({"alg": "ES256", "jku": <base>/.well-known/jwks.json, "kid": ...,
                           "typ": "JOSE"})
    signature = base64url(ECDSA-P256-SHA256(protected + "." + base64url(canonical card)))

The payload is left out ("detached") - the card itself is the payload. The public key is
served as a JWK Set at /.well-known/jwks.json, and `kid` names the key that signed it. ES256
because every browser's WebCrypto verifies it - the page checks the signature itself.

The key comes from ONE setting, CARD_SIGNING_SEED: its SHA-256 is the P-256 private scalar.
Render generates the seed (render.yaml, `generateValue`), so a deploy signs with no key file to
manage. Without the setting (locally, the tests) the card is STILL signed - with a seed drawn at
boot, so the key changes on every restart (`/health` says `card_key: "ephemeral"`). Rotating
the seed rotates the key - callers re-fetch the JWKS by `kid`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

from app.config import get_settings

JWKS_PATH = "/.well-known/jwks.json"
_P256_ORDER = int("FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551", 16)
# ECDSA draws a fresh nonce per signature, so the same card would get a different signature on
# every request. One signature per (key, card) keeps the card's bytes stable across requests.
_SIGNED: dict[tuple[str, bytes], str] = {}
# No CARD_SIGNING_SEED: one random seed for this process - the card is never served unsigned.
_BOOT_SEED = secrets.token_urlsafe(32)


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def canonical(obj: Any) -> bytes:
    """The bytes that get signed: keys sorted, no insignificant whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def key_is_stable() -> bool:
    """True when the key comes from CARD_SIGNING_SEED (survives restarts), not the boot seed."""
    return bool(get_settings().card_signing_seed)


def _private_key() -> ec.EllipticCurvePrivateKey:
    seed = get_settings().card_signing_seed or _BOOT_SEED
    scalar = int.from_bytes(hashlib.sha256(seed.encode()).digest(), "big") % (_P256_ORDER - 1) + 1
    return ec.derive_private_key(scalar, ec.SECP256R1())


def _xy(key: ec.EllipticCurvePrivateKey) -> tuple[bytes, bytes]:
    numbers = key.public_key().public_numbers()
    return numbers.x.to_bytes(32, "big"), numbers.y.to_bytes(32, "big")


def key_id(x: bytes, y: bytes) -> str:
    return hashlib.sha256(x + y).hexdigest()[:16]


def jwks() -> dict[str, list[dict[str, str]]]:
    """The public half, as a JWK Set."""
    key = _private_key()
    x, y = _xy(key)
    return {"keys": [{"kty": "EC", "crv": "P-256", "x": b64url(x), "y": b64url(y),
                      "kid": key_id(x, y), "alg": "ES256", "use": "sig"}]}


def sign_card(card: dict[str, Any], base_url: str) -> list[dict[str, str]]:
    """`signatures` for this card."""
    key = _private_key()
    kid = key_id(*_xy(key))
    header = {"alg": "ES256", "jku": f"{base_url}{JWKS_PATH}", "kid": kid, "typ": "JOSE"}
    protected = b64url(canonical(header))
    payload = {k: v for k, v in card.items() if k != "signatures"}
    signing_input = f"{protected}.{b64url(canonical(payload))}".encode()
    cache_key = (kid, signing_input)
    if cache_key not in _SIGNED:
        r, s = decode_dss_signature(key.sign(signing_input, ec.ECDSA(hashes.SHA256())))
        _SIGNED[cache_key] = b64url(r.to_bytes(32, "big") + s.to_bytes(32, "big"))  # JWS: r||s
    return [{"protected": protected, "signature": _SIGNED[cache_key]}]


def verify_card(card: dict[str, Any], key_set: dict[str, Any]) -> bool:
    """True when the card's first signature verifies against a key in `key_set` - what a
    careful caller does after fetching the card and the JWKS its `jku` names."""
    signatures = card.get("signatures") or []
    if not signatures:
        return False
    sig = signatures[0]
    try:
        header = json.loads(b64url_decode(sig["protected"]))
        jwk = next(k for k in key_set.get("keys", []) if k.get("kid") == header.get("kid"))
        if header.get("alg") != "ES256" or jwk.get("crv") != "P-256":
            return False
        public = ec.EllipticCurvePublicNumbers(
            int.from_bytes(b64url_decode(jwk["x"]), "big"),
            int.from_bytes(b64url_decode(jwk["y"]), "big"), ec.SECP256R1()).public_key()
        raw = b64url_decode(sig["signature"])
        if len(raw) != 64:
            return False
        der = encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
        payload = {k: v for k, v in card.items() if k != "signatures"}
        public.verify(der, f"{sig['protected']}.{b64url(canonical(payload))}".encode(),
                      ec.ECDSA(hashes.SHA256()))
        return True
    except (StopIteration, KeyError, ValueError, TypeError, AttributeError, InvalidSignature):
        return False
