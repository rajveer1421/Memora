"""Generate a VAPID key pair for Web Push (free, nothing to sign up for).

    python scripts/gen_vapid_keys.py
Copy the two printed lines into your .env (or HF Space secrets). Never commit them."""
import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def b64url(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


key = ec.generate_private_key(ec.SECP256R1())
private = b64url(key.private_numbers().private_value.to_bytes(32, "big"))
public = b64url(key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint))
print(f"MEMORA_VAPID_PUBLIC_KEY={public}")
print(f"MEMORA_VAPID_PRIVATE_KEY={private}")
