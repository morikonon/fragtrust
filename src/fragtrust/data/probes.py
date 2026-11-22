"""Out-of-distribution probe blocks: types the classifiers never see in training."""

from __future__ import annotations

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

PROBE_DESCRIPTIONS = {
    "ood_aes_ctr": "AES-256-CTR encryption of randomly chosen test blocks; one key, fresh random nonce per block",
    "ood_random": "uniformly random bytes",
}


def aes_ctr_blocks(plaintext: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    key = rng.bytes(32)
    out = np.empty_like(plaintext)
    for i, block in enumerate(plaintext):
        encryptor = Cipher(algorithms.AES(key), modes.CTR(rng.bytes(16))).encryptor()
        out[i] = np.frombuffer(encryptor.update(block.tobytes()) + encryptor.finalize(), dtype=np.uint8)
    return out


def make_probes(test_x: np.ndarray, n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = min(n, len(test_x))
    plaintext = np.asarray(test_x[np.sort(rng.choice(len(test_x), n, replace=False))])
    return {
        "ood_aes_ctr": aes_ctr_blocks(plaintext, rng),
        "ood_random": rng.integers(0, 256, size=(n, test_x.shape[1]), dtype=np.uint8),
    }
