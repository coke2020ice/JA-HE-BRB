from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Tuple


def _is_probable_prime(n: int, rounds: int = 12) -> bool:
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small_primes:
        return True
    if any(n % p == 0 for p in small_primes):
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    rng = random.Random(n)
    for _ in range(rounds):
        a = rng.randrange(2, n - 2)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _generate_prime(bits: int, rng: random.Random) -> int:
    if bits < 16:
        raise ValueError("Prime size must be at least 16 bits")
    while True:
        candidate = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(candidate):
            return candidate


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def _mod_inverse(a: int, modulus: int) -> int:
    return pow(a, -1, modulus)


@dataclass(frozen=True)
class PaillierPublicKey:
    n: int

    @property
    def n_square(self) -> int:
        return self.n * self.n

    @property
    def ciphertext_bytes(self) -> int:
        return (self.n_square.bit_length() + 7) // 8

    def encrypt_int(self, value: int, rng: random.Random | None = None) -> int:
        encoded = value % self.n
        generator = self.n + 1
        local_rng = rng or random.SystemRandom()
        while True:
            r = local_rng.randrange(1, self.n)
            if math.gcd(r, self.n) == 1:
                break
        return (pow(generator, encoded, self.n_square) * pow(r, self.n, self.n_square)) % self.n_square

    def add(self, ciphertexts: Iterable[int]) -> int:
        total = 1
        for ciphertext in ciphertexts:
            total = (total * ciphertext) % self.n_square
        return total


@dataclass(frozen=True)
class PaillierPrivateKey:
    public_key: PaillierPublicKey
    lam: int
    mu: int

    def decrypt_int(self, ciphertext: int) -> int:
        nsquare = self.public_key.n_square
        u = pow(ciphertext, self.lam, nsquare)
        l_value = (u - 1) // self.public_key.n
        value = (l_value * self.mu) % self.public_key.n
        if value > self.public_key.n // 2:
            value -= self.public_key.n
        return value


def generate_paillier_keypair(bits: int = 128, seed: int = 13) -> Tuple[PaillierPublicKey, PaillierPrivateKey]:
    """Generate a compact Paillier keypair for reproducible experiments.

    The quick experiments intentionally use small keys so they finish quickly.
    Use 1024 bits or more for a real deployment study.
    """

    if bits < 64:
        raise ValueError("Paillier modulus must be at least 64 bits")
    rng = random.Random(seed)
    half = bits // 2
    while True:
        p = _generate_prime(half, rng)
        q = _generate_prime(bits - half, rng)
        if p != q:
            n = p * q
            lam = _lcm(p - 1, q - 1)
            generator = n + 1
            nsquare = n * n
            l_value = (pow(generator, lam, nsquare) - 1) // n
            if math.gcd(l_value, n) == 1:
                break
    public_key = PaillierPublicKey(n=n)
    private_key = PaillierPrivateKey(public_key=public_key, lam=lam, mu=_mod_inverse(l_value, n))
    return public_key, private_key


def encrypt_float(value: float, public_key: PaillierPublicKey, scale: int, rng: random.Random) -> int:
    return public_key.encrypt_int(int(round(value * scale)), rng=rng)


def decrypt_float(ciphertext: int, private_key: PaillierPrivateKey, scale: int) -> float:
    return private_key.decrypt_int(ciphertext) / float(scale)

