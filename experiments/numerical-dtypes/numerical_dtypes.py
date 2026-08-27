#!/usr/bin/env python3
"""Small, dependency-free experiments for the numerical dtypes article."""

import math
import struct


def fp32_round(value: float) -> float:
    """Round a Python binary64 float through IEEE binary32 storage."""
    return struct.unpack(">f", struct.pack(">f", value))[0]


def show_fp32_point_one() -> None:
    encoded = struct.pack(">f", 0.1)
    bits = struct.unpack(">I", encoded)[0]
    decoded = struct.unpack(">f", encoded)[0]
    print("Experiment 1: decimal 0.1 stored as binary32")
    print(f"bits:  0x{bits:08X}")
    print(f"value: {decoded:.17g}")


def show_non_associativity() -> None:
    a, b, c = 1e16, -1e16, 1.0
    print("\nExperiment 2: rounded addition is not associative")
    print(f"(a + b) + c = {(a + b) + c}")
    print(f"a + (b + c) = {a + (b + c)}")


def show_summation_error() -> None:
    values = [1e16, 1.0, -1e16]
    naive = 0.0
    for value in values:
        naive += value
    print("\nExperiment 3: accumulation strategy matters")
    print(f"naive:     {naive}")
    print(f"math.fsum: {math.fsum(values)}")


def show_binary16_example() -> None:
    encoded = struct.pack(">e", 13.25)
    bits = struct.unpack(">H", encoded)[0]
    print("\nCheck: 13.25 stored as binary16")
    print(f"bits: 0x{bits:04X}")


if __name__ == "__main__":
    show_fp32_point_one()
    show_non_associativity()
    show_summation_error()
    show_binary16_example()
