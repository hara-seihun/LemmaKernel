"""Cases for exact concise-circuit fire tests."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lemmakernel as lk  # noqa: E402
from tools.harness import Case  # noqa: E402

FANO_DIRECTIONS = [list(vector) for vector in itertools.product(range(2), repeat=3) if any(vector)]
FANO = [direction + direction for direction in FANO_DIRECTIONS]
BROKEN = [row[:] for row in FANO]
BROKEN[-1] = BROKEN[0][:]
TERNARY_NONFIRE = [
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [1, 1, 1, 1],
    [1, 2, 2, 1],
]
FANO_DEFECTS = [1, 0, 0, 0, 0, 0, 0]
FANO_POTENTIAL = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]


def cases(ctx, rng):
    del rng
    family = ctx.explicit(lk.matrix(2, [FANO, BROKEN]))
    defects = lk.matrix(2, [FANO_DEFECTS])
    potential = lk.matrix(2, [FANO_POTENTIAL])
    out = [
        Case("binary Fano circuit", family, "is_circuit", {"base_dim": 3, "limit": 2}),
        Case("noncircuit is not a fire", ctx.explicit(lk.matrix(2, [BROKEN])),
             "is_fire", {"base_dim": 3, "limit": 1}),
        Case("ternary circuit without a fire",
             ctx.explicit(lk.matrix(3, [TERNARY_NONFIRE])),
             "is_fire", {"base_dim": 2, "limit": 1}),
        Case("binary Fano fire", ctx.explicit(lk.matrix(2, [FANO])),
             "is_fire", {"base_dim": 3, "limit": 1}, oracle=False, bench="all",
             what="the seven-row binary Fano concise circuit"),
        Case("binary Fano constructed potential", family,
             "find_potential", {"base_dim": 3}),
        Case("binary Fano listed potential", ctx.explicit(lk.matrix(2, [FANO])),
             "verifies_potential",
             {"base_dim": 3, "defects": defects, "potential": potential, "limit": 1}),
        Case("natural-number configuration", ctx.explicit(lk.naturals([FANO])),
             "is_fire", {"base_dim": 3}, reductions=["all"], oracle=False),
        Case("base dimension consumes every column", family,
             "is_circuit", {"base_dim": 6}, reductions=["all"], oracle=False),
        Case("potential has the wrong number of rows", ctx.explicit(lk.matrix(2, [FANO])),
             "verifies_potential",
             {"base_dim": 3, "defects": defects, "potential": lk.matrix(2, [[[0, 0, 0]]])},
             reductions=["all"], oracle=False),
        Case("potential construction is too large",
             ctx.explicit(lk.matrix(7, [[[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]]])),
             "find_potential", {"base_dim": 5}, reductions=["all"], oracle=False),
    ]
    return out


def invariants(ctx):
    family = ctx.explicit(lk.matrix(2, [FANO, BROKEN]))
    circuits = ctx.value("circuit_fires.is_circuit", family, base_dim=3).values
    fires = ctx.value("circuit_fires.is_fire", family, base_dim=3).values
    fano_family = ctx.explicit(lk.matrix(2, [FANO]))
    verified = ctx.value(
        "circuit_fires.verifies_potential", fano_family,
        base_dim=3,
        defects=lk.matrix(2, [FANO_DEFECTS]),
        potential=lk.matrix(2, [FANO_POTENTIAL]),
    ).values
    witness = ctx.value("circuit_fires.find_potential", fano_family, base_dim=3).member(0)
    assert list(circuits) == [1, 0]
    assert list(fires) == [1, 0]
    assert list(verified) == [1]
    assert witness is not None
    witness_potential = [witness[index:index + 3] for index in range(0, 24, 3)]
    witness_defects = witness[24:]
    assert list(ctx.value(
        "circuit_fires.verifies_potential", fano_family,
        base_dim=3,
        defects=lk.matrix(2, [witness_defects]),
        potential=lk.matrix(2, [witness_potential]),
    ).values) == [1]
