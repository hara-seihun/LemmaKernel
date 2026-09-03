import Mathlib
import Sieve_ranges.Reference
import Lk.Contract

/-!
# sieve_ranges: contract

What each output of `Reference.lean` means in Mathlib's terms. These are the statements a caller
relies on when they use the module. They are stated, not proved: the backend is checked against
`Reference.lean` by `decide`, and `Reference.lean` is related to Mathlib here. A theorem below
with `sorry` is an obligation that has been written down and not yet discharged; there is no
tooling here that would make the native code "verified", and we do not claim it is.

Every convention at the ends of the domain is Mathlib's, which is why each statement below is an
equation with no side condition on `n`: `Nat.factorization 0 = 0`, `Nat.totient 0 = 0`,
`Nat.divisors 0 = ∅` (so `sigma` and `divisorCount` vanish at `0`), `¬ Squarefree 0`, and
`ArithmeticFunction.moebius 0 = 0`.

The one place the module does not return the mathematical value is `mobius`, which returns
`μ n + 1` so that the value is a natural number; `mobius_spec` states exactly that.
-/

namespace Sieve_ranges.Contract

open ArithmeticFunction

variable {n : ℕ}

/-! ## The factorisation, from which everything else is read -/

/-- The primes listed are exactly the prime factors of `n`, in increasing order. -/
theorem factorisation_primes :
    (factorisation n).map Prod.fst = n.primeFactors.sort (· ≤ ·) := by
  sorry

/-- Each listed exponent is the multiplicity Mathlib's `Nat.factorization` gives that prime. -/
theorem factorisation_exponents (p e : ℕ) (h : (p, e) ∈ factorisation n) :
    n.factorization p = e ∧ e ≠ 0 := by
  sorry

/-- The pairs multiply back to `n`, for every `n` that has a factorisation into primes. -/
theorem factorisation_prod (h : n ≠ 0) :
    ((factorisation n).map fun pe => pe.1 ^ pe.2).prod = n := by
  sorry

/-! ## The arithmetic functions -/

theorem isPrime_spec : isPrime n = true ↔ n.Prime := by
  sorry

theorem isSquarefree_spec : isSquarefree n = true ↔ Squarefree n := by
  sorry

/-- `ω`, the number of distinct prime factors. -/
theorem omega_spec : omega n = cardDistinctFactors n := by
  sorry

/-- `Ω`, the number of prime factors with multiplicity. -/
theorem bigOmega_spec : bigOmega n = cardFactors n := by
  sorry

theorem totient_spec : totient n = n.totient := by
  sorry

/-- `σ₁`, the sum of the divisors. -/
theorem sigma_spec : sigma n = ArithmeticFunction.sigma 1 n := by
  sorry

/-- `τ = σ₀`, the number of divisors. -/
theorem divisorCount_spec : divisorCount n = n.divisors.card := by
  sorry

/-- Moebius, shifted by one into the unsigned values a reduction carries. -/
theorem mobius_spec : (mobius n : ℤ) = moebius n + 1 := by
  sorry

/-- The largest prime factor, and `0` when `n` has none (`n = 0` or `n = 1`). -/
theorem largestPrimeFactor_spec : largestPrimeFactor n = n.primeFactors.sup id := by
  sorry

/-! ## What a `range` family answers

An operation over `range a b` answers for every integer of the interval, once, in increasing
order; the reference's `numbers?` is that list. -/

theorem numbers_range (a b : ℕ) :
    numbers? (.range a b) = some (List.range' a (b - a)) := by
  sorry

end Sieve_ranges.Contract
