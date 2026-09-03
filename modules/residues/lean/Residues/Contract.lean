import Mathlib
import Residues.Reference
import Lk.Contract

/-!
# residues: contract

What the executable reference computes, said in Mathlib's language: `ZMod n` and its units,
`orderOf`, `IsSquare`, `Nat.totient`, `Nat.primeFactorsList`, `legendreSym` and `jacobiSym`.
The native backend is checked against the executable reference, not against this file; these are
the statements a caller relies on when they read an answer as mathematics. A `sorry` is an
obligation written down and not yet discharged.

The reference works with naturals and returns naturals, so two conventions are named here once:
a residue `a : ℕ` means `(a : ZMod n)`, and a quadratic symbol comes back encoded, with `2`
standing for `-1`. `decodeSymbol` undoes that encoding.
-/

namespace Residues.Contract

open Lk

/-- The encoding every symbol-valued operation uses: `0` and `1` are themselves, `2` is `-1`. -/
def decodeSymbol (s : ℕ) : ℤ := if s = 2 then -1 else (s : ℤ)

/-- `a` is a unit of `Z/n` exactly when the reference's `gcd` says it is coprime to `n`. -/
theorem gcd_spec (a b : ℕ) : gcd a b = Nat.gcd a b := by
  sorry

theorem totient_spec (n : ℕ) : totient n = Nat.totient n := by
  sorry

theorem factors_spec (n : ℕ) : factors n = n.primeFactorsList := by
  sorry

theorem isPrime_spec (n : ℕ) : isPrime n = true ↔ n.Prime := by
  sorry

/-! ## Multiplicative order -/

/-- A unit's order is the order of the corresponding element of `(ZMod n)ˣ`. -/
theorem order_spec {n a : ℕ} (hn : 1 ≤ n) (h : Nat.Coprime a n) :
    order n a = orderOf (ZMod.unitOfCoprime a h) := by
  sorry

/-- Nothing else has an order: no power of a non-unit is `1`. -/
theorem order_eq_zero {n a : ℕ} (hn : 1 ≤ n) (h : ¬ Nat.Coprime a n) :
    order n a = 0 ∧ ∀ k, 1 ≤ k → ((a : ZMod n) ^ k ≠ 1) := by
  sorry

/-- A primitive root generates the whole unit group. -/
theorem isPrimitiveRoot_spec {n a : ℕ} (hn : 1 ≤ n) :
    isPrimitiveRoot n a = true ↔
      ∃ h : Nat.Coprime a n, ∀ u : (ZMod n)ˣ, u ∈ Subgroup.zpowers (ZMod.unitOfCoprime a h) := by
  sorry

/-- The least primitive root, when there is one, and `0` exactly when `(ZMod n)ˣ` is not cyclic. -/
theorem leastPrimitiveRoot_spec {n : ℕ} (hn : 1 ≤ n) (h : leastPrimitiveRoot n ≠ 0) :
    isPrimitiveRoot n (leastPrimitiveRoot n) = true ∧
      ∀ g, 1 ≤ g → g < leastPrimitiveRoot n → isPrimitiveRoot n g = false := by
  sorry

theorem leastPrimitiveRoot_eq_zero {n : ℕ} (hn : 1 ≤ n) :
    leastPrimitiveRoot n = 0 ↔ ¬ IsCyclic (ZMod n)ˣ := by
  sorry

/-! ## Quadratic residues -/

/-- A quadratic residue is a unit with a square root; `0` and the other non-units are excluded. -/
theorem isQuadraticResidue_spec {n a : ℕ} (hn : 1 ≤ n) :
    isQuadraticResidue n a = true ↔ IsUnit (a : ZMod n) ∧ IsSquare (a : ZMod n) := by
  sorry

theorem legendre_spec {p : ℕ} [Fact p.Prime] (hodd : p % 2 = 1) (a : ℕ) :
    decodeSymbol (legendre p a) = legendreSym p (a : ℤ) := by
  sorry

theorem jacobi_spec {n : ℕ} (hodd : n % 2 = 1) (a : ℕ) :
    decodeSymbol (jacobi n a) = jacobiSym (a : ℤ) n := by
  sorry

/-! ## Discrete logarithms -/

/-- When a logarithm is reported it is a logarithm, and the least one. -/
theorem discreteLog_spec {n b a : ℕ} (hn : 1 ≤ n) (h : discreteLog n b a < n) :
    ((b : ZMod n) ^ discreteLog n b a = (a : ZMod n)) ∧
      ∀ x < discreteLog n b a, (b : ZMod n) ^ x ≠ (a : ZMod n) := by
  sorry

/-- The modulus is reported exactly when there is no logarithm at all: the powers of `b` repeat
within `n` steps, so a search below `n` misses nothing. -/
theorem discreteLog_eq_modulus {n b a : ℕ} (hn : 1 ≤ n) :
    discreteLog n b a = n ↔ ∀ x : ℕ, (b : ZMod n) ^ x ≠ (a : ZMod n) := by
  sorry

end Residues.Contract
