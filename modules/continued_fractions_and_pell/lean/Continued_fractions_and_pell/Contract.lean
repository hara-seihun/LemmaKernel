import Mathlib
import Continued_fractions_and_pell.Reference
import Lk.Contract

/-!
# continued_fractions_and_pell: contract

What the executable reference computes, said in Mathlib's language: `Nat.sqrt`, `Real.sqrt`,
`IsSquare`, `Nat.gcd`, and `Pell.Solution₁` with `Pell.IsFundamental`. The native backend is
checked against the executable reference, not against this file; these are the statements a
caller relies on when they read an answer as mathematics. A `sorry` is an obligation written
down and not yet discharged.

Two conventions are named once here. A member `d : ℕ` is a radicand, and the quadratic order in
play is `Z[sqrt d]`, not the ring of integers of `ℚ(sqrt d)`: for `d ≡ 1 mod 4` those differ, and
every operation of this module works in `Z[sqrt d]`. And a unit comes back encoded, as
`(s, x, y)` with `s = 1` exactly when the norm `x^2 - d y^2` is `-1`.

`classNumber` is the one operation that reads its member as `n` for the discriminant `D = -n`. It
counts classes of binary quadratic forms, which is the classical definition of `h(D)` for any
discriminant; Mathlib has the ideal class group of a Dedekind domain but not of a non-maximal
quadratic order, so the identification of the two is stated only through the form count.
-/

namespace Continued_fractions_and_pell.Contract

open Lk

/-! ## Integer square root -/

theorem isqrt_spec (n : ℕ) : isqrt n = Nat.sqrt n := by
  sorry

theorem isqrt_floor (n : ℕ) : (isqrt n : ℝ) = ⌊Real.sqrt n⌋ := by
  sorry

theorem isSquare_spec (n : ℕ) : isSquare n = true ↔ IsSquare n := by
  sorry

theorem gcd_spec (a b : ℕ) : gcd a b = Nat.gcd a b := by
  sorry

/-! ## The continued fraction of `sqrt d`

`evalCF as t` is the finite continued fraction `[a_0; a_1, ..., a_{k-1}, t]`: the value of the
partial quotients `as` with the real `t` as the last complete quotient. -/

noncomputable def evalCF : List ℕ → ℝ → ℝ
  | [], t => t
  | a :: rest, t => (a : ℝ) + 1 / evalCF rest t

/-- `a_0` is the integer part of `sqrt d`. -/
theorem expansionTerms_head (d : ℕ) : (expansionTerms d).headD 0 = Nat.sqrt d := by
  sorry

/-- A perfect square has no period: its expansion is the single term `sqrt d`. -/
theorem expansionTerms_of_isSquare {d : ℕ} (h : IsSquare d) : expansionTerms d = [Nat.sqrt d] := by
  sorry

/-- Otherwise the period is nonempty and ends at `2 a_0`, which is what makes it a period: the
complete quotient there is `a_0 + sqrt d`, the same one the expansion started from. -/
theorem periodTerms_getLast {d : ℕ} (h : ¬ IsSquare d) :
    (periodTerms d).getLast? = some (2 * Nat.sqrt d) := by
  sorry

/-- The expansion is the continued fraction of `sqrt d`, and one period of it is enough: reading
`a_0, a_1, ..., a_{L-1}` and then the complete quotient `a_0 + sqrt d` returns `sqrt d` exactly.
Unfolding that identity into itself is the infinite expansion
`sqrt d = [a_0; a_1, ..., a_L, a_1, ..., a_L, ...]`. -/
theorem expansionTerms_spec {d : ℕ} (h : ¬ IsSquare d) :
    evalCF (expansionTerms d).dropLast ((Nat.sqrt d : ℝ) + Real.sqrt d) = Real.sqrt d := by
  sorry

/-- `cf_period`, `cf_period_max` and `cf_period_sum` read off the same period. -/
theorem periodStatistics (d : ℕ) :
    periodLength d = (periodTerms d).length ∧
      periodMax d = (periodTerms d).foldl max 0 ∧
      periodSum d = (periodTerms d).sum := by
  sorry

/-! ## Units of `Z[sqrt d]` -/

/-- The norm the encoded sign stands for. -/
def normOf (s : ℕ) : ℤ := if s = 1 then -1 else 1

/-- Whatever `fundamental_unit` returns is a unit of `Z[sqrt d]`: it solves `x^2 - d y^2 = ±1`,
with the norm its flag claims. -/
theorem unitOf_norm {d s x y : ℕ} (h : unitOf d = some (s, x, y)) :
    (x : ℤ) ^ 2 - (d : ℤ) * (y : ℤ) ^ 2 = normOf s := by
  sorry

/-- It is the fundamental one: every other solution in positive integers is at least as large,
so `x + y sqrt d` is the least unit of `Z[sqrt d]` above `1`. -/
theorem unitOf_least {d s x y : ℕ} (h : unitOf d = some (s, x, y)) :
    0 < x ∧ 0 < y ∧ ∀ u v : ℕ, 0 < u → 0 < v →
      ((u : ℤ) ^ 2 - (d : ℤ) * (v : ℤ) ^ 2 = 1 ∨ (u : ℤ) ^ 2 - (d : ℤ) * (v : ℤ) ^ 2 = -1) →
      x ≤ u := by
  sorry

/-- There is a unit exactly when `d` is not a perfect square. -/
theorem unitOf_isSome (d : ℕ) : (unitOf d).isSome = true ↔ ¬ IsSquare d := by
  sorry

/-- The norm is `-1` exactly when the period has odd length, which is exactly when the negative
Pell equation is solvable. -/
theorem negativePell_spec (d : ℕ) :
    negativePell d = true ↔ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ (x : ℤ) ^ 2 - (d : ℤ) * (y : ℤ) ^ 2 = -1 := by
  sorry

/-! ## The fundamental Pell solution -/

/-- `pell_fundamental` returns Mathlib's fundamental solution of `x^2 - d y^2 = 1`. -/
theorem pellOf_isFundamental {d s x y : ℕ} (h : pellOf d = some (s, x, y)) :
    s = 0 ∧ ∃ a : Pell.Solution₁ (d : ℤ), a.x = (x : ℤ) ∧ a.y = (y : ℤ) ∧ Pell.IsFundamental a := by
  sorry

/-- And there is one exactly when `d` is not a perfect square, which for `d > 0` is Mathlib's
hypothesis for the existence of a fundamental solution. -/
theorem pellOf_isSome (d : ℕ) : (pellOf d).isSome = true ↔ ¬ IsSquare d := by
  sorry

/-! ## Class numbers of imaginary quadratic orders -/

/-- A primitive reduced positive definite form of discriminant `-n`: `b^2 - 4ac = -n` with
`gcd (a, b, c) = 1`, `-a < b ≤ a ≤ c`, and `b ≥ 0` when `a = c`. Each class of forms of
discriminant `-n` contains exactly one of these. -/
def ReducedForm (n : ℕ) (f : ℤ × ℤ × ℤ) : Prop :=
  let (a, b, c) := f
  b ^ 2 - 4 * a * c = -(n : ℤ) ∧ Int.gcd (Int.gcd a b) c = 1 ∧
    0 < a ∧ -a < b ∧ b ≤ a ∧ a ≤ c ∧ (a = c → 0 ≤ b)

/-- `class_number` counts them. -/
theorem classNumber_spec (n : ℕ) :
    classNumber n = Nat.card {f : ℤ × ℤ × ℤ // ReducedForm n f} := by
  sorry

/-- `-n` is a discriminant exactly when it is `0` or `1` modulo `4` and nonzero, and every
discriminant has at least the principal form. -/
theorem classNumber_eq_zero (n : ℕ) : classNumber n = 0 ↔ n = 0 ∨ n % 4 = 1 ∨ n % 4 = 2 := by
  sorry

end Continued_fractions_and_pell.Contract
