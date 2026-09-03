import Mathlib
import Polynomials_fq.Reference
import Lk.Contract

/-!
# polynomials_fq: contract

What each output of `Reference.lean` means in Mathlib's terms. These are the statements a caller
relies on when they use the module. They are stated, not proved: the native backends are checked
against `Reference.lean` by `decide`, and `Reference.lean` is related to Mathlib here. A theorem
below with `sorry` is an obligation that has been written down and not yet discharged; there is
no other tooling that would make the native code "verified", and we do not claim it is.

`toPoly` is the bridge: a coefficient list is read as `∑ aᵢ Xⁱ`, entry `i` being the coefficient
of `Xⁱ`. A member row of length `d` is `polyOf`, one degree higher and monic.
-/

namespace Polynomials_fq.Contract

open Polynomial Lk Lk.Contract

variable {p : ℕ} [Fact p.Prime] {f a b : Poly}

/-- The polynomial a coefficient list denotes. -/
noncomputable def toPoly (p : ℕ) (f : Poly) : Polynomial (ZMod p) :=
  ∑ i ∈ Finset.range f.length, C ((f.getD i 0 : ZMod p)) * X ^ i

/-- A coefficient list is well formed over `F_p` when its entries are reduced and it carries no
trailing zero, so that its last entry is the leading coefficient. -/
def WellFormedPoly (p : ℕ) (f : Poly) : Prop := (∀ x ∈ f, x < p) ∧ (f = [] ∨ lead f ≠ 0)

/-! ## Members

A member row `[a₀, …, a_{d-1}]` is the monic polynomial of degree `d` whose lower coefficients it
lists. This is the choice the whole module is built on: the degree is the number of columns, so
`all_matrices q 1 d` is exactly the monic polynomials of degree `d`. -/

theorem polyOf_spec (row : Vec) :
    toPoly p (polyOf row) = X ^ row.length + toPoly p row := by
  sorry

theorem polyOf_monic (row : Vec) : (toPoly p (polyOf row)).Monic ∧
    (toPoly p (polyOf row)).natDegree = row.length := by
  sorry

/-! ## Operations -/

theorem irreducible_spec (h : WellFormedPoly p f) :
    irreducible p f = true ↔ Irreducible (toPoly p f) := by
  sorry

/-- The degrees of the irreducible factors, with multiplicity, sorted. `UniqueFactorizationMonoid.factors`
is the same multiset of irreducible factors, up to the units this module has already fixed by
returning only degrees. -/
theorem factorDegrees_spec (h : WellFormedPoly p f) (hf : toPoly p f ≠ 0) :
    (factorDegrees p f).Pairwise (· ≤ ·) ∧
    (factorDegrees p f : Multiset ℕ) =
      (UniqueFactorizationMonoid.factors (toPoly p f)).map Polynomial.natDegree := by
  sorry

/-- Hence the degrees sum to the degree, which is what a caller usually checks. -/
theorem factorDegrees_sum (h : WellFormedPoly p f) (hf : toPoly p f ≠ 0) :
    (factorDegrees p f).sum = (toPoly p f).natDegree := by
  sorry

/-- The order of `f`: the least `e ≥ 1` for which `f` divides `X^e - 1` once the largest power of
`X` dividing it has been removed (Lidl and Niederreiter's `ord`). -/
theorem polyOrder_spec (h : WellFormedPoly p f) (hf : f ≠ []) :
    IsLeast {e : ℕ | 1 ≤ e ∧ toPoly p (stripX f) ∣ X ^ e - 1} (polyOrder p f) := by
  sorry

/-- `x` generates the nonzero elements of `F_p[x]/(f)`: `f` is the minimal polynomial of a
generator of the multiplicative group of the field of `p ^ d` elements. -/
theorem primitive_spec (h : WellFormedPoly p f) :
    primitive p f = true ↔ Irreducible (toPoly p f) ∧
      ∀ y : AdjoinRoot (toPoly p f), y ≠ 0 → ∃ k : ℕ, y = AdjoinRoot.root (toPoly p f) ^ k := by
  sorry

theorem rootsOf_spec (h : WellFormedPoly p f) :
    (rootsOf p f).Pairwise (· < ·) ∧ (∀ x ∈ rootsOf p f, x < p) ∧
    ∀ x < p, x ∈ rootsOf p f ↔ (toPoly p f).IsRoot (x : ZMod p) := by
  sorry

theorem rootCount_spec (h : WellFormedPoly p f) (hf : toPoly p f ≠ 0) :
    (rootsOf p f).length = (toPoly p f).roots.toFinset.card := by
  sorry

/-- The monic greatest common divisor, spelled out rather than deferred to a normalisation
convention: it divides both, every common divisor divides it, and it is monic. -/
theorem pgcd_spec (ha : WellFormedPoly p a) (hb : WellFormedPoly p b) :
    let d := toPoly p (pgcd p a b)
    d ∣ toPoly p a ∧ d ∣ toPoly p b ∧
    (∀ e : Polynomial (ZMod p), e ∣ toPoly p a → e ∣ toPoly p b → e ∣ d) ∧
    (d ≠ 0 → d.Monic) := by
  sorry

/-- What `gcd` returns is that polynomial in the module's convention: the coefficients below the
leading one, so that reading it back as a member row gives the gcd. -/
theorem gcd_value_spec (ha : WellFormedPoly p a) (hb : WellFormedPoly p b)
    (hd : pgcd p a b ≠ []) :
    polyOf (pgcd p a b).dropLast = pgcd p a b := by
  sorry

end Polynomials_fq.Contract
