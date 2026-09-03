import Mathlib
import Quadratic_forms.Reference
import Lk.Contract

/-!
# quadratic_forms: contract

The executable reference reads a symmetric matrix `A` over an odd prime field as the Mathlib
quadratic form `x ↦ x^T A x`. These statements connect rank, radical, Witt index, isometry, type,
and projective zero counts to that form. Proofs remain obligations. Native backends are tested
against the executable reference, not against these statements.
-/

namespace Quadratic_forms.Contract

open Matrix Lk Lk.Contract

variable {p n : ℕ} [Fact p.Prime]

/-- The Mathlib quadratic form represented by a Gram matrix. -/
def toQuadraticForm (p n : ℕ) (m : Mat) : QuadraticForm (ZMod p) (Fin n → ZMod p) :=
  (toMatrix p n n m).toQuadraticForm'

/-- Every vector in the subspace is a zero of the quadratic form. -/
def IsTotallyIsotropic (Q : QuadraticForm (ZMod p) (Fin n → ZMod p))
    (W : Submodule (ZMod p) (Fin n → ZMod p)) : Prop :=
  ∀ x ∈ W, Q x = 0

/-- `w` is the number of hyperbolic planes after removing the radical. Equivalently, maximal
totally isotropic subspaces have dimension `dim(radical) + w`. -/
def HasWittIndex (Q : QuadraticForm (ZMod p) (Fin n → ZMod p)) (w : ℕ) : Prop :=
  ∃ W : Submodule (ZMod p) (Fin n → ZMod p),
    IsTotallyIsotropic Q W ∧
    Module.finrank (ZMod p) W = Module.finrank (ZMod p) Q.radical + w ∧
    ∀ U : Submodule (ZMod p) (Fin n → ZMod p),
      IsTotallyIsotropic Q U → Module.finrank (ZMod p) U ≤ Module.finrank (ZMod p) W

theorem rank_spec {m : Mat} (h : WellFormed p n n m) (hsymm : (toMatrix p n n m).IsSymm)
    (hodd : p % 2 = 1) :
    rank p m = (toMatrix p n n m).rank := by
  sorry

theorem radical_spec {m : Mat} (h : WellFormed p n n m) (hsymm : (toMatrix p n n m).IsSymm)
    (hodd : p % 2 = 1) :
    let vs := (Gfp.nullspace p m).map (toVec p n)
    LinearIndependent (ZMod p) (fun i : Fin vs.length => vs[i]) ∧
    Submodule.span (ZMod p) (Set.range fun i : Fin vs.length => vs[i]) =
      (toQuadraticForm p n m).radical := by
  sorry

theorem wittIndex_spec {m : Mat} (h : WellFormed p n n m)
    (hsymm : (toMatrix p n n m).IsSymm) (hodd : p % 2 = 1) :
    HasWittIndex (toQuadraticForm p n m) (wittIndex p m) := by
  sorry

/-- The three type codes are determined by the rank and Witt index of the nonsingular quotient.
Rank zero is hyperbolic. -/
theorem formType_spec {m : Mat} (h : WellFormed p n n m)
    (hsymm : (toMatrix p n n m).IsSymm) (hodd : p % 2 = 1) :
    let r := rank p m
    let w := wittIndex p m
    (formType p m = 0 ↔ r % 2 = 0 ∧ w = r / 2) ∧
    (formType p m = 1 ↔ r % 2 = 0 ∧ w + 1 = r / 2) ∧
    (formType p m = 2 ↔ r % 2 = 1) := by
  sorry

theorem isIsometric_spec {m other : Mat} (hm : WellFormed p n n m)
    (ho : WellFormed p n n other) (hmsymm : (toMatrix p n n m).IsSymm)
    (hosymm : (toMatrix p n n other).IsSymm) (hodd : p % 2 = 1) :
    isIsometric p m other = true ↔
      QuadraticMap.Equivalent (toQuadraticForm p n m) (toQuadraticForm p n other) := by
  sorry

theorem isotropicPointCount_spec {m : Mat} (h : WellFormed p n n m)
    (hsymm : (toMatrix p n n m).IsSymm) (hodd : p % 2 = 1) :
    isotropicPointCount p m =
      ((Finset.univ.filter fun x : Fin n → ZMod p => x ≠ 0 ∧ toQuadraticForm p n m x = 0).card /
        (p - 1)) := by
  sorry

end Quadratic_forms.Contract
