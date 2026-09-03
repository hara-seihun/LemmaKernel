import Mathlib
import Bilinear_invariants.Reference
import Lk.Contract

/-!
# bilinear_invariants: contract

The executable reference fixes the three square-class labels and one representative for each
congruence class. These statements relate those choices to Mathlib matrices over `ZMod p`.
-/

namespace Bilinear_invariants.Contract

open Matrix Lk Lk.Contract

variable {p n : ℕ} [Fact p.Prime] {a b : Mat}

def Alternating (A : Matrix (Fin n) (Fin n) (ZMod p)) : Prop :=
  (∀ i, A i i = 0) ∧ A.transpose = -A

def Congruent (A B : Matrix (Fin n) (Fin n) (ZMod p)) : Prop :=
  ∃ P : Matrix (Fin n) (Fin n) (ZMod p), IsUnit P ∧ P.transpose * A * P = B

def HasSquareClass (x : ZMod p) (c : ℕ) : Prop :=
  (x = 0 ∧ c = 0) ∨ (x ≠ 0 ∧ IsSquare x ∧ c = 1) ∨ (¬IsSquare x ∧ c = 2)

def HasDiscriminantClass (A : Matrix (Fin n) (Fin n) (ZMod p)) (c : ℕ) : Prop :=
  if A.rank = 0 then c = 0 else
    ∃ e : Fin A.rank ↪ Fin n, IsUnit (A.submatrix e e) ∧ HasSquareClass (A.submatrix e e).det c

theorem rank_spec (h : WellFormed p n n a) :
    Gfp.rank p a = (toMatrix p n n a).rank := by
  sorry

theorem radicalDimension_spec (h : WellFormed p n n a) :
    n - Gfp.rank p a = Module.finrank (ZMod p) (LinearMap.ker (toMatrix p n n a).mulVecLin) := by
  sorry

theorem determinant_spec (h : WellFormed p n n a) :
    (determinant p a : ZMod p) = (toMatrix p n n a).det := by
  sorry

theorem determinantClass_spec (h : WellFormed p n n a) :
    HasSquareClass (toMatrix p n n a).det (squareClass p (determinant p a)) := by
  sorry

theorem discriminantClass_spec (h : WellFormed p n n a)
    (hs : (toMatrix p n n a).IsSymm ∨ Alternating (toMatrix p n n a)) :
    HasDiscriminantClass (toMatrix p n n a) (discriminantClass p a) := by
  sorry

theorem isNondegenerate_spec (h : WellFormed p n n a) :
    (Gfp.rank p a = n) ↔ LinearMap.ker (toMatrix p n n a).mulVecLin = ⊥ := by
  sorry

theorem isAlternating_spec (h : WellFormed p n n a) :
    isAlternating p a = true ↔ Alternating (toMatrix p n n a) := by
  sorry

theorem congruenceLabel_spec (h : WellFormed p n n a) (hv : validForm p a = true) :
    Congruent (toMatrix p n n a) (toMatrix p n n (congruenceLabel p a)) := by
  sorry

/-- Equality of labels is exactly congruence for the symmetric and alternating forms accepted by
this module. -/
theorem congruenceLabel_complete (ha : WellFormed p n n a) (hb : WellFormed p n n b)
    (hva : validForm p a = true) (hvb : validForm p b = true) :
    congruenceLabel p a = congruenceLabel p b ↔
      Congruent (toMatrix p n n a) (toMatrix p n n b) := by
  sorry

end Bilinear_invariants.Contract
