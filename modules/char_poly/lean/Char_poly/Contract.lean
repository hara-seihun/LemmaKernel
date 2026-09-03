import Mathlib
import Char_poly.Reference
import Lk.Contract

/-!
# char_poly: contract

The executable reference fixes coefficient order, padding, companion orientation, and block order.
These statements relate those byte-level choices to Mathlib's characteristic polynomial, minimal
polynomial, semisimplicity, and matrix conjugacy. The proofs remain explicit obligations.
-/

namespace Char_poly.Contract

open Matrix Lk Lk.Contract Polynomial

variable {p n : ℕ} [Fact p.Prime] {a b : Mat}

noncomputable def toPolynomial (p : ℕ) (cs : Poly) : Polynomial (ZMod p) :=
  ∑ i ∈ Finset.range cs.length, Polynomial.monomial i (cs.getD i 0 : ZMod p)

def Conjugate (A B : Matrix (Fin n) (Fin n) (ZMod p)) : Prop :=
  ∃ P Q : Matrix (Fin n) (Fin n) (ZMod p),
    IsUnit P ∧ P * Q = 1 ∧ Q * P = 1 ∧ Q * A * P = B

theorem charpoly_spec (h : WellFormed p n n a) :
    toPolynomial p (characteristicPolynomial p a) = (toMatrix p n n a).charpoly := by
  sorry

theorem minpoly_spec (h : WellFormed p n n a) :
    toPolynomial p (minimalPolynomial p a) = minpoly (ZMod p) (toMatrix p n n a) := by
  sorry

/-- The returned block matrix is similar to the input and is the row-companion convention for
its divisibility-ordered invariant factors. -/
theorem rationalCanonicalForm_spec (h : WellFormed p n n a) :
    Conjugate (toMatrix p n n a) (toMatrix p n n (rationalCanonicalForm p a)) := by
  sorry

/-- Equality of labels is exactly conjugacy in GL(n,p). -/
theorem conjugacyLabel_complete (ha : WellFormed p n n a) (hb : WellFormed p n n b) :
    conjugacyLabel p a = conjugacyLabel p b ↔
      Conjugate (toMatrix p n n a) (toMatrix p n n b) := by
  sorry

theorem isRegular_spec (h : WellFormed p n n a) :
    isRegular p a = true ↔ (minpoly (ZMod p) (toMatrix p n n a)).natDegree = n := by
  sorry

theorem isSemisimple_spec (h : WellFormed p n n a) :
    isSemisimple p a = true ↔ Module.End.IsSemisimple (toMatrix p n n a).mulVecLin := by
  sorry

theorem elementOrder_spec (h : WellFormed p n n a) :
    elementOrder p a = orderOf (toMatrix p n n a) := by
  sorry

end Char_poly.Contract
