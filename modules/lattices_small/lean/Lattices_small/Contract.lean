import Mathlib
import Lattices_small.Reference
import Lk.Contract

/-!
# Integral lattices: contract

These statements relate the executable reference to Mathlib matrices and positive-definite
quadratic forms. Proofs left as `sorry` are explicit obligations. Native implementations are
checked against `Reference.lean`; they are not formally verified.
-/

namespace Lattices_small.Contract

open Matrix Lk Lattices_small


def toZMatrix (n : Nat) (g : Mat) : Matrix (Fin n) (Fin n) Int :=
  fun i j => decodeInt ((g.getD i []).getD j 0)

def toRMatrix (n : Nat) (g : Mat) : Matrix (Fin n) (Fin n) Real :=
  fun i j => decodeInt ((g.getD i []).getD j 0)

def quadratic (n : Nat) (g : Mat) (x : Fin n → Int) : Int :=
  x ⬝ᵥ (toZMatrix n g *ᵥ x)

def listVector (n : Nat) (x : List Int) : Fin n → Int := fun i => x.getD i 0

def WellFormedGram (n : Nat) (g : Mat) : Prop :=
  g.length = n ∧ (∀ row ∈ g, row.length = n) ∧ (toRMatrix n g).PosDef

/-- The reported minimum is the least positive value of the integral quadratic form. -/
theorem minimum_spec {n : Nat} {g : Mat} (h : WellFormedGram n g) :
    (∃ x : Fin n → Int, x ≠ 0 ∧ quadratic n g x = minimum g) ∧
    ∀ x : Fin n → Int, x ≠ 0 → (minimum g : Int) ≤ quadratic n g x := by
  sorry

/-- The kissing number counts signed minimal vectors, so `x` and `-x` are separate unless zero. -/
theorem kissingNumber_spec {n : Nat} {g : Mat} (h : WellFormedGram n g) :
    kissingNumber g = Set.ncard {x : Fin n → Int | x ≠ 0 ∧ quadratic n g x = minimum g} := by
  sorry

/-- Coefficient `k` counts all lattice vectors of squared norm `k`, including zero at `k = 0`. -/
theorem thetaSeries_spec {n bound k : Nat} {g : Mat} (h : WellFormedGram n g) (hk : k ≤ bound) :
    (thetaSeries g bound).getD k 0 = Set.ncard {x : Fin n → Int | quadratic n g x = k} := by
  sorry

theorem isUnimodular_spec {n : Nat} {g : Mat} (h : WellFormedGram n g) :
    isUnimodular g = true ↔ IsUnit (toZMatrix n g).det := by
  sorry

theorem isEven_spec {n : Nat} {g : Mat} (h : WellFormedGram n g) :
    isEven g = true ↔ ∀ x : Fin n → Int, Even (quadratic n g x) := by
  sorry

/-- `shortVectors` is complete, has no duplicates, and uses signed lexicographic order. -/
theorem shortVectors_spec {n bound : Nat} {g : Mat} (h : WellFormedGram n g) :
    (shortVectors g bound).Nodup ∧
    (∀ x ∈ shortVectors g bound, x.length = n) ∧
    ∀ x : List Int, x.length = n →
      (x ∈ shortVectors g bound ↔ x ≠ List.replicate n 0 ∧ quadratic n g (listVector n x) ≤ bound) := by
  sorry

/-- The runtime's sublattice family uses the Gram matrix pulled back along each canonical upper
row-Hermite-normal-form basis matrix. Those matrices have positive diagonal product `index`. -/
theorem sublattices_spec (g : Mat) (index : Nat) :
    (Family.sublattices g index).members =
      (hnfMatrices g.length index).map (sublatticeGram g) := by
  rfl

end Lattices_small.Contract
