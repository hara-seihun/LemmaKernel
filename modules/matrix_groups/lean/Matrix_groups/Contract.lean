import Mathlib
import Matrix_groups.Reference
import Lk.Contract

/-!
# matrix_groups: contract

The executable reference uses row vectors. This file states the corresponding Mathlib objects and
properties. Proofs are left as explicit obligations; native backends are checked against the
executable reference, not proved correct here.
-/

namespace Matrix_groups.Contract

open Matrix Lk Lk.Contract

variable {p n : ℕ} [Fact p.Prime]

abbrev Space (p n : ℕ) := Fin n → ZMod p
abbrev Square (p n : ℕ) := Matrix (Fin n) (Fin n) (ZMod p)

def generated (p n : ℕ) (gens : List Mat) : Submonoid (Square p n) :=
  Submonoid.closure {a | ∃ g ∈ gens, a = toMatrix p n n g}

/-- The linear right action `v ↦ v A`. -/
def rightAction (p n : ℕ) (a : Mat) : Space p n →ₗ[ZMod p] Space p n :=
  (toMatrix p n n a).vecMulLinear

def IsInvariant (p n : ℕ) (gens : List Mat) (w : Submodule (ZMod p) (Space p n)) : Prop :=
  ∀ a ∈ gens, Submodule.map (rightAction p n a) w ≤ w

def MathlibIrreducible (p n : ℕ) (gens : List Mat) : Prop :=
  ∀ w : Submodule (ZMod p) (Space p n), IsInvariant p n gens w → w = ⊥ ∨ w = ⊤

/-- For an irreducible representation over a finite field, absolute irreducibility is equivalent
to the commuting algebra containing only scalars. -/
def MathlibAbsolutelyIrreducible (p n : ℕ) (gens : List Mat) : Prop :=
  MathlibIrreducible p n gens ∧
    ∀ x : Square p n, (∀ a ∈ gens, x * toMatrix p n n a = toMatrix p n n a * x) →
      ∃ c : ZMod p, x = c • (1 : Square p n)

def PreservesNondegenerateForm (p n : ℕ) (gens : List Mat) : Prop :=
  ∃ b : Square p n, IsUnit b ∧
    ∀ a ∈ gens, toMatrix p n n a * b * (toMatrix p n n a)ᵀ = b

/-- A nontrivial direct decomposition which the generated group permutes transitively. -/
def MathlibImprimitive (p n : ℕ) (gens : List Mat) : Prop :=
  ∃ r : ℕ, ∃ blocks : Fin r → Submodule (ZMod p) (Space p n),
    1 < r ∧
    (∀ i, blocks i ≠ ⊥) ∧
    DirectSum.IsInternal blocks ∧
    (∀ a : generated p n gens, ∀ i, ∃ j,
      Submodule.map a.1.vecMulLinear (blocks i) = blocks j) ∧
    (∀ i j, ∃ a : generated p n gens,
      Submodule.map a.1.vecMulLinear (blocks i) = blocks j)


theorem order_spec (gens : List Mat)
    (h : ∀ a ∈ gens, WellFormed p n n a ∧ Matrix.rank (toMatrix p n n a) = n) :
    groupOrder p gens = Nat.card (generated p n gens) := by
  sorry

theorem irreducible_spec (gens : List Mat)
    (h : ∀ a ∈ gens, WellFormed p n n a ∧ Matrix.rank (toMatrix p n n a) = n) :
    isIrreducible p n gens = true ↔ MathlibIrreducible p n gens := by
  sorry

theorem absolutelyIrreducible_spec (gens : List Mat)
    (h : ∀ a ∈ gens, WellFormed p n n a ∧ Matrix.rank (toMatrix p n n a) = n) :
    isAbsolutelyIrreducible p n gens = true ↔ MathlibAbsolutelyIrreducible p n gens := by
  sorry

theorem preservesForm_spec (gens : List Mat)
    (h : ∀ a ∈ gens, WellFormed p n n a ∧ Matrix.rank (toMatrix p n n a) = n) :
    preservesForm p n gens = true ↔ PreservesNondegenerateForm p n gens := by
  sorry

theorem imprimitive_spec (gens : List Mat)
    (h : ∀ a ∈ gens, WellFormed p n n a ∧ Matrix.rank (toMatrix p n n a) = n) :
    isImprimitive p n gens = true ↔
      MathlibIrreducible p n gens ∧ MathlibImprimitive p n gens := by
  sorry

end Matrix_groups.Contract
