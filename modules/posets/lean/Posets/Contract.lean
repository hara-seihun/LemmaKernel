import Mathlib
import Posets.Reference

/-!
# posets: contract

The executable relation is connected here to Mathlib's finite-order vocabulary. These statements
are the meaning promised by the module. Proofs marked `sorry` remain obligations; native results
are tested against `Reference.lean`, not claimed to be formally verified.
-/

namespace Posets.Contract

open Lk Posets

structure IsPoset (r : Rel) : Prop where
  refl : ∀ i : Fin r.length, relates r i i = true
  antisymm : ∀ i j : Fin r.length, relates r i j = true → relates r j i = true → i = j
  trans : ∀ i j k : Fin r.length, relates r i j = true → relates r j k = true → relates r i k = true

@[instance_reducible]
def partialOrderOf (r : Rel) (h : IsPoset r) : PartialOrder (Fin r.length) where
  le i j := relates r i j = true
  lt i j := relates r i j = true ∧ ¬relates r j i = true
  le_refl := h.refl
  le_trans _ _ _ := h.trans _ _ _
  le_antisymm _ _ := h.antisymm _ _
  lt_iff_le_not_ge := by intros; rfl

/-- Pairwise meets and joins, stated through Mathlib's order relation rather than chosen
operations. This definition also covers the empty finite order. -/
def HasMeetJoin (α : Type*) [PartialOrder α] : Prop :=
  ∀ a b : α,
    (∃ m, m ≤ a ∧ m ≤ b ∧ ∀ z, z ≤ a → z ≤ b → z ≤ m) ∧
    (∃ j, a ≤ j ∧ b ≤ j ∧ ∀ z, a ≤ z → b ≤ z → j ≤ z)

/-- A distributive lattice stated with explicit meet and join operations and their universal
properties. -/
def IsDistributiveOrder (α : Type*) [PartialOrder α] : Prop :=
  ∃ inf sup : α → α → α,
    (∀ a b, inf a b ≤ a ∧ inf a b ≤ b ∧ ∀ z, z ≤ a → z ≤ b → z ≤ inf a b) ∧
    (∀ a b, a ≤ sup a b ∧ b ≤ sup a b ∧ ∀ z, a ≤ z → b ≤ z → sup a b ≤ z) ∧
    (∀ x y z, inf x (sup y z) = sup (inf x y) (inf x z)) ∧
    (∀ x y z, sup x (inf y z) = inf (sup x y) (sup x z))

def IsAntichainFinset {α : Type*} [PartialOrder α] (s : Finset α) : Prop :=
  ∀ x ∈ s, ∀ y ∈ s, x ≠ y → ¬x ≤ y ∧ ¬y ≤ x

def IsChainFinset {α : Type*} [PartialOrder α] (s : Finset α) : Prop :=
  ∀ x ∈ s, ∀ y ∈ s, x ≠ y → x ≤ y ∨ y ≤ x

/-! ## Presentation contracts -/

theorem relationFromMatrix_poset {m : Mat} {r : Rel} (h : relationFromMatrix m = some r) : IsPoset r := by
  sorry

theorem subsetRelation_spec {rows : Mat} {r : Rel} (h : subsetRelation rows = some r)
    (i j : Fin rows.length) :
    relates r i j = true ↔ ∀ c, c < (rows.getD i []).length →
      (rows.getD i []).getD c 0 ≠ 0 → (rows.getD j []).getD c 0 ≠ 0 := by
  sorry

theorem divisorRelation_spec {x : ℕ} {r : Rel} (h : divisorRelation x = some r)
    (i j : Fin (divisors x).length) :
    relates r i j = true ↔ (divisors x).getD i 1 ∣ (divisors x).getD j 1 := by
  sorry

/-! ## Operation contracts -/

theorem mobius_spec (r : Rel) (h : IsPoset r) :
    letI := partialOrderOf r h
    letI : LocallyFiniteOrder (Fin r.length) :=
      @Fintype.toLocallyFiniteOrder _ (partialOrderOf r h).toPreorder inferInstance
        (Classical.decRel _) (Classical.decRel _)
    ∀ i j : Fin r.length,
      ((mobiusFunction r).getD i []).getD j 0 = IncidenceAlgebra.mu ℤ i j := by
  sorry

theorem linearExtensionCount_spec (r : Rel) (h : IsPoset r) :
    letI := partialOrderOf r h
    linearExtensionCount r = Nat.card {σ : Equiv.Perm (Fin r.length) //
      ∀ i j, i ≤ j → σ i ≤ σ j} := by
  sorry

theorem isLattice_spec (r : Rel) (h : IsPoset r) :
    letI := partialOrderOf r h
    isLattice r = true ↔ HasMeetJoin (Fin r.length) := by
  sorry

theorem isDistributive_spec (r : Rel) (h : IsPoset r) :
    letI := partialOrderOf r h
    isDistributive r = true ↔ IsDistributiveOrder (Fin r.length) := by
  sorry

theorem width_spec (r : Rel) (h : IsPoset r) (w : ℕ) :
    letI := partialOrderOf r h
    width r = w ↔
      (∃ s : Finset (Fin r.length), IsAntichainFinset s ∧ s.card = w) ∧
      ∀ s : Finset (Fin r.length), IsAntichainFinset s → s.card ≤ w := by
  sorry

theorem height_spec (r : Rel) (h : IsPoset r) (d : ℕ) :
    letI := partialOrderOf r h
    height r = d ↔
      (∃ s : Finset (Fin r.length), IsChainFinset s ∧ s.card = d) ∧
      ∀ s : Finset (Fin r.length), IsChainFinset s → s.card ≤ d := by
  sorry

theorem orderPolynomial_spec (r : Rel) (h : IsPoset r) (t : ℕ) :
    letI := partialOrderOf r h
    orderPolynomial r t = Nat.card {f : Fin r.length → Fin t // Monotone f} := by
  sorry

end Posets.Contract
