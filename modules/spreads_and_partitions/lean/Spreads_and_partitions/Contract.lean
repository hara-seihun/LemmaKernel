import Mathlib
import Spreads_and_partitions.Reference
import Lk.Contract

/-!
# spreads_and_partitions: contract

What each output of `Reference.lean` means in Mathlib's terms: these are the statements a caller
relies on when they use the module. They are stated, not proved: the backend is checked against
`Reference.lean` by `decide`, and `Reference.lean` is related to Mathlib here. A theorem below
with `sorry` is an obligation that has been written down and not yet discharged; there is no
tooling here that would make the native code "verified", and we do not claim it is.

`WellFormed`, `toMatrix` and `rowSpace` come from `Lk.Contract`.
-/

namespace Spreads_and_partitions.Contract

open Lk Lk.Contract

variable {p n h : ℕ} [Fact p.Prime]

/-- The subspace one row of a member stands for: the row read as an `h x n` matrix, row-major,
and then its row space. -/
def component (p n h : ℕ) (row : Vec) : Submodule (ZMod p) (Fin n → ZMod p) :=
  rowSpace p h n (chunk n h row)

/-- The components of a member, in row order. -/
def comps (p n h : ℕ) (m : Mat) : List (Submodule (ZMod p) (Fin n → ZMod p)) :=
  m.map (component p n h)

/-- The components of one row of a packing candidate, in block order. -/
def rowComps (p n h : ℕ) (row : Vec) : List (Submodule (ZMod p) (Fin n → ZMod p)) :=
  (blocks n h row).map (rowSpace p h n)

/-- A member of a family this module accepts: `k` rows of `h * n` entries below `p`. -/
def Member (p n h : ℕ) (m : Mat) : Prop := ∀ r ∈ m, r.length = h * n ∧ ∀ x ∈ r, x < p

/-- The components of `m`, as a family indexed by position, and what "covers exactly once"
means for them: every nonzero vector of `F_p^n` lies in exactly one. -/
def PartitionsNonzero (cs : List (Submodule (ZMod p) (Fin n → ZMod p))) : Prop :=
  ∀ v : Fin n → ZMod p, v ≠ 0 → ∃! i : Fin cs.length, v ∈ cs[i]

/-! ## Elimination

The basis every operation is computed from, and the canonical form that decides when two
components are the same subspace. -/

theorem echelon_rowSpace {rows cols : ℕ} {m : Mat} (hm : WellFormed p rows cols m) :
    rowSpace p (echelon p m).length cols (echelon p m) = rowSpace p rows cols m := by
  sorry

theorem echelon_length {rows cols : ℕ} {m : Mat} (hm : WellFormed p rows cols m) :
    (echelon p m).length = Module.finrank (ZMod p) (rowSpace p rows cols m) := by
  sorry

/-- `rrefOf` is a canonical form: two lists of rows have the same reduced basis exactly when they
span the same subspace. This is what `is_packing` compares components by. -/
theorem rrefOf_canonical {rows rows' cols : ℕ} {m m' : Mat} (hm : WellFormed p rows cols m)
    (hm' : WellFormed p rows' cols m') :
    rrefOf p m = rrefOf p m' ↔ rowSpace p rows cols m = rowSpace p rows' cols m' := by
  sorry

theorem meetsTrivially_spec {rows rows' cols : ℕ} {a b : Mat} (ha : WellFormed p rows cols a)
    (hb : WellFormed p rows' cols b) :
    meetsTrivially p (echelon p a) (echelon p b) = true ↔
      rowSpace p rows cols a ⊓ rowSpace p rows' cols b = ⊥ := by
  sorry

/-! ## Operations -/

/-- A partial spread: nonzero components of one dimension, meeting pairwise in `0` only. -/
theorem isPartialSpread_spec {m : Mat} (hm : Member p n h m) :
    isPartialSpread p (components p n m) = true ↔
      (∀ W ∈ comps p n h m, W ≠ ⊥) ∧
      (∀ W ∈ comps p n h m, ∀ V ∈ comps p n h m,
          Module.finrank (ZMod p) W = Module.finrank (ZMod p) V) ∧
      (comps p n h m).Pairwise (fun W V => W ⊓ V = ⊥) := by
  sorry

/-- A vector space partition: every nonzero vector of `F_p^n` in exactly one component. -/
theorem isPartition_spec {m : Mat} (hm : Member p n h m) :
    isPartition p n (components p n m) = true ↔
      (∀ W ∈ comps p n h m, W ≠ ⊥) ∧ PartitionsNonzero (comps p n h m) := by
  sorry

/-- A spread: a vector space partition into components of one dimension. -/
theorem isSpread_spec {m : Mat} (hm : Member p n h m) :
    isSpread p n (components p n m) = true ↔
      isPartialSpread p (components p n m) = true ∧ PartitionsNonzero (comps p n h m) := by
  sorry

/-- The count `intersecting_pairs` returns: pairs of positions whose components meet. -/
theorem intersecting_spec {m : Mat} (hm : Member p n h m) :
    intersecting p (components p n m) =
      ((List.range (comps p n h m).length).flatMap fun i =>
        ((List.range i).filter fun j =>
          decide ((comps p n h m).getD i ⊥ ⊓ (comps p n h m).getD j ⊥ ≠ ⊥))).length := by
  sorry

/-- The Gaussian binomial counts the `h`-dimensional subspaces of `F_p^n`. -/
theorem gaussBinom_spec :
    gaussBinom p n h =
      Nat.card {W : Submodule (ZMod p) (Fin n → ZMod p) // Module.finrank (ZMod p) W = h} := by
  sorry

/-- A packing: every row a spread by `h`-dimensional subspaces, and every `h`-dimensional
subspace of `F_p^n` a component of exactly one row. -/
theorem isPacking_spec {m : Mat} (hm : ∀ r ∈ m, ∀ x ∈ r, x < p) :
    isPacking p n h m = true ↔
      (∀ row ∈ m, (∀ W ∈ rowComps p n h row, Module.finrank (ZMod p) W = h) ∧
                  PartitionsNonzero (rowComps p n h row)) ∧
      ∀ W : Submodule (ZMod p) (Fin n → ZMod p), Module.finrank (ZMod p) W = h →
        ∃! ir : ℕ × ℕ, ir.1 < m.length ∧ ir.2 < (rowComps p n h (m.getD ir.1 [])).length ∧
          (rowComps p n h (m.getD ir.1 [])).getD ir.2 ⊥ = W := by
  sorry

end Spreads_and_partitions.Contract
