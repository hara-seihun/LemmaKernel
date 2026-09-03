import Mathlib
import Linear_codes.Reference
import Lk.Contract

/-!
# linear_codes: contract

Each generator matrix denotes its row space in `(Fin n → ZMod p)`. These statements connect the
executable reference to Mathlib's submodules and Hamming metric. The implementation tests use the
reference as their oracle. The `sorry` proofs below are the remaining proof obligations.
-/

namespace Linear_codes.Contract

open Matrix Lk Lk.Contract

variable {p rows n : ℕ} [Fact p.Prime] {m : Mat}

/-- Euclidean orthogonal complement with the standard coordinate dot product. -/
noncomputable def euclideanDual (C : Submodule (ZMod p) (Fin n → ZMod p)) :
    Submodule (ZMod p) (Fin n → ZMod p) where
  carrier := {x | ∀ y ∈ C, dotProduct x y = 0}
  zero_mem' := by
    sorry
  add_mem' := by
    sorry
  smul_mem' := by
    sorry

/-- The reference enumerates every row-space element exactly once. -/
theorem codewords_spec (h : WellFormed p rows n m) :
    (codewords p m).Nodup ∧
    ∀ v, WellFormedVec p n v →
      (v ∈ codewords p m ↔ toVec p n v ∈ rowSpace p rows n m) := by
  sorry

/-- The zero-code convention is 0. Otherwise the answer is attained and is the least nonzero
Hamming weight in the code. -/
theorem minimumDistance_spec (h : WellFormed p rows n m) :
    let C := rowSpace p rows n m
    (minimumDistance p m = 0 ↔ C = ⊥) ∧
    (∀ c : Fin n → ZMod p, c ∈ C → c ≠ 0 → minimumDistance p m ≤ hammingNorm c) ∧
    (C ≠ ⊥ → ∃ c : Fin n → ZMod p,
      c ∈ C ∧ c ≠ 0 ∧ hammingNorm c = minimumDistance p m) := by
  sorry

/-- Coefficient `A_w` counts the distinct codewords of Hamming weight `w`. -/
theorem weightEnumerator_spec (h : WellFormed p rows n m) (w : ℕ) (hw : w ≤ n) :
    (weightEnumerator p m).getD w 0 =
      Nat.card {c : Fin n → ZMod p //
        c ∈ rowSpace p rows n m ∧ hammingNorm c = w} := by
  sorry

/-- `dual` is a canonical basis for the Euclidean dual code. -/
theorem dual_spec (h : WellFormed p rows n m) :
    WellFormed p (dual p m).length n (dual p m) ∧
    rowSpace p (dual p m).length n (dual p m) = euclideanDual (rowSpace p rows n m) := by
  sorry

theorem isSelfDual_spec (h : WellFormed p rows n m) :
    isSelfDual p m = true ↔
      rowSpace p rows n m = euclideanDual (rowSpace p rows n m) := by
  sorry

/-- `r` is the covering radius exactly when every ambient word lies within `r` of the code and
some ambient word has distance at least `r` from every codeword. -/
theorem coveringRadius_spec (h : WellFormed p rows n m) (r : ℕ) :
    coveringRadius p m = r ↔
      (∀ x : Fin n → ZMod p, ∃ c : Fin n → ZMod p,
        c ∈ rowSpace p rows n m ∧ hammingDist x c ≤ r) ∧
      (∃ x : Fin n → ZMod p, ∀ c : Fin n → ZMod p,
        c ∈ rowSpace p rows n m → r ≤ hammingDist x c) := by
  sorry

/-- MDS means positive dimension and equality in the Singleton bound. -/
theorem isMds_spec (h : WellFormed p rows n m) :
    isMds p m = true ↔
      0 < Module.finrank (ZMod p) (rowSpace p rows n m) ∧
      minimumDistance p m = n - Module.finrank (ZMod p) (rowSpace p rows n m) + 1 := by
  sorry

def permute (σ : Equiv.Perm (Fin n)) (x : Fin n → ZMod p) : Fin n → ZMod p :=
  fun i => x (σ i)

def PreservesCode (C : Submodule (ZMod p) (Fin n → ZMod p)) (σ : Equiv.Perm (Fin n)) : Prop :=
  ∀ x, x ∈ C ↔ permute σ x ∈ C

/-- `aut_order` counts coordinate permutations only. Symbols are not scaled and field
 automorphisms are not included. -/
theorem autOrder_spec (h : WellFormed p rows n m) :
    autOrder p m = Nat.card {σ : Equiv.Perm (Fin n) // PreservesCode (rowSpace p rows n m) σ} := by
  sorry

end Linear_codes.Contract
