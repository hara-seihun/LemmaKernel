import Mathlib
import Lk.Reference

/-!
# Runtime contract

What the families of `Lk/Reference.lean` mean in Mathlib's terms: each family lists exactly the
objects its manifest entry names, once, in the order the reference defines. Stated, not proved:
a theorem with `sorry` is an obligation written down and not yet discharged.

Conventions: a `Mat` is well formed for `(p, rows, cols)` when it has `rows` rows of `cols`
entries each below `p`. `toMatrix` reads it as a Mathlib matrix over `ZMod p`.
-/

namespace Lk.Contract

open Matrix

def WellFormed (p rows cols : ℕ) (m : Mat) : Prop :=
  m.length = rows ∧ ∀ r ∈ m, r.length = cols ∧ ∀ x ∈ r, x < p

def WellFormedVec (p cols : ℕ) (v : Vec) : Prop := v.length = cols ∧ ∀ x ∈ v, x < p

def toMatrix (p rows cols : ℕ) (m : Mat) : Matrix (Fin rows) (Fin cols) (ZMod p) :=
  fun i j => ((m.getD i []).getD j 0 : ZMod p)

def toVec (p cols : ℕ) (v : Vec) : Fin cols → ZMod p := fun j => (v.getD j 0 : ZMod p)

/-- The row space of a matrix. -/
def rowSpace (p rows cols : ℕ) (m : Mat) : Submodule (ZMod p) (Fin cols → ZMod p) :=
  Submodule.span (ZMod p) (Set.range (toMatrix p rows cols m))

/-- Reduced row echelon form with pivot columns `piv`, spelled out: nonzero rows come first;
each nonzero row has leading entry 1 at a column strictly right of the previous row's; and every
leading column is zero in every other row. -/
def IsRREF (p rows cols : ℕ) (m : Mat) (piv : List ℕ) : Prop :=
  let R := toMatrix p rows cols m
  piv.Pairwise (· < ·) ∧ (∀ c ∈ piv, c < cols) ∧
  (∀ i : Fin rows, (i : ℕ) < piv.length ↔ R i ≠ 0) ∧
  (∀ i : Fin rows, ∀ hi : (i : ℕ) < piv.length,
      ∀ j : Fin cols, (j : ℕ) < piv[i] → R i j = 0) ∧
  (∀ i : Fin rows, ∀ hi : (i : ℕ) < piv.length,
      ∀ j : Fin cols, (j : ℕ) = piv[i] → R i j = 1) ∧
  (∀ i k : Fin rows, ∀ hi : (i : ℕ) < piv.length, k ≠ i →
      ∀ j : Fin cols, (j : ℕ) = piv[i] → R k j = 0)

variable {p : ℕ} [Fact p.Prime]

theorem allMatrices_spec (rows cols : ℕ) :
    let ms := (Family.allMatrices p rows cols).members
    ms.Nodup ∧ ms.length = p ^ (rows * cols) ∧ ∀ m, WellFormed p rows cols m → m ∈ ms := by
  sorry

/-- Every `k`-subset of the dictionary once (in the order `combos` defines: lexicographic by
chosen positions). -/
theorem subsets_spec (d : List Vec) (k : ℕ) :
    (Family.subsets p d k).members.Perm (List.sublistsLen k d) := by
  sorry

/-- The Grassmannian lists every `h`-dimensional subspace of `F_p^n` exactly once, as its rref. -/
theorem grassmannian_spec (n h : ℕ) :
    let ms := (Family.grassmannian p n h).members
    ms.Nodup ∧ (∀ m ∈ ms, WellFormed p h n m ∧ ∃ piv, piv.length = h ∧ IsRREF p h n m piv) ∧
    ∀ W : Submodule (ZMod p) (Fin n → ZMod p), Module.finrank (ZMod p) W = h →
      ∃ m ∈ ms, rowSpace p h n m = W := by
  sorry

theorem transform_spec (f : Family) (c : Mat) :
    (Family.transform f c).members = f.members.map fun m => matmul f.p m c := by
  rfl

theorem stack_spec (f : Family) (rows : Mat) :
    (Family.stack f rows).members = f.members.map (· ++ rows) := by
  rfl

/-- Every `k`-subset of the inner family's members, each flattened to one row, once. -/
theorem subsetsOf_spec (f : Family) (k : ℕ) :
    (Family.subsetsOf f k).members.Perm (List.sublistsLen k (f.members.map List.flatten)) := by
  sorry

/-- Every symmetric `n x n` matrix over `F_p` exactly once. -/
theorem symmetricMatrices_spec (n : ℕ) :
    let ms := (Family.symmetricMatrices p n).members
    ms.Nodup ∧ ms.length = p ^ (n * (n + 1) / 2) ∧
    ∀ m, WellFormed p n n m → (toMatrix p n n m).IsSymm → m ∈ ms := by
  sorry

theorem range_spec (a b : ℕ) :
    (Family.range a b).members = (List.range (b - a)).map fun i => [[a + i]] := by
  rfl

/-- Every word of the given length over `0..alphabet-1` exactly once. -/
theorem words_spec (q len : ℕ) :
    let ms := (Family.words q len).members
    ms.Nodup ∧ ms.length = q ^ len ∧ ∀ w : List ℕ, w.length = len → (∀ x ∈ w, x < q) → [w] ∈ ms := by
  sorry

/-- `permElements` is the generated subgroup of `Equiv.Perm (Fin n)`, each element once. -/
theorem permElements_spec (gens : List Perm) (n : ℕ) (h : gens.all (·.length = n) = true) :
    (permElements gens).Nodup ∧ (permElements gens).length = Nat.card (Subgroup.closure
      {σ : Equiv.Perm (Fin n) | ∃ g ∈ gens, ∀ i : Fin n, (σ i : ℕ) = g.getD i 0}) := by
  sorry

end Lk.Contract
