import Mathlib
import Gfp.Reference

/-!
# gfp: contract

What each output of `Reference.lean` means in Mathlib's terms. These are the statements a caller
relies on when they use the module. They are stated, not proved: the native backends are checked
against `Reference.lean` by `decide`, and `Reference.lean` is related to Mathlib here. A theorem
below with `sorry` is an obligation that has been written down and not yet discharged; there is
no other tooling that would make the native code "verified", and we do not claim it is.

Conventions: a `Mat` is well formed for `(p, rows, cols)` when it has `rows` rows of `cols`
entries each below `p`. `toMatrix` reads it as a Mathlib matrix over `ZMod p`.
-/

namespace Gfp.Contract

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

/-- Reduced row echelon form, spelled out: nonzero rows come first; each nonzero row has leading
entry 1 at a column strictly right of the previous row's; and every leading column is zero in
every other row. -/
def IsRREF (p rows cols : ℕ) (m : Mat) : Prop :=
  let R := toMatrix p rows cols m
  let piv := pivots p m
  piv.Pairwise (· < ·) ∧ (∀ c ∈ piv, c < cols) ∧
  (∀ i : Fin rows, (i : ℕ) < piv.length ↔ R i ≠ 0) ∧
  (∀ i : Fin rows, ∀ hi : (i : ℕ) < piv.length,
      ∀ j : Fin cols, (j : ℕ) < piv[i] → R i j = 0) ∧
  (∀ i : Fin rows, ∀ hi : (i : ℕ) < piv.length,
      ∀ j : Fin cols, (j : ℕ) = piv[i] → R i j = 1) ∧
  (∀ i k : Fin rows, ∀ hi : (i : ℕ) < piv.length, k ≠ i →
      ∀ j : Fin cols, (j : ℕ) = piv[i] → R k j = 0)

variable {p rows cols : ℕ} [Fact p.Prime] {m : Mat}

/-! ## Operations -/

theorem rank_spec (h : WellFormed p rows cols m) :
    rank p m = (toMatrix p rows cols m).rank := by
  sorry

theorem nullity_spec (h : WellFormed p rows cols m) :
    nullity p m = Module.finrank (ZMod p) (LinearMap.ker (toMatrix p rows cols m).mulVecLin) := by
  sorry

theorem rref_spec (h : WellFormed p rows cols m) :
    WellFormed p rows cols (rref p m) ∧ IsRREF p rows cols (rref p m) ∧
    rowSpace p rows cols (rref p m) = rowSpace p rows cols m := by
  sorry

/-- The rref of a row space is unique, so `rref` is a canonical form for row spaces. -/
theorem rref_canonical {m' : Mat} (h : WellFormed p rows cols m) (h' : WellFormed p rows cols m') :
    rowSpace p rows cols m = rowSpace p rows cols m' ↔ rref p m = rref p m' := by
  sorry

theorem nullspace_spec (h : WellFormed p rows cols m) :
    let vs := (nullspace p m).map (toVec p cols)
    LinearIndependent (ZMod p) (fun i : Fin vs.length => vs[i]) ∧
    Submodule.span (ZMod p) (Set.range fun i : Fin vs.length => vs[i]) =
      LinearMap.ker (toMatrix p rows cols m).mulVecLin := by
  sorry

theorem inSpan_spec {t : Vec} (h : WellFormed p rows cols m) (ht : WellFormedVec p cols t) :
    inSpan p m t = true ↔ toVec p cols t ∈ rowSpace p rows cols m := by
  sorry

theorem solve_spec {b : Vec} (h : WellFormed p rows cols m) (hb : WellFormedVec p rows b) :
    (∀ x, solve p m b = some x →
        (toMatrix p rows cols m).mulVec (toVec p cols x) = toVec p rows b) ∧
    (solve p m b = none → ∀ x : Fin cols → ZMod p, (toMatrix p rows cols m).mulVec x ≠ toVec p rows b) := by
  sorry

theorem inverse_spec {n : ℕ} (h : WellFormed p n n m) :
    (∀ b, inverse p m = some b → toMatrix p n n m * toMatrix p n n b = 1) ∧
    (inverse p m = none → ¬ IsUnit (toMatrix p n n m)) := by
  sorry

theorem witness_spec (h : WellFormed p rows cols m) :
    let (r, t) := witness p m
    r = rref p m ∧ IsUnit (toMatrix p rows rows t) ∧
    toMatrix p rows rows t * toMatrix p rows cols m = toMatrix p rows cols r := by
  sorry

/-! ## Families

Each family's member list is what the manifest says, in the order the reference defines. -/

theorem allMatrices_spec (p rows cols : ℕ) :
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
    ms.Nodup ∧ (∀ m ∈ ms, WellFormed p h n m ∧ IsRREF p h n m ∧ rank p m = h) ∧
    ∀ W : Submodule (ZMod p) (Fin n → ZMod p), Module.finrank (ZMod p) W = h →
      ∃ m ∈ ms, rowSpace p h n m = W := by
  sorry

theorem transform_spec (f : Family) (c : Mat) :
    (Family.transform f c).members = f.members.map fun m => matmul f.p m c := by
  rfl

theorem stack_spec (f : Family) (rows : Mat) :
    (Family.stack f rows).members = f.members.map (· ++ rows) := by
  rfl

end Gfp.Contract
