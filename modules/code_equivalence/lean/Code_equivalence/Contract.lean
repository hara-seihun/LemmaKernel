import Mathlib
import Code_equivalence.Reference
import Lk.Contract

/-!
# code_equivalence: contract

What the reference computes, in Mathlib's terms. A member is read as the code its rows span. A
monomial map of `F_p^n` is a coordinate permutation together with a unit on each coordinate; the
maps whose units are all one are the coordinate permutations. Two codes are equivalent when some
map of the selected group carries one onto the other, and the representative of a class is its
least member in the Grassmannian order of `Lk.grassmannianMembers`.

Proofs marked `sorry` are open obligations. The native backends are tested against
`Reference.lean`; they are not claimed to be verified.
-/

namespace Code_equivalence.Contract

open Matrix Lk Lk.Contract Code_equivalence

/-- The code a generator matrix denotes. -/
def denotedCode (p rows n : ℕ) (m : Mat) : Submodule (ZMod p) (Fin n → ZMod p) :=
  rowSpace p rows n m

/-- The monomial matrix of a permutation and a unit per coordinate: acting on row vectors,
coordinate `j` of `v * M` is `d j` times coordinate `σ j` of `v`. -/
def monomialMatrix (p n : ℕ) (σ : Equiv.Perm (Fin n)) (d : Fin n → (ZMod p)ˣ) :
    Matrix (Fin n) (Fin n) (ZMod p) := fun i j => if σ j = i then (d j : ZMod p) else 0

/-- The group that acts: every monomial matrix when `scalars = 1`, and the permutation matrices
when `scalars = 0`. -/
def monomialGroup (p n scalars : ℕ) : Set (Matrix (Fin n) (Fin n) (ZMod p)) :=
  {a | ∃ (σ : Equiv.Perm (Fin n)) (d : Fin n → (ZMod p)ˣ),
        (scalars = 0 → ∀ j, d j = 1) ∧ a = monomialMatrix p n σ d}

/-- Codes equivalent under the selected group. -/
def Equivalent (p n scalars : ℕ) (C D : Submodule (ZMod p) (Fin n → ZMod p)) : Prop :=
  ∃ a ∈ monomialGroup p n scalars, D = C.map (Matrix.vecMulLinear a)

/-- The maps that fix the code, which is what `aut_order` reports. -/
def fixingMaps (p n scalars : ℕ) (m : Mat) : List (Vec × Vec) :=
  (monomialMaps p n scalars).filter fun sd => decide (imageCode p sd.1 sd.2 (code p n m) = code p n m)

variable {p n : ℕ} [Fact p.Prime]

/-- `code p n m` lists the codewords of the denoted code, each exactly once. -/
theorem code_spec (rows : ℕ) (m : Mat) (hm : WellFormed p rows n m) :
    (code p n m).Nodup ∧
      ∀ v : Vec, v ∈ code p n m ↔ WellFormedVec p n v ∧ toVec p n v ∈ denotedCode p rows n m := by
  sorry

/-- The maps the reference enumerates are the selected group, once each: `(p-1)^n n!` of them
with scalars, `n!` without. -/
theorem monomialMaps_spec (scalars : ℕ) (hs : scalars ≤ 1) :
    (monomialMaps p n scalars).length =
      (if scalars = 1 then (p - 1) ^ n else 1) * Nat.factorial n := by
  sorry

/-- Applying a map coordinatewise to a codeword is the action of its monomial matrix. -/
theorem imageCode_spec (σ : Equiv.Perm (Fin n)) (d : Fin n → (ZMod p)ˣ) (perm scale v : Vec)
    (hperm : ∀ j : Fin n, perm.getD j 0 = (σ j : ℕ))
    (hscale : ∀ j : Fin n, ((scale.getD j 0 : ℕ) : ZMod p) = (d j : ZMod p))
    (hv : WellFormedVec p n v) :
    toVec p n (monomialWord p perm scale v) =
      Matrix.vecMulLinear (monomialMatrix p n σ d) (toVec p n v) := by
  sorry

/-- A code of dimension `k` is the entry of `codeTable p n k` at its reported index, so the
Grassmannian order is a total order on the codes of one dimension. -/
theorem index_spec (rows : ℕ) (m : Mat) (hm : WellFormed p rows n m) :
    let k := dimOf p n (code p n m).length
    let table := codeTable p n k
    (classify p n 1 m).index < table.length ∧
      (table.getD (classify p n 1 m).index ([], [])).2 = code p n m := by
  sorry

/-- `canonical_index` is the least Grassmannian index of an equivalent code, and `is_canonical`
holds exactly on the member that attains it. -/
theorem canonical_spec (scalars rows : ℕ) (hs : scalars ≤ 1) (m : Mat) (hm : WellFormed p rows n m) :
    let k := dimOf p n (code p n m).length
    let c := classify p n scalars m
    let equivalentAt := fun i =>
      Equivalent p n scalars (denotedCode p rows n m)
        (denotedCode p k n ((grassmannianMembers p n k).getD i []))
    equivalentAt c.least ∧ (∀ i, i < c.least → ¬ equivalentAt i) ∧
      (c.index = c.least ↔ ∀ i, i < c.index → ¬ equivalentAt i) := by
  sorry

/-- The class representative is systematic: its pivots are `0, …, k-1`, so `canonical_form`
returns `[I_k | A]`. This is what lets a backend search over information sets instead of over the
whole group. -/
theorem canonicalForm_systematic (scalars rows : ℕ) (hs : scalars ≤ 1) (m : Mat)
    (hm : WellFormed p rows n m) :
    let k := dimOf p n (code p n m).length
    let form := (classify p n scalars m).form
    form.length = k ∧ IsRREF p k n form (List.range k) := by
  sorry

/-- Orbit-stabiliser: `orbit_size` counts the distinct equivalent codes, and the reported
automorphism order is the number of maps that fix the code. -/
theorem autOrder_spec (scalars rows : ℕ) (hs : scalars ≤ 1) (m : Mat) (hm : WellFormed p rows n m) :
    let c := classify p n scalars m
    let order := (monomialMaps p n scalars).length
    c.size * (order / c.size) = order ∧ order / c.size = (fixingMaps p n scalars m).length := by
  sorry

/-- Over `F_2` the only unit is one, so monomial equivalence is permutation equivalence. -/
theorem binary_monomial_is_permutation : monomialMaps 2 n 1 = monomialMaps 2 n 0 := by
  sorry

end Code_equivalence.Contract
