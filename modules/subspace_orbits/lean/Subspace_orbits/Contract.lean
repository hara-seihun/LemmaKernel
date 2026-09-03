import Mathlib
import Subspace_orbits.Reference
import Lk.Contract

/-!
# subspace_orbits: contract

Each matrix is interpreted as the subspace spanned by its rows. The supplied invertible matrices
act on those subspaces by right multiplication. The executable reference removes zero rows from
rref and orders subspaces by the canonical Grassmannian enumeration.

The statements here connect that computation to Mathlib. Proofs marked `sorry` are open proof
obligations. Native backends are tested against `Reference.lean`, not claimed to be verified.
-/

namespace Subspace_orbits.Contract

open Matrix Lk Lk.Contract Subspace_orbits

/-- The Mathlib subspace represented by a row matrix. -/
def representedSubspace (p n : ℕ) (m : Mat) : Submodule (ZMod p) (Fin n → ZMod p) :=
  rowSpace p m.length n m

/-- The subgroup of permutations of subspaces induced by the generator matrices. -/
noncomputable def generatedAction (p n : ℕ) (gens : List Mat) :
    Subgroup (Equiv.Perm (Submodule (ZMod p) (Fin n → ZMod p))) := by
  sorry

variable {p n : ℕ} [Fact p.Prime]

/-- Removing zero rows from rref preserves exactly the represented subspace. -/
theorem subspaceRref_spec (m : Mat) (hcols : ∀ r ∈ m, r.length = n) :
    representedSubspace p n (subspaceRref p m) = representedSubspace p n m := by
  sorry

/-- The reference orbit contains each member of the Mathlib orbit exactly once. -/
theorem orbit_spec (gens : List Mat) (m : Mat)
    (hgen : ∀ a ∈ gens, WellFormed p n n a ∧ Matrix.det (toMatrix p n n a) ≠ 0)
    (hm : ∀ r ∈ m, r.length = n) :
    (orbit p gens m).Nodup ∧
      ∀ r, r ∈ orbit p gens m ↔
        representedSubspace p n r ∈
          MulAction.orbit (generatedAction p n gens) (representedSubspace p n m) := by
  sorry

/-- `canonical_index` selects the least Grassmannian index in the mathematical orbit. -/
theorem canonicalIndex_spec (f : Family) (gens : List Mat) (projective : ℕ) (xs : List ℕ)
    (hact : actsOn (.mats p gens) projective f = true)
    (hout : run (.canonicalIndex (.mats p gens) projective) f .all = .integers xs) :
    ∀ i, i < f.members.length →
      xs.getD i 0 = leastOrbitIndex p (familyCols f) (orbit p gens (f.members.getD i [])) := by
  sorry

/-- `is_canonical` is true exactly on an orbit's least rref representative. -/
theorem isCanonical_spec (f : Family) (gens : List Mat) (projective : ℕ) (xs : List ℕ)
    (hact : actsOn (.mats p gens) projective f = true)
    (hout : run (.isCanonical (.mats p gens) projective) f .all = .integers xs) :
    ∀ i, i < f.members.length →
      (xs.getD i 0 = 1 ↔
        subspaceIndex p (familyCols f) (f.members.getD i []) =
          leastOrbitIndex p (familyCols f) (orbit p gens (f.members.getD i []))) := by
  sorry

/-- Orbit size times the reported stabilizer order is the selected group order. For
`projective = 0` this is the generated matrix group. For `projective = 1` scalar multiples are
identified, so it is the induced PGL group. -/
theorem stabilizerOrder_spec (gens : List Mat) (projective : ℕ) (m : Mat)
    (hprojective : projective ≤ 1)
    (hgen : ∀ a ∈ gens, WellFormed p n n a ∧ Matrix.det (toMatrix p n n a) ≠ 0) :
    let o := orbit p gens m
    o.length * ((matrixElements p projective gens).length / o.length) =
      (matrixElements p projective gens).length := by
  sorry

/-- Nonzero scalar multiples induce the same subspace action. -/
theorem projective_action_spec (a m : Mat) (ha : Matrix.det (toMatrix p n n a) ≠ 0)
    (hm : ∀ r ∈ m, r.length = n) :
    representedSubspace p n (subspaceRref p (matmul p m (normaliseMatrix p a))) =
      representedSubspace p n (subspaceRref p (matmul p m a)) := by
  sorry

/-- Frobenius is the identity on the prime field. Thus PGammaL has the same action as PGL in the
runtime's current field domain. -/
theorem primeField_frobenius (x : ZMod p) : x ^ p = x := by
  sorry

end Subspace_orbits.Contract
