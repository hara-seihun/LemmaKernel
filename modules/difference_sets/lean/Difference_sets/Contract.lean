import Mathlib
import Difference_sets.Reference
import Lk.Contract

/-!
# difference_sets: contract

The statements below relate the executable list representation to finite permutations in Mathlib.
They are obligations, not completed proofs. Native backends are checked against
`Reference.lean`; these theorems state the mathematics represented by that oracle.
-/

namespace Difference_sets.Contract

open Lk

/-- A list is a permutation of `Fin n`. -/
def IsPerm (n : Nat) (g : Perm) : Prop :=
  g.length = n ∧ g.Nodup ∧ ∀ x ∈ g, x < n

/-- The Mathlib permutation represented by a valid image list. -/
noncomputable def toEquiv (n : Nat) (g : Perm) (h : IsPerm n g) : Equiv.Perm (Fin n) := by
  sorry

/-- Count ordered pairs whose right quotient is `g`, using Mathlib's permutation group. -/
noncomputable def quotientCount (n : Nat) (d : List Perm) (g : Perm) : Nat := by
  classical
  exact ((Finset.univ : Finset (Fin d.length × Fin d.length)).filter fun ij : Fin d.length × Fin d.length =>
    ∃ hx : IsPerm n (d.getD ij.1 []), ∃ hy : IsPerm n (d.getD ij.2 []), ∃ hg : IsPerm n g,
      toEquiv n (d.getD ij.1 []) hx * (toEquiv n (d.getD ij.2 []) hy)⁻¹ = toEquiv n g hg).card

/-- Every entry of the executable multiset is the corresponding Mathlib quotient count. -/
theorem differenceMultiset_spec (n : Nat) (elems d : List Perm)
    (he : ∀ g ∈ elems, IsPerm n g) (hd : ∀ g ∈ d, IsPerm n g) (i : Nat) (hi : i < elems.length) :
    (differenceMultiset elems d).getD i 0 = quotientCount n d (elems.getD i []) := by
  sorry

/-- `is_difference_set` means that every nonidentity group element has the forced common
multiplicity `k(k-1)/(v-1)`. -/
theorem isDifferenceSet_spec (n : Nat) (elems d : List Perm)
    (he : ∀ g ∈ elems, IsPerm n g) (hd : ∀ g ∈ d, IsPerm n g) (hv : 1 < elems.length) :
    isDifferenceSet elems d = true ↔
      d.length * (d.length - 1) % (elems.length - 1) = 0 ∧
      ∀ g ∈ elems, g ≠ identityPerm n →
        quotientCount n d g = d.length * (d.length - 1) / (elems.length - 1) := by
  sorry

/-- `is_pds` is the regular partial difference-set condition. Identity is excluded, inversion
preserves `D`, and quotient multiplicities are `lambda` on `D` and `mu` off `D`. -/
theorem isPds_spec (n lam mu : Nat) (elems d : List Perm)
    (he : ∀ g ∈ elems, IsPerm n g) (hd : ∀ g ∈ d, IsPerm n g) :
    isPds elems d lam mu = true ↔
      identityPerm n ∉ d ∧ (∀ x ∈ d, inversePerm x ∈ d) ∧
      d.length * (d.length - 1) = d.length * lam + (elems.length - 1 - d.length) * mu ∧
      ∀ g ∈ elems, g ≠ identityPerm n →
        quotientCount n d g = if g ∈ d then lam else mu := by
  sorry

/-- `is_relative_difference_set` forbids every nonidentity quotient in `N` and gives every
quotient outside `N` the forced common multiplicity. -/
theorem isRelativeDifferenceSet_spec (n : Nat) (elems d forbidden : List Perm)
    (he : ∀ g ∈ elems, IsPerm n g) (hd : ∀ g ∈ d, IsPerm n g)
    (hf : validForbidden elems forbidden = true) :
    isRelativeDifferenceSet elems d forbidden = true ↔
      let N := permElements forbidden
      let outside := elems.length - N.length
      outside ≠ 0 ∧ d.length * (d.length - 1) % outside = 0 ∧
      (∀ g ∈ N, g ≠ identityPerm n → quotientCount n d g = 0) ∧
      ∀ g ∈ elems, g ∉ N → quotientCount n d g = d.length * (d.length - 1) / outside := by
  sorry

end Difference_sets.Contract
