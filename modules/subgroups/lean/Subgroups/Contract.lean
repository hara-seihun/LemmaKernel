import Mathlib
import Subgroups.Reference

/-!
# subgroups contract

The executable reference represents a subgroup by the sorted list of all its permutation image
lists. These statements connect that finite representation to Mathlib subgroups. The remaining
proofs are explicit obligations; native code is checked against the executable reference.
-/

namespace Subgroups.Contract

open Lk

noncomputable def toPerm (n : Nat) (g : Perm) : Equiv.Perm (Fin n) := by
  classical
  sorry

noncomputable def generatedGroup (n : Nat) (gens : List Perm) : Subgroup (Equiv.Perm (Fin n)) :=
  Subgroup.closure (Set.range fun i : Fin gens.length => toPerm n (gens.getD i []))

noncomputable def representedGroup (n : Nat) (elements : List Perm) : Subgroup (Equiv.Perm (Fin n)) :=
  generatedGroup n elements


def ConjugateBy {n : Nat} (g : Equiv.Perm (Fin n))
    (H K : Subgroup (Equiv.Perm (Fin n))) : Prop :=
  ∀ x, x ∈ K ↔ g⁻¹ * x * g ∈ H


def MaximalProper {n : Nat} (G H : Subgroup (Equiv.Perm (Fin n))) : Prop :=
  H < G ∧ ∀ K : Subgroup (Equiv.Perm (Fin n)), H < K → K ≤ G → K = G


variable {n : Nat} {gens : List Perm}

/-- The executable enumeration contains each Mathlib subgroup of the generated group exactly
once. -/
theorem allSubgroups_spec (hvalid : validGenerators n gens = true) :
    (∀ h ∈ allSubgroups n gens, representedGroup n h ≤ generatedGroup n gens) ∧
    (∀ H : Subgroup (Equiv.Perm (Fin n)), H ≤ generatedGroup n gens →
      ∃! h, h ∈ allSubgroups n gens ∧ representedGroup n h = H) := by
  sorry

/-- `subgroup_count` counts all labelled subgroups, before quotienting by conjugacy. -/
theorem subgroupCount_spec (hvalid : validGenerators n gens = true) :
    (allSubgroups n gens).length =
      Set.ncard {H : Subgroup (Equiv.Perm (Fin n)) | H ≤ generatedGroup n gens} := by
  sorry

/-- The returned list is a transversal for conjugation by the parent group. The executable
lexicographic minimum in each class fixes the representative uniquely. -/
theorem conjugacyClasses_spec (hvalid : validGenerators n gens = true) :
    ∀ K ∈ allSubgroups n gens,
      ∃! H, H ∈ conjugacyClasses n gens ∧
        ∃ g ∈ generatedGroup n gens,
          ConjugateBy g (representedGroup n H) (representedGroup n K) := by
  sorry

/-- The maximal-subgroup output contains exactly one representative from each conjugacy class of
maximal proper subgroups. -/
theorem maximalSubgroups_spec (hvalid : validGenerators n gens = true) :
    (∀ H ∈ maximalSubgroups n gens,
      MaximalProper (generatedGroup n gens) (representedGroup n H)) ∧
    (∀ K : Subgroup (Equiv.Perm (Fin n)), MaximalProper (generatedGroup n gens) K →
      ∃! H, H ∈ maximalSubgroups n gens ∧
        ∃ g ∈ generatedGroup n gens, ConjugateBy g (representedGroup n H) K) := by
  sorry

/-- `is_normal` also checks containment, so a candidate generated outside the parent is false. -/
theorem normalIn_spec {parent candidate : List Perm}
    (hp : validGenerators n parent = true) (hc : validGenerators n candidate = true) :
    normalIn n parent candidate = true ↔
      representedGroup n candidate ≤ generatedGroup n parent ∧
      ∀ g ∈ generatedGroup n parent, ∀ h ∈ representedGroup n candidate,
        g⁻¹ * h * g ∈ representedGroup n candidate := by
  sorry

end Subgroups.Contract
