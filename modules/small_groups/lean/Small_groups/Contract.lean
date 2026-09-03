import Mathlib
import Small_groups.Reference
import Lk.Contract

/-!
# small_groups: contract

The reference computes with labels in a multiplication table. This file says what those labels
mean against Mathlib: list the elements of a finite group, read off its table, and each operation
of the module is the Mathlib notion under that dictionary. The proofs are stated obligations
carrying `sorry`, as in the other modules; the statements are what a caller relies on.
-/

namespace Small_groups.Contract

open Lk

/-- The table of a finite group whose elements are listed by `xs`, labelled by position. -/
def cayleyTable {G : Type*} [Group G] [DecidableEq G] (xs : List G) : Mat :=
  xs.map fun a => xs.map fun b => xs.idxOf (a * b)

variable {G : Type*} [Group G] [Fintype G] [DecidableEq G]

/-- A listing of a finite group: every element once. -/
structure Listing (G : Type*) [Group G] [DecidableEq G] where
  xs : List G
  nodup : xs.Nodup
  complete : ∀ g : G, g ∈ xs

/-- The subgroup of `G` named by a list of labels. -/
def named (l : Listing G) (labels : List Nat) : Set G := {g | l.xs.idxOf g ∈ labels}

/-- `order` is the cardinality of the group. -/
theorem order_spec (l : Listing G) : (cayleyTable l.xs).length = Nat.card G := by
  sorry

/-- `exponent` is Mathlib's `Monoid.exponent`. -/
theorem exponent_spec (l : Listing G) :
    exponent (cayleyTable l.xs) = Monoid.exponent G := by
  sorry

/-- `centre` lists exactly the centre, and in increasing label order. -/
theorem centre_spec (l : Listing G) :
    named l (centre (cayleyTable l.xs)) = (Subgroup.center G : Set G) ∧
      (centre (cayleyTable l.xs)).Pairwise (· < ·) := by
  sorry

/-- `centre_order` counts the centre. -/
theorem centreOrder_spec (l : Listing G) :
    (centre (cayleyTable l.xs)).length = Nat.card (Subgroup.center G) := by
  sorry

/-- `class_count` counts the conjugacy classes. -/
theorem classCount_spec (l : Listing G) :
    classCount (cayleyTable l.xs) = Nat.card (ConjClasses G) := by
  sorry

/-- Term `i` of `derived_series`, while the chain is still descending, is Mathlib's `derivedSeries`
at `i`; the last term repeats forever after, which is why the reference stops there. -/
theorem derivedSeries_spec (l : Listing G) (i : Nat) (hi : i < (derivedSeries (cayleyTable l.xs)).length) :
    named l ((derivedSeries (cayleyTable l.xs)).getD i []) = (_root_.derivedSeries G i : Set G) := by
  sorry

/-- The series really is stationary at its last term, so nothing was cut short. -/
theorem derivedSeries_stable (l : Listing G) :
    let series := derivedSeries (cayleyTable l.xs)
    let last := lastTerm series
    commutatorSubgroup (cayleyTable l.xs) last last = last := by
  sorry

/-- `derived_length` counts the steps of that chain: for a solvable group it is the derived
length, the least `k` with `derivedSeries G k = ⊥`. -/
theorem derivedLength_spec (l : Listing G) (h : Group.IsSolvable G) :
    derivedLength (cayleyTable l.xs) = sInf {k | _root_.derivedSeries G k = ⊥} := by
  sorry

/-- `is_solvable` decides Mathlib's `IsSolvable`. -/
theorem isSolvable_spec (l : Listing G) :
    isSolvable (cayleyTable l.xs) = true ↔ Group.IsSolvable G := by
  sorry

/-- `is_nilpotent` decides Mathlib's `Group.IsNilpotent`. -/
theorem isNilpotent_spec (l : Listing G) :
    isNilpotent (cayleyTable l.xs) = true ↔ Group.IsNilpotent G := by
  sorry

/-- `generated` gives the subgroup closure of the named elements. -/
theorem generated_spec (l : Listing G) (gens : List Nat) :
    named l (generated (cayleyTable l.xs) gens) =
      (Subgroup.closure {g : G | l.xs.idxOf g ∈ gens} : Set G) := by
  sorry

/-- `subgroups` lists every subgroup of `G` exactly once, so `subgroup_count` counts them. -/
theorem subgroupCount_spec (l : Listing G) :
    subgroupCount (cayleyTable l.xs) = Nat.card (Subgroup G) := by
  sorry

/-- `normal_subgroup_count` counts the normal subgroups. -/
theorem normalSubgroupCount_spec (l : Listing G) :
    normalSubgroupCount (cayleyTable l.xs) = Nat.card {H : Subgroup G // H.Normal} := by
  sorry

end Small_groups.Contract
