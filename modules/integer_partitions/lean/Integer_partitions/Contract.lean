import Mathlib
import Integer_partitions.Reference

/-!
# Constrained integer partitions and compositions: contract

The runtime represents a partition or composition of `n` by a row of length `n`. Positive entries
are its parts and the remaining entries are zero. The family theorems state the constraints on the
positive prefix. Rank and crank use the standard integer-valued definitions; the executable result
adds `n` to fit the runtime's natural-number reductions.
-/

namespace Integer_partitions.Contract

open Lk Integer_partitions


def IsPartitionOf (total : Nat) (xs : List Nat) : Prop :=
  xs.sum = total ∧ (∀ x ∈ xs, 0 < x) ∧ List.Pairwise (fun x y => x ≥ y) xs

def PartitionConstraints (maxPart maxParts maxMultiplicity distinct odd : Nat) (xs : List Nat) : Prop :=
  (maxPart = 0 ∨ ∀ x ∈ xs, x ≤ maxPart) ∧
  (maxParts = 0 ∨ xs.length ≤ maxParts) ∧
  (maxMultiplicity = 0 ∨ ∀ x ∈ xs, xs.count x ≤ maxMultiplicity) ∧
  (distinct ≠ 1 ∨ xs.Nodup) ∧
  (odd ≠ 1 ∨ ∀ x ∈ xs, x % 2 = 1)

def IsCompositionOf (total maxPart : Nat) (xs : List Nat) : Prop :=
  xs.sum = total ∧ (∀ x ∈ xs, 0 < x) ∧ (maxPart = 0 ∨ ∀ x ∈ xs, x ≤ maxPart)

/-- The partition family contains every admissible partition exactly once, in descending
lexicographic order. -/
theorem partition_family_spec (total maxPart maxParts maxMultiplicity distinct odd : Nat) :
    let rows := (Family.partitions total maxPart maxParts maxMultiplicity distinct odd).members
    rows.Pairwise (fun a b => lexLe (positiveParts b) (positiveParts a)) ∧ rows.Nodup ∧
    ∀ xs, xs ∈ rows.map positiveParts ↔
      IsPartitionOf total xs ∧ PartitionConstraints maxPart maxParts maxMultiplicity distinct odd xs := by
  sorry

/-- The composition family contains every admissible composition exactly once. When `parts = 0`,
length increases first; within a length, rows are descending lexicographic. -/
theorem composition_family_spec (total parts maxPart : Nat) :
    let rows := (Family.compositions total parts maxPart).members
    rows.Nodup ∧ ∀ xs, xs ∈ rows.map positiveParts ↔
      IsCompositionOf total maxPart xs ∧ (parts = 0 ∨ xs.length = parts) := by
  sorry

def dysonRank (xs : List Nat) : Int := (xs.headD 0 : Int) - (xs.length : Int)

def andrewsGarvanCrank (xs : List Nat) : Int :=
  let ones := (xs.filter (· = 1)).length
  if ones = 0 then xs.headD 0
  else ((xs.filter (· > ones)).length : Int) - (ones : Int)

theorem numberOfParts_spec (m : Mat) : numberOfParts m = (positiveParts m).length := by
  rfl

theorem largestPart_spec (m : Mat) : largestPart m = (positiveParts m).headD 0 := by
  rfl

theorem rank_spec {total : Nat} {m : Mat} (h : IsPartitionOf total (positiveParts m)) :
    (rankOffset total m : Int) - total = dysonRank (positiveParts m) := by
  sorry

theorem crank_spec {total : Nat} {m : Mat} (h : IsPartitionOf total (positiveParts m)) :
    (crankOffset total m : Int) - total = andrewsGarvanCrank (positiveParts m) := by
  sorry

end Integer_partitions.Contract
