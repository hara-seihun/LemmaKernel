import Mathlib
import Set_systems.Reference

/-!
# set_systems: contract

The executable reference uses binary incidence words. This file states the same operations with
Mathlib `Finset`s. The native backend is tested against the reference; the statements marked
`sorry` are the remaining proof obligations connecting that reference to these definitions.
-/

namespace Set_systems.Contract

open Lk

/-- The finite subset of `Fin n` encoded by a binary incidence word. -/
def asSet (n : Nat) (s : Vec) : Finset (Fin n) :=
  Finset.univ.filter fun i => s.getD i.val 0 = 1

/-- The finite family encoded by a matrix of distinct incidence rows. -/
def asFamily (n : Nat) (system : Mat) : Finset (Finset (Fin n)) :=
  (system.map (asSet n)).toFinset

def WellFormedSystem (n : Nat) (system : Mat) : Prop :=
  system.Nodup ∧ ∀ s ∈ system, s.length = n ∧ ∀ x ∈ s, x = 0 ∨ x = 1

def PairwiseIntersecting {n : Nat} (family : Finset (Finset (Fin n))) : Prop :=
  ∀ a ∈ family, ∀ b ∈ family, a ≠ b → (a ∩ b).Nonempty

def Antichain {n : Nat} (family : Finset (Finset (Fin n))) : Prop :=
  ∀ a ∈ family, ∀ b ∈ family, a ≠ b → ¬a ⊆ b

/-- A `k`-sunflower has `k` distinct petals and one common value for every pairwise intersection. -/
def ContainsSunflower {n : Nat} (k : Nat) (family : Finset (Finset (Fin n))) : Prop :=
  ∃ petals : Finset (Finset (Fin n)), petals ⊆ family ∧ petals.card = k ∧
    ∃ core : Finset (Fin n), ∀ a ∈ petals, ∀ b ∈ petals, a ≠ b → a ∩ b = core

def degree {n : Nat} (family : Finset (Finset (Fin n))) (x : Fin n) : Nat :=
  (family.filter fun a => x ∈ a).card

def semanticMaxDegree {n : Nat} (family : Finset (Finset (Fin n))) : Nat :=
  (Finset.univ : Finset (Fin n)).sup (degree family)

/-- The immediate lower shadow, with duplicate deletions identified. -/
def semanticLowerShadow {n : Nat} (family : Finset (Finset (Fin n))) : Finset (Finset (Fin n)) :=
  family.biUnion fun a => a.image fun x => a.erase x

def Uniform {n : Nat} (r : Nat) (family : Finset (Finset (Fin n))) : Prop :=
  ∀ a ∈ family, a.card = r

def EkrExtremal {n : Nat} (family : Finset (Finset (Fin n))) : Prop :=
  ∃ r, 0 < r ∧ 2 * r ≤ n ∧ Uniform r family ∧ PairwiseIntersecting family ∧
    family.card = Nat.choose (n - 1) (r - 1)

def SpernerExtremal {n : Nat} (family : Finset (Finset (Fin n))) : Prop :=
  Antichain family ∧ family.card = Nat.choose n (n / 2)

variable {n : Nat} {system : Mat}

theorem isIntersecting_spec (h : WellFormedSystem n system) :
    isIntersecting system = true ↔ PairwiseIntersecting (asFamily n system) := by
  sorry

theorem isAntichain_spec (h : WellFormedSystem n system) :
    isAntichain system = true ↔ Antichain (asFamily n system) := by
  sorry

theorem isSunflowerFree_spec (h : WellFormedSystem n system) (k : Nat) (hk : 2 ≤ k) :
    isSunflowerFree k system = true ↔ ¬ContainsSunflower k (asFamily n system) := by
  sorry

theorem maxDegree_spec (h : WellFormedSystem n system) :
    maxDegree n system = semanticMaxDegree (asFamily n system) := by
  sorry

theorem shadowSize_spec (h : WellFormedSystem n system) :
    shadowSize system = (semanticLowerShadow (asFamily n system)).card := by
  sorry

theorem isEkrExtremal_spec (h : WellFormedSystem n system) :
    isEkrExtremal n system = true ↔ EkrExtremal (asFamily n system) := by
  sorry

theorem isSpernerExtremal_spec (h : WellFormedSystem n system) :
    isSpernerExtremal n system = true ↔ SpernerExtremal (asFamily n system) := by
  sorry

end Set_systems.Contract
