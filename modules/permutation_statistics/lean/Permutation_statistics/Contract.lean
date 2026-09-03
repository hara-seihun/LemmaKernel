import Mathlib
import Permutation_statistics.Reference
import Lk.Contract

/-!
# permutation_statistics: contract

These statements relate the executable list definitions to permutations of `Fin n` and finite
sets in Mathlib. The native backend is tested against `Reference.lean`; the `sorry` proofs below
record the remaining bridge from that oracle to Mathlib.
-/

namespace Permutation_statistics.Contract

open Lk Permutation_statistics

/-- Turn a validated zero-based image list into Mathlib's permutation of `Fin n`. -/
noncomputable def toPerm (g : Perm) (h : validPerm g = true) : Equiv.Perm (Fin g.length) := by
  sorry


def inversionPairs {n : Nat} (w : Equiv.Perm (Fin n)) : Finset (Fin n × Fin n) :=
  Finset.univ.filter fun ij => ij.1 < ij.2 ∧ w ij.1 > w ij.2


def descentPairs {n : Nat} (w : Equiv.Perm (Fin n)) : Finset (Fin n × Fin n) :=
  Finset.univ.filter fun ij => ij.1.val + 1 = ij.2.val ∧ w ij.1 > w ij.2


def mathlibMajorIndex {n : Nat} (w : Equiv.Perm (Fin n)) : Nat :=
  ∑ ij ∈ descentPairs w, ij.1.val + 1


theorem inversions_spec (g : Perm) (h : validPerm g = true) :
    inversions g = (inversionPairs (toPerm g h)).card := by
  sorry


theorem descents_spec (g : Perm) (h : validPerm g = true) :
    descents g = (descentPairs (toPerm g h)).card := by
  sorry


theorem majorIndex_spec (g : Perm) (h : validPerm g = true) :
    majorIndex g = mathlibMajorIndex (toPerm g h) := by
  sorry

/-- The executable cycle lengths are Mathlib's complete cycle partition, including fixed points. -/
theorem cycleLengths_spec (g : Perm) (h : validPerm g = true) :
    (cycleLengths g : Multiset Nat) = (toPerm g h).partition.parts := by
  sorry

/-- `cycleTypeCode` selects the complete cycle partition from the manifest's canonical list. -/
theorem cycleTypeCode_spec (g : Perm) (h : validPerm g = true) :
    let part := (partitions g.length).getD (cycleTypeCode g) []
    part = cycleLengths g ∧ (part : Multiset Nat) = (toPerm g h).partition.parts := by
  sorry

/-- Classical pattern containment through an order-preserving choice of positions. -/
def ContainsPattern {n k : Nat} (w : Equiv.Perm (Fin n)) (pattern : Equiv.Perm (Fin k)) : Prop :=
  ∃ positions : Fin k ↪o Fin n, ∀ i j, pattern i < pattern j ↔ w (positions i) < w (positions j)


theorem containsPattern_spec (g pattern : Perm)
    (hg : validPerm g = true) (hp : validPerm pattern = true) :
    containsPattern g pattern = true ↔ ContainsPattern (toPerm g hg) (toPerm pattern hp) := by
  sorry


theorem patternAvoids_spec (g : Perm) (patterns : List Perm)
    (hg : validPerm g = true) (hp : ∀ pattern ∈ patterns, validPerm pattern = true) :
    patternAvoids g patterns = true ↔
      ∀ pattern, ∀ hmem : pattern ∈ patterns,
        ¬ContainsPattern (toPerm g hg) (toPerm pattern (hp pattern hmem)) := by
  sorry


def upperRankMathlib {n : Nat} (w : Equiv.Perm (Fin n)) (pref threshold : Nat) : Nat :=
  (Finset.univ.filter fun i : Fin n => i.val < pref ∧ threshold ≤ (w i).val).card

/-- Strong Bruhat order in its rank-matrix characterization. -/
def BruhatLE {n : Nat} (lower upper : Equiv.Perm (Fin n)) : Prop :=
  ∀ pref threshold, pref ≤ n → threshold ≤ n →
    upperRankMathlib lower pref threshold ≤ upperRankMathlib upper pref threshold

noncomputable def castUpper (lower upper : Perm) (hu : validPerm upper = true)
    (hn : lower.length = upper.length) : Equiv.Perm (Fin lower.length) := by
  sorry


theorem bruhatLeq_spec (lower upper : Perm)
    (hl : validPerm lower = true) (hu : validPerm upper = true)
    (hn : lower.length = upper.length) :
    bruhatLeq lower upper = true ↔
      BruhatLE (toPerm lower hl) (castUpper lower upper hu hn) := by
  sorry

end Permutation_statistics.Contract
