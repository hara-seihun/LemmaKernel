import Mathlib
import Automorphisms.Reference
import Lk.Contract

/-!
# automorphisms: contract

The executable reference works on labels in a Cayley table. This file relates those labels to a
Mathlib finite group. Proofs are left as stated obligations, as with the other modules.
-/

namespace Automorphisms.Contract

open Lk

/-- The Cayley table obtained by listing every element of a finite group in `xs`. -/
def cayleyTable {G : Type*} [Group G] [DecidableEq G] (xs : List G) : Mat :=
  xs.map fun a => xs.map fun b => xs.idxOf (a * b)

variable {G : Type*} [Group G] [Fintype G] [DecidableEq G]

/-- Table automorphisms are exactly Mathlib multiplicative automorphisms. -/
theorem autOrder_spec (xs : List G) (hnodup : xs.Nodup) (hcomplete : ∀ g : G, g ∈ xs) :
    (automorphisms (cayleyTable xs)).length = Nat.card (MulAut G) := by
  sorry

/-- The canonical list really generates every table automorphism. Its choice is fixed by the
lexicographic scan in `Reference.lean`, rather than only up to another generating set. -/
theorem canonicalGenerators_spec (table : Mat) :
    generated table.length (canonicalGenerators table) = automorphisms table := by
  sorry

/-- Each canonical table generator transports to a Mathlib automorphism, and those automorphisms
generate `MulAut G`. -/
theorem canonicalGenerators_mulAut (xs : List G) (hnodup : xs.Nodup) (hcomplete : ∀ g : G, g ∈ xs) :
    ∃ autos : List (MulAut G),
      autos.length = (canonicalGenerators (cayleyTable xs)).length ∧
      Subgroup.closure (↑autos.toFinset : Set (MulAut G)) = ⊤ ∧
      ∀ i, i < autos.length → ∀ j, j < xs.length →
        xs.getD (((canonicalGenerators (cayleyTable xs)).getD i []).getD j 0) 1 =
          (autos.getD i (MulEquiv.refl G)) (xs.getD j 1) := by
  sorry

/-- A holomorph has underlying cardinality `|G| * |Aut(G)|`. -/
theorem holomorphOrder_spec (xs : List G) (hnodup : xs.Nodup) (hcomplete : ∀ g : G, g ∈ xs) :
    xs.length * (automorphisms (cayleyTable xs)).length = Nat.card G * Nat.card (MulAut G) := by
  sorry

/-- `inner_aut_index` is the index of the range of Mathlib's conjugation homomorphism in the full
automorphism group. -/
theorem innerAutIndex_spec (xs : List G) (hnodup : xs.Nodup) (hcomplete : ∀ g : G, g ∈ xs) :
    (automorphisms (cayleyTable xs)).length * centerSize (cayleyTable xs) / xs.length =
      (MulAut.conj : G →* MulAut G).range.index := by
  sorry

end Automorphisms.Contract
