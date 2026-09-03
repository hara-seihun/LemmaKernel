import Mathlib
import Lattice_of_subspaces.Reference
import Lk.Contract

/-!
# lattice_of_subspaces: contract

The executable formulas are related here to Mathlib submodules over `ZMod p`. Matrix members and
fixed subspaces denote row spaces. The statements are the obligations callers rely on; proofs that
remain to be connected to Mathlib are marked with `sorry`.
-/

namespace Lattice_of_subspaces.Contract

open Lk Lk.Contract

abbrev Ambient (p n : Nat) := Fin n → ZMod p

/-- Strict chains of submodules whose dimensions are listed by `dims`. -/
def StrictFlag (p n : Nat) (dims : List Nat) :=
  { flag : Fin dims.length → Submodule (ZMod p) (Ambient p n) //
      (∀ i, Module.finrank (ZMod p) (flag i) = dims.get i) ∧
      (∀ i j, i.val < j.val → flag i < flag j) }

variable {p rows cols h : Nat} [Fact p.Prime] {m : Mat}

/-- Gaussian binomials count subspaces of a finite vector space. -/
theorem gaussianBinomial_spec (n k : Nat) :
    gaussianBinomial p n k =
      Nat.card { W : Submodule (ZMod p) (Ambient p n) // Module.finrank (ZMod p) W = k } := by
  sorry

/-- `flagCount` counts strict flags with the requested dimensions. This also covers invalid
sequences: both sides are zero, except that the empty flag has count one. -/
theorem flagCount_spec (n : Nat) (dims : List Nat) :
    flagCount p n dims = Nat.card (StrictFlag p n dims) := by
  sorry

/-- The member denotes its row space, including when its rows are dependent. -/
theorem containedSubspaceCount_spec (hm : WellFormed p rows cols m) :
    containedSubspaceCount p m h =
      Nat.card { W : Submodule (ZMod p) (Ambient p cols) //
        Module.finrank (ZMod p) W = h ∧ W ≤ rowSpace p rows cols m } := by
  sorry

/-- Containing subspaces live in the member's ambient coordinate space. -/
theorem containingSubspaceCount_spec (hm : WellFormed p rows cols m) :
    containingSubspaceCount p m h =
      Nat.card { W : Submodule (ZMod p) (Ambient p cols) //
        Module.finrank (ZMod p) W = h ∧ rowSpace p rows cols m ≤ W } := by
  sorry

/-- `contains` is inclusion of the fixed row space in the member's row space. -/
theorem contains_spec {fixedRows : Nat} {u : Mat}
    (hm : WellFormed p rows cols m) (hu : WellFormed p fixedRows cols u) :
    contains p m u = true ↔ rowSpace p fixedRows cols u ≤ rowSpace p rows cols m := by
  sorry

/-- `isContainedIn` is inclusion of the member's row space in the fixed row space. -/
theorem isContainedIn_spec {fixedRows : Nat} {u : Mat}
    (hm : WellFormed p rows cols m) (hu : WellFormed p fixedRows cols u) :
    isContainedIn p m u = true ↔ rowSpace p rows cols m ≤ rowSpace p fixedRows cols u := by
  sorry

end Lattice_of_subspaces.Contract
