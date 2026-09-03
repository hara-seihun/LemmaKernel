import Mathlib
import Vertex_transitive.Reference
import Lk.Contract

/-!
# vertex_transitive contract

The executable predicates use the full automorphism group of a labelled simple graph. This file
states their meaning with Mathlib permutations and simple graphs. The remaining proofs are marked
`sorry`; native answers are tested against the executable reference rather than treated as proved.
-/

namespace Vertex_transitive.Contract

open Lk

/-- The Mathlib simple graph represented by a well-formed binary adjacency matrix. -/
noncomputable def toGraph {n : Nat} (m : Mat) : SimpleGraph (Fin n) := by
  classical
  sorry

/-- The image list of a Mathlib permutation. -/
def permList {n : Nat} (σ : Equiv.Perm (Fin n)) : Perm :=
  List.ofFn fun i => (σ i : Nat)

/-- A permutation preserves adjacency. -/
def Preserves {n : Nat} (X : SimpleGraph (Fin n)) (σ : Equiv.Perm (Fin n)) : Prop :=
  ∀ i j, X.Adj (σ i) (σ j) ↔ X.Adj i j

/-- The full automorphism group, viewed as a subgroup of the symmetric group. -/
noncomputable def graphAut {n : Nat} (X : SimpleGraph (Fin n)) :
    Subgroup (Equiv.Perm (Fin n)) := by
  classical
  sorry

/-- A permutation subgroup acts regularly when exactly one element sends any vertex to any other. -/
def Regular {n : Nat} (H : Subgroup (Equiv.Perm (Fin n))) : Prop :=
  ∀ i j : Fin n, ∃! h : H, (h : Equiv.Perm (Fin n)) i = j

/-- `hs` is the complete image-list representation of `H`. -/
def Represents {n : Nat} (hs : List Perm) (H : Subgroup (Equiv.Perm (Fin n))) : Prop :=
  ∀ q, q ∈ hs ↔ ∃ h : H, q = permList (h : Equiv.Perm (Fin n))

/-- Matrix adjacency agrees with the Mathlib graph. -/
theorem toGraph_adj {n : Nat} {m : Mat} (h : Lk.Contract.WellFormed 2 n n m)
    (hsymm : ∀ i j, (m.getD i []).getD j 0 = (m.getD j []).getD i 0)
    (hdiag : ∀ i, i < n → (m.getD i []).getD i 0 = 0)
    (i j : Fin n) :
    (toGraph m).Adj i j ↔ (m.getD i []).getD j 0 = 1 := by
  sorry

/-- Vertex transitivity is transitivity of the full Mathlib automorphism group. -/
theorem isVertexTransitive_spec {n : Nat} {m : Mat}
    (hvalid : validGraph m = true) (hn : m.length = n) :
    isVertexTransitive m = true ↔
      ∀ i j : Fin n, ∃ σ ∈ graphAut (toGraph m), σ i = j := by
  sorry

/-- Arc transitivity means that every ordered adjacent pair can be carried to every other one.
The statement is vacuous when the graph has no arcs. -/
theorem isArcTransitive_spec {n : Nat} {m : Mat}
    (hvalid : validGraph m = true) (hn : m.length = n) :
    isArcTransitive m = true ↔
      ∀ i j i' j' : Fin n, (toGraph m).Adj i j → (toGraph m).Adj i' j' →
        ∃ σ ∈ graphAut (toGraph m), σ i = i' ∧ σ j = j' := by
  sorry

/-- The returned value lists every regular subgroup exactly once. Complete sorted element lists
make the representation canonical. -/
theorem regularSubgroups_spec {n : Nat} {m : Mat}
    (hvalid : validGraph m = true) (hn : m.length = n) :
    let groups := regularSubgroups m
    groups.Nodup ∧
    groups.Pairwise (fun a b => lexLe a.flatten b.flatten = true) ∧
    ∀ hs, hs ∈ groups ↔
      ∃ H : Subgroup (Equiv.Perm (Fin n)), H ≤ graphAut (toGraph m) ∧ Regular H ∧ Represents hs H := by
  sorry

/-- Sabidussi's criterion: a finite graph is Cayley exactly when its automorphism group contains
a regular subgroup. -/
theorem isCayley_spec {n : Nat} {m : Mat}
    (hvalid : validGraph m = true) (hn : m.length = n) :
    isCayley m = true ↔
      ∃ H : Subgroup (Equiv.Perm (Fin n)), H ≤ graphAut (toGraph m) ∧ Regular H := by
  sorry

end Vertex_transitive.Contract
