import Mathlib.Combinatorics.Hypergraph.Basic
import Hypergraphs.Reference
import Lk.Contract

/-!
# hypergraphs: contract

The executable reference represents a finite uniform hypergraph by sorted lists. This file turns
that representation into Mathlib's `Hypergraph (Fin n)` and states what each operation means.
The statements are the contract; the native backend is tested against the executable reference.
-/

namespace Hypergraphs.Contract

open Set

/-- The set represented by one canonical edge row. -/
def edgeSet (n : Nat) (edge : List Nat) : Set (Fin n) :=
  {v | v.val ∈ edge}

/-- The Mathlib hypergraph represented by a list of edge rows on `Fin n`. -/
def model (n : Nat) (edges : Hypergraphs.Hypergraph) : _root_.Hypergraph (Fin n) where
  vertexSet := Set.univ
  edgeSet := {e | ∃ edge ∈ edges, e = edgeSet n edge}
  subset_vertexSet_of_mem_edgeSet' := by simp

/-- Every edge has cardinality `k`. -/
def IsUniform (H : _root_.Hypergraph (Fin n)) (k : Nat) : Prop :=
  ∀ e ∈ H.edgeSet, e.ncard = k

/-- Distinct edges meet in at most one vertex. -/
def IsLinear (H : _root_.Hypergraph (Fin n)) : Prop :=
  ∀ e ∈ H.edgeSet, ∀ f ∈ H.edgeSet, e ≠ f → (e ∩ f).ncard ≤ 1

/-- A weak vertex colouring has no monochromatic edge. -/
def IsWeakColouring (H : _root_.Hypergraph (Fin n)) {C : Type} (colour : Fin n → C) : Prop :=
  ∀ e ∈ H.edgeSet, ∃ u ∈ e, ∃ v ∈ e, colour u ≠ colour v

/-- `q` is the least size of a weak vertex-colour set. -/
def IsWeakChromaticNumber (H : _root_.Hypergraph (Fin n)) (q : Nat) : Prop :=
  (∃ colour : Fin n → Fin q, IsWeakColouring H colour) ∧
    ∀ r < q, ¬∃ colour : Fin n → Fin r, IsWeakColouring H colour

/-- A Berge cycle alternates distinct vertices and distinct edges. Consecutive vertices, including
last and first, lie in the corresponding edge. -/
def HasBergeCycle (H : _root_.Hypergraph (Fin n)) (length : Nat) : Prop :=
  ∃ vertices : List (Fin n), ∃ edges : List (Set (Fin n)),
    vertices.length = length ∧ edges.length = length ∧ vertices.Nodup ∧ edges.Nodup ∧
      List.Forall₂ (fun pair edge => edge ∈ H.edgeSet ∧ pair.1 ∈ edge ∧ pair.2 ∈ edge)
        (vertices.zip (vertices.drop 1 ++ vertices.take 1)) edges

/-- A red clique contains every `k`-edge on one vertex set. -/
def HasClique (H : _root_.Hypergraph (Fin n)) (k size : Nat) : Prop :=
  ∃ vertices : Set (Fin n), vertices.ncard = size ∧
    ∀ edge : Set (Fin n), edge ⊆ vertices → edge.ncard = k → edge ∈ H.edgeSet

/-- A blue clique contains no red `k`-edge on one vertex set. -/
def HasComplementClique (H : _root_.Hypergraph (Fin n)) (k size : Nat) : Prop :=
  ∃ vertices : Set (Fin n), vertices.ncard = size ∧
    ∀ edge : Set (Fin n), edge ⊆ vertices → edge.ncard = k → edge ∉ H.edgeSet

/-- Canonical rows represent sets with the same cardinality as their list representation. -/
theorem model_uniform (n k edgeCount : Nat) (edges : Hypergraphs.Hypergraph)
    (hvalid : validHypergraph n k edgeCount edges = true) : IsUniform (model n edges) k := by
  sorry

/-- `is_linear` is Mathlib set intersection linearity. -/
theorem isLinear_spec (n k edgeCount : Nat) (edges : Hypergraphs.Hypergraph)
    (hvalid : validHypergraph n k edgeCount edges = true) :
    isLinear edges = true ↔ IsLinear (model n edges) := by
  sorry

/-- `colouring_number` is the least number of colours in a weak colouring. -/
theorem colouringNumber_spec (n k edgeCount : Nat) (edges : Hypergraphs.Hypergraph)
    (hvalid : validHypergraph n k edgeCount edges = true) :
    IsWeakChromaticNumber (model n edges) (colouringNumber n edges) := by
  sorry

/-- The alternating-list search is exactly the standard Berge-cycle predicate. -/
theorem hasBergeCycle_spec (n k edgeCount length : Nat) (edges : Hypergraphs.Hypergraph)
    (hvalid : validHypergraph n k edgeCount edges = true) (hlength : 2 ≤ length) :
    hasBergeCycle n length edges = true ↔ HasBergeCycle (model n edges) length := by
  sorry

/-- `berge_girth` is zero for an acyclic hypergraph and otherwise the least Berge-cycle length. -/
theorem bergeGirth_spec (n k edgeCount : Nat) (edges : Hypergraphs.Hypergraph)
    (hvalid : validHypergraph n k edgeCount edges = true) :
    let girth := bergeGirth n edges
    (girth = 0 ∧ ∀ length, 2 ≤ length → ¬HasBergeCycle (model n edges) length) ∨
      (2 ≤ girth ∧ HasBergeCycle (model n edges) girth ∧
        ∀ length, 2 ≤ length → length < girth → ¬HasBergeCycle (model n edges) length) := by
  sorry

/-- `is_clique_free` is the absence of a complete uniform subhypergraph of the requested size. -/
theorem isCliqueFree_spec (n k edgeCount size : Nat) (edges : Hypergraphs.Hypergraph)
    (hvalid : validHypergraph n k edgeCount edges = true) (hsize : k ≤ size) :
    isCliqueFree n size edges = true ↔ ¬HasClique (model n edges) k size := by
  sorry

/-- The Ramsey predicate uses the listed edges as red and their uniform complement as blue. -/
theorem isRamseyColouring_spec (n k edgeCount redSize blueSize : Nat)
    (edges : Hypergraphs.Hypergraph) (hvalid : validHypergraph n k edgeCount edges = true)
    (hred : k ≤ redSize) (hblue : k ≤ blueSize) :
    isRamseyColouring n redSize blueSize edges = true ↔
      ¬HasClique (model n edges) k redSize ∧
      ¬HasComplementClique (model n edges) k blueSize := by
  sorry

end Hypergraphs.Contract
