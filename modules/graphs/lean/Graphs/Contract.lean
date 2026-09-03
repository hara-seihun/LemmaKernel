import Mathlib
import Graphs.Reference
import Lk.Contract

/-!
# graphs: contract

The executable adjacency-list searches are related here to Mathlib's `SimpleGraph` invariants.
The native backend is tested against `Reference.lean`; the statements carrying `sorry` are the
remaining proof obligations between that reference and Mathlib.
-/

namespace Graphs.Contract

open Lk

/-- A `SimpleGraph` on the matrix indices. Requiring both directed entries makes this definition
symmetric even before the well-formedness hypothesis is supplied. -/
def toSimpleGraph (g : Mat) : SimpleGraph (Fin g.length) where
  Adj u v := u ≠ v ∧ adjacent g u.val v.val = true ∧ adjacent g v.val u.val = true
  symm := ⟨by
    intro u v h
    exact ⟨h.1.symm, h.2.2, h.2.1⟩⟩
  loopless := ⟨by
    intro v h
    exact h.1 rfl⟩

variable {g : Mat}

theorem connected_spec (h : wellFormed g = true) :
    connected g = true ↔ (toSimpleGraph g).Connected := by
  sorry

theorem girth_spec (h : wellFormed g = true) :
    girth g = (toSimpleGraph g).girth := by
  sorry

/-- Mathlib assigns diameter zero to disconnected graphs. The module uses `n` as a distinct
sentinel, since every connected `n`-vertex graph has diameter below `n`. -/
theorem diameter_spec (h : wellFormed g = true) :
    ((toSimpleGraph g).Connected → diameter g = (toSimpleGraph g).diam) ∧
    (¬(toSimpleGraph g).Connected → diameter g = g.length) := by
  sorry

theorem chromaticNumber_spec (h : wellFormed g = true) :
    chromaticNumber g = ENat.toNat (toSimpleGraph g).chromaticNumber := by
  sorry

theorem cliqueNumber_spec (h : wellFormed g = true) :
    cliqueNumber g = (toSimpleGraph g).cliqueNum := by
  sorry

theorem independenceNumber_spec (h : wellFormed g = true) :
    independenceNumber g = (toSimpleGraph g).indepNum := by
  sorry

theorem isBipartite_spec (h : wellFormed g = true) :
    isBipartite g = true ↔ (toSimpleGraph g).IsBipartite := by
  sorry

noncomputable def mathDegree (g : Mat) (v : Fin g.length) : ℕ :=
  Set.ncard ((toSimpleGraph g).neighborSet v)

theorem degreeSequence_spec (h : wellFormed g = true) :
    degreeSequence g = sortBy (· ≥ ·) (List.ofFn fun v : Fin g.length => mathDegree g v) := by
  sorry

/-- Canonical form preserves the graph and is no larger than any relabelling, byte for byte in
row-major order. -/
theorem canonicalForm_spec (h : wellFormed g = true) :
    Nonempty (toSimpleGraph g ≃g toSimpleGraph (graphCanonical g)) ∧
    ∀ p ∈ permutations (List.range g.length),
      lexLe (graphCanonical g).flatten (graphRelabel g p).flatten = true := by
  sorry

/-- `all_graphs n` has one canonical representative per isomorphism class. -/
theorem allGraphs_spec (n : ℕ) (hn : 0 < n) :
    (allGraphMembers n).Pairwise (fun a b => ¬Nonempty (toSimpleGraph a ≃g toSimpleGraph b)) ∧
    ∀ g, wellFormed g = true → g.length = n →
      ∃ c ∈ allGraphMembers n, Nonempty (toSimpleGraph g ≃g toSimpleGraph c) := by
  sorry

/-- Every member is a spanning subgraph with exactly `k` host edges. -/
theorem edgeSubgraphs_spec (host : Mat) (k : ℕ) (g : Mat)
    (hg : g ∈ (Family.edgeSubgraphs host k).members) :
    graphEdges g ⊆ graphEdges host ∧ (graphEdges g).length = k := by
  sorry

/-- Inverse-closed connection sets make every generated Cayley adjacency matrix simple and
undirected. -/
theorem cayleyGraphs_spec (gens : List Perm) (g : Mat)
    (hg : g ∈ (Family.cayleyGraphs gens).members) :
    wellFormed g = true := by
  sorry

end Graphs.Contract
