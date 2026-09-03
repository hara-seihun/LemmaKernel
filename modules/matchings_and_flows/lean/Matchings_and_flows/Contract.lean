import Mathlib
import Matchings_and_flows.Reference
import Lk.Contract

/-!
# matchings_and_flows: contract

The executable reference uses lists so Lean's kernel can evaluate test cases. This file states its
meaning with Mathlib matrices, permanents, determinants, finite sums, and feasible integral flows.
The matrix-tree and max-flow/min-cut connections are explicit proof obligations. They are left as
`sorry`; native answers are checked against the executable reference.
-/

namespace Matchings_and_flows.Contract

open Matrix Lk


def WellFormed (n : Nat) (m : Mat) : Prop :=
  m.length = n ∧ ∀ row ∈ m, row.length = n

def toNatMatrix (n : Nat) (m : Mat) : Matrix (Fin n) (Fin n) Nat :=
  fun i j => (m.getD i []).getD j 0


def laplacian (n : Nat) (m : Mat) : Matrix (Fin n) (Fin n) Int :=
  fun i j =>
    if i = j then ∑ k ∈ Finset.univ.erase i, (toNatMatrix n m i k : Int)
    else -(toNatMatrix n m i j : Int)

/-- The principal Laplacian minor obtained by deleting the last vertex. -/
def laplacianMinor (n : Nat) (m : Mat) : Matrix (Fin (n - 1)) (Fin (n - 1)) Int :=
  fun i j => laplacian n m (Fin.castLE (Nat.sub_le n 1) i) (Fin.castLE (Nat.sub_le n 1) j)


def IsFlow {n : Nat} (capacity flow : Matrix (Fin n) (Fin n) Nat) (source sink : Fin n) : Prop :=
  (∀ i j, flow i j ≤ capacity i j) ∧
  (∀ v, v ≠ source → v ≠ sink → ∑ u, flow u v = ∑ w, flow v w)

def flowValue {n : Nat} (flow : Matrix (Fin n) (Fin n) Nat) (source : Fin n) : Nat :=
  (∑ v, flow source v) - ∑ v, flow v source


/-- `perfect_matching_count` is Mathlib's permanent over the natural numbers. -/
theorem perfectMatchingCount_spec {n : Nat} {m : Mat} (h : WellFormed n m) :
    perfectMatchingCount m = (toNatMatrix n m).permanent := by
  sorry

/-- The weighted matrix-tree theorem. Symmetry makes each off-diagonal entry the multiplicity of
one undirected edge; diagonal entries represent loops and do not enter either side. -/
theorem spanningTreeCount_spec {n : Nat} {m : Mat} (hn : 0 < n) (h : WellFormed n m)
    (hsymm : (toNatMatrix n m).IsSymm) :
    (spanningTreeCount m : Int) = (laplacianMinor n m).det := by
  sorry

/-- Integral max-flow/min-cut: the reference value bounds every feasible integral flow and some
feasible integral flow attains it. -/
theorem maxFlow_spec {n : Nat} {m : Mat} (h : WellFormed n m) (source sink : Fin n)
    (hne : source ≠ sink) :
    (∀ f, IsFlow (toNatMatrix n m) f source sink → flowValue f source ≤ maxFlow m source sink) ∧
    (∃ f, IsFlow (toNatMatrix n m) f source sink ∧ flowValue f source = maxFlow m source sink) := by
  sorry

end Matchings_and_flows.Contract
