import Mathlib
import Graph_polynomials.Reference
import Lk.Contract

/-!
# graph_polynomials: contract

The executable reference works on edge lists. This file interprets a valid list as a Mathlib
`SimpleGraph (Fin vertices)` and states the four standard polynomial characterisations. The
proofs are open obligations. Native backends are checked against the executable reference, not
claimed to be verified by these statements.
-/

namespace Graph_polynomials.Contract

open Lk Graph_polynomials


def graph (vertices : Nat) (edges : List Edge) : SimpleGraph (Fin vertices) :=
  SimpleGraph.fromRel fun u v => (u.val, v.val) ∈ edges


noncomputable def adjacencyMatrix (vertices : Nat) (edges : List Edge) : Matrix (Fin vertices) (Fin vertices) Int := by
  classical
  exact fun u v => if (graph vertices edges).Adj u v then 1 else 0


def ProperColoring (vertices colors : Nat) (edges : List Edge) :=
  { coloring : Fin vertices → Fin colors //
      ∀ u v, (graph vertices edges).Adj u v → coloring u ≠ coloring v }


def MatchingOfSize (vertices : Nat) (edges : List Edge) (size : Nat) :=
  { matching : Finset (Fin vertices × Fin vertices) //
      matching.card = size ∧
      (∀ e ∈ matching, e.1.val < e.2.val ∧ (graph vertices edges).Adj e.1 e.2) ∧
      (∀ e ∈ matching, ∀ f ∈ matching, e ≠ f →
        e.1 ≠ f.1 ∧ e.1 ≠ f.2 ∧ e.2 ≠ f.1 ∧ e.2 ≠ f.2) }


noncomputable def componentCount (vertices : Nat) (edges : List Edge) : Nat :=
  Nat.card (graph vertices edges).ConnectedComponent


def edgeSubsets (edges : List Edge) : List (List Edge) :=
  (List.range (edges.length + 1)).flatMap (combos edges)


noncomputable def tutteEvaluation (vertices : Nat) (edges : List Edge) (x y : Int) : Int :=
  let wholeComponents := componentCount vertices edges
  (edgeSubsets edges).foldl (fun total chosen =>
    let chosenComponents := componentCount vertices chosen
    total + (x - 1) ^ (chosenComponents - wholeComponents) *
      (y - 1) ^ (chosen.length + chosenComponents - vertices)) 0


def evaluate (coefficients : List Int) (x : Int) : Int :=
  (List.range coefficients.length).foldl
    (fun total i => total + coefficients.getD i 0 * x ^ i) 0


def evaluateTutte (vertices edgeCount : Nat) (coefficients : List Int) (x y : Int) : Int :=
  (List.range (vertices + 1)).foldl (fun total i =>
    total + (List.range (edgeCount + 1)).foldl (fun row j =>
      row + coefficients.getD (i * (edgeCount + 1) + j) 0 * x ^ i * y ^ j) 0) 0


variable {vertices : Nat} {edges : List Edge}


theorem chromatic_spec (h : validDictionary vertices (edges.map fun e => [e.1, e.2]) = true) :
    ∀ colors : Nat,
      evaluate (chromatic vertices edges) colors =
        (Nat.card (ProperColoring vertices colors edges) : Int) := by
  sorry


theorem tutte_spec (h : validDictionary vertices (edges.map fun e => [e.1, e.2]) = true) :
    ∀ x y : Int,
      evaluateTutte vertices edges.length (tutte vertices edges.length edges) x y =
        tutteEvaluation vertices edges x y := by
  sorry


theorem characteristic_spec (h : validDictionary vertices (edges.map fun e => [e.1, e.2]) = true) :
    ∀ i : Nat,
      (characteristic vertices edges).getD i 0 =
        (adjacencyMatrix vertices edges).charpoly.coeff i := by
  sorry


theorem matching_spec (h : validDictionary vertices (edges.map fun e => [e.1, e.2]) = true) :
    ∀ x : Int,
      evaluate (matching vertices edges) x =
        (List.range (vertices / 2 + 1)).foldl (fun total k =>
          total + (-1 : Int) ^ k * (Nat.card (MatchingOfSize vertices edges k) : Int) *
            x ^ (vertices - 2 * k)) 0 := by
  sorry

end Graph_polynomials.Contract
