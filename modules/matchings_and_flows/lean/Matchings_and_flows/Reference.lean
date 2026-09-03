import Lk.Reference

/-!
# matchings_and_flows: executable reference

A matrix entry is a nonnegative integer. `perfectMatchingCount` is the permanent. A symmetric
matrix defines an undirected multigraph with `A[i][j]` parallel edges between distinct vertices;
`spanningTreeCount` sums the resulting multiplicities over spanning trees. A general square
matrix is a directed capacity network for `maxFlow`.

The permanent expands one row at a time, spanning trees are enumerated directly, and maximum
flow is evaluated as the minimum directed cut. These definitions use structural recursion and
small lists because this file is the test oracle. The native backend uses Ryser's formula, the
matrix-tree theorem, and an augmenting-path algorithm.
-/

namespace Matchings_and_flows

open Lk


def square (m : Mat) : Bool := m.all fun row => row.length = m.length

def symmetric (m : Mat) : Bool :=
  square m && (List.range m.length).all fun i =>
    (List.range m.length).all fun j => (m.getD i []).getD j 0 = (m.getD j []).getD i 0


def eraseAt : Nat → List α → List α
  | _, [] => []
  | 0, _ :: xs => xs
  | i + 1, x :: xs => x :: eraseAt i xs

/-- Laplace expansion of the permanent, with fuel equal to the number of rows. -/
def permanentAux : Nat → Mat → Nat
  | 0, m => if m.isEmpty then 1 else 0
  | _ + 1, [] => 1
  | fuel + 1, row :: rows =>
      (List.range row.length).foldl (fun total j =>
        total + row.getD j 0 * permanentAux fuel (rows.map (eraseAt j))) 0

def perfectMatchingCount (m : Mat) : Nat := permanentAux m.length m


structure Edge where
  u : Nat
  v : Nat
  weight : Nat
  deriving DecidableEq, Repr

/-- One weighted edge for each nonzero upper-triangular entry. -/
def weightedEdges (m : Mat) : List Edge :=
  (List.range m.length).flatMap fun i =>
    ((List.range m.length).filter fun j => i < j ∧ (m.getD i []).getD j 0 ≠ 0).map fun j =>
      { u := i, v := j, weight := (m.getD i []).getD j 0 }

def edgeNeighbours (edges : List Edge) (x : Nat) : List Nat :=
  edges.filterMap fun e =>
    if e.u = x then some e.v else if e.v = x then some e.u else none

def connectedEdges (n : Nat) (edges : List Edge) : Bool :=
  if n = 0 then false
  else (saturate (edgeNeighbours edges) (n + 1) [0]).length = n

def treeWeight (n : Nat) (edges : List Edge) : Nat :=
  if connectedEdges n edges then edges.foldl (fun a e => a * e.weight) 1 else 0

/-- Direct weighted spanning-tree enumeration. Loops do not appear in `weightedEdges`. -/
def spanningTreeCount (m : Mat) : Nat :=
  let n := m.length
  if n = 0 then 0
  else (combos (weightedEdges m) (n - 1)).foldl (fun total tree => total + treeWeight n tree) 0


/-- Every sublist, used to enumerate source sides of directed cuts. -/
def powerset : List α → List (List α)
  | [] => [[]]
  | x :: xs =>
      let rest := powerset xs
      rest ++ rest.map (x :: ·)

def cutCapacity (m : Mat) (side : List Nat) : Nat :=
  (List.range m.length).foldl (fun total i =>
    if i ∈ side then
      total + (List.range m.length).foldl (fun rowTotal j =>
        if j ∈ side then rowTotal else rowTotal + (m.getD i []).getD j 0) 0
    else total) 0

def minimum : List Nat → Nat
  | [] => 0
  | x :: xs => xs.foldl min x

/-- Maximum flow via the minimum directed cut. `run` checks the terminal indices first. -/
def maxFlow (m : Mat) (source sink : Nat) : Nat :=
  minimum (((powerset (List.range m.length)).filter fun side => source ∈ side ∧ sink ∉ side).map
    (cutCapacity m))


inductive Op
  | perfectMatchingCount
  | spanningTreeCount
  | maxFlow (sink source : Nat)

inductive Value deriving DecidableEq, Repr


def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let ms := f.members
  let n := (ms.headD []).length
  if !(ms.all square) then .invalid
  else match op with
    | .perfectMatchingCount => reduceInt red ms (ms.map perfectMatchingCount)
    | .spanningTreeCount =>
        if !(ms.all symmetric) then .invalid
        else reduceInt red ms (ms.map spanningTreeCount)
    | .maxFlow sink source =>
        if source = sink || source ≥ n || sink ≥ n then .invalid
        else reduceInt red ms (ms.map fun m => maxFlow m source sink)

end Matchings_and_flows
