import Lk.Reference

/-!
# vertex_transitive executable reference

The input is a nonempty simple undirected graph given by its adjacency matrix over `F_2`.
Automorphisms are all vertex permutations preserving that matrix. An arc is an ordered adjacent
pair. The action on an empty arc set is transitive.

A regular subgroup has exactly `n` elements and is transitive on `n` vertices. The reference
examines every `n`-subset of the automorphism group, which is deliberately only for small oracle
cases. Each subgroup contains every element in lexicographic order, and the list of subgroups has
the order induced by combinations of that list.
-/

namespace Vertex_transitive

open Lk

abbrev Arc := Nat × Nat

/-- A square symmetric zero-diagonal adjacency matrix on one to ten vertices. -/
def validGraph (m : Mat) : Bool :=
  let n := m.length
  n > 0 && n ≤ 10 && m.all (·.length = n) &&
    (List.range n).all fun i =>
      (m.getD i []).getD i 0 = 0 &&
      (List.range n).all fun j => (m.getD i []).getD j 0 = (m.getD j []).getD i 0

/-- The runtime families whose members this module interprets as graphs. -/
def supportedFamily : Family → Bool
  | .explicit _ _ | .allGraphs _ | .edgeSubgraphs _ _ | .cayleyGraphs _ => true
  | _ => false

def acceptedFamily (f : Family) : Bool :=
  supportedFamily f && f.p = 2 && !f.members.isEmpty && f.members.all validGraph

/-- Every graph automorphism, sorted by its image list. -/
def automorphisms (m : Mat) : List Perm :=
  sortBy lexLe ((permutations (List.range m.length)).filter fun g => graphRelabel m g = m)

def pointOrbit (elements : List Perm) (i : Nat) : List Nat :=
  (sortBy (· ≤ ·) (elements.map fun g => g.getD i i)).eraseDups

def isVertexTransitive (m : Mat) : Bool :=
  (pointOrbit (automorphisms m) 0).length = m.length

def arcs (m : Mat) : List Arc :=
  (List.range m.length).flatMap fun i =>
    ((List.range m.length).filter fun j => (m.getD i []).getD j 0 = 1).map fun j => (i, j)

def arcLe (a b : Arc) : Bool :=
  a.1 < b.1 || (a.1 = b.1 && a.2 ≤ b.2)

def arcOrbit (elements : List Perm) (a : Arc) : List Arc :=
  (sortBy arcLe (elements.map fun g => (g.getD a.1 a.1, g.getD a.2 a.2))).eraseDups

def isArcTransitive (m : Mat) : Bool :=
  match arcs m with
  | [] => true
  | a :: as => (arcOrbit (automorphisms m) a).length = as.length + 1

def isSubgroup (n : Nat) (hs : List Perm) : Bool :=
  identityPerm n ∈ hs && hs.all fun g => hs.all fun h => compose g h ∈ hs

def isRegularSubgroup (n : Nat) (hs : List Perm) : Bool :=
  hs.length = n && isSubgroup n hs && (pointOrbit hs 0).length = n

/-- Every regular subgroup of `Aut(m)`. The complete element lists remove generator choices. -/
def regularSubgroups (m : Mat) : List (List Perm) :=
  let n := m.length
  (combos (automorphisms m) n).filter (isRegularSubgroup n)

def isCayley (m : Mat) : Bool := !(regularSubgroups m).isEmpty

inductive Op
  | isVertexTransitive
  | isArcTransitive
  | isCayley
  | regularSubgroups

inductive Value
  | regularSubgroups (groups : List (List Perm))
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  if !acceptedFamily f then .invalid else
  let ms := f.members
  match op with
  | .isVertexTransitive => reduceBool red ms (ms.map isVertexTransitive)
  | .isArcTransitive => reduceBool red ms (ms.map isArcTransitive)
  | .isCayley => reduceBool red ms (ms.map isCayley)
  | .regularSubgroups =>
    reduceValues red (ms.map fun m => .regularSubgroups (regularSubgroups m))

end Vertex_transitive
