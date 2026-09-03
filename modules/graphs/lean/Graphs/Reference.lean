import Lk.Reference

/-!
# graphs: executable reference

Graphs are nonempty square symmetric zero-diagonal adjacency matrices over `F_2`. Distances use
breadth-first search. Exponential invariants and canonical labelling use direct finite searches so
Lean's kernel can evaluate the small oracle cases without well-founded recursion.
-/

namespace Graphs

open Lk

def adjacent (g : Mat) (u v : Nat) : Bool := (g.getD u []).getD v 0 = 1

def wellFormed (g : Mat) : Bool :=
  let n := g.length
  n > 0 ∧ g.all (·.length = n) ∧
    (List.range n).all fun i =>
      !adjacent g i i ∧ (List.range n).all fun j => adjacent g i j = adjacent g j i

def neighbors (g : Mat) (v : Nat) : List Nat :=
  (List.range g.length).filter (adjacent g v)

/-- Queue entries carry a vertex and its distance from the source. -/
def bfs (g : Mat) : Nat → List (Nat × Nat) → List Nat → List (Nat × Nat) → List (Nat × Nat)
  | 0, _, _, out => out
  | _ + 1, [], _, out => out
  | fuel + 1, (v, d) :: queue, seen, out =>
    let fresh := (neighbors g v).filter (· ∉ seen)
    bfs g fuel (queue ++ fresh.map (·, d + 1)) (seen ++ fresh) (out ++ fresh.map (·, d + 1))

def distances (g : Mat) (source : Nat) : List (Nat × Nat) :=
  bfs g (g.length + 1) [(source, 0)] [source] [(source, 0)]

def distance (g : Mat) (u v : Nat) : Option Nat :=
  ((distances g u).find? fun x => x.1 = v).map (·.2)

def connected (g : Mat) : Bool :=
  (List.range g.length).all fun v => (distance g 0 v).isSome

def removeEdge (g : Mat) (u v : Nat) : Mat :=
  (List.range g.length).map fun i => (List.range g.length).map fun j =>
    if (i = u ∧ j = v) ∨ (i = v ∧ j = u) then 0 else (g.getD i []).getD j 0

def girth (g : Mat) : Nat :=
  let lengths := (graphEdges g).filterMap fun (u, v) => (distance (removeEdge g u v) u v).map (· + 1)
  match lengths with
  | [] => 0
  | x :: xs => xs.foldl min x

def diameter (g : Mat) : Nat :=
  if !connected g then g.length else
    ((List.range g.length).flatMap fun u =>
      (List.range g.length).filterMap fun v => distance g u v).foldl max 0

def properColoring (g : Mat) (colors : List Nat) : Bool :=
  (graphEdges g).all fun (u, v) => colors.getD u 0 ≠ colors.getD v 0

def colorable (g : Mat) (k : Nat) : Bool :=
  (tuples k g.length).any (properColoring g)

def chromaticNumber (g : Mat) : Nat :=
  (((List.range g.length).map (· + 1)).find? (colorable g)).getD g.length

def isClique (g : Mat) (vertices : List Nat) : Bool :=
  vertices.all fun u => vertices.all fun v => if u < v then adjacent g u v else true

def isIndependent (g : Mat) (vertices : List Nat) : Bool :=
  vertices.all fun u => vertices.all fun v => if u < v then !adjacent g u v else true

def largestSet (g : Mat) (accepts : Mat → List Nat → Bool) : Nat :=
  (List.range (g.length + 1)).foldl (fun best k =>
    if (combos (List.range g.length) k).any (accepts g) then k else best) 0

def cliqueNumber (g : Mat) : Nat := largestSet g isClique

def independenceNumber (g : Mat) : Nat := largestSet g isIndependent

def isBipartite (g : Mat) : Bool := colorable g 2

def degreeSequence (g : Mat) : List Nat :=
  sortBy (· ≥ ·) ((List.range g.length).map fun v => (neighbors g v).length)

def graphFamily : Family → Bool
  | .explicit _ _ | .allGraphs _ | .edgeSubgraphs _ _ | .cayleyGraphs _ => true
  | _ => false

inductive Op
  | connected | girth | diameter | chromaticNumber | cliqueNumber | independenceNumber
  | isBipartite | degreeSequence | canonicalForm

inductive Value
  | degreeSequence (degrees : List Nat)
  | matrix (adjacency : Mat)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let ms := f.members
  if !graphFamily f || !ms.all wellFormed then .invalid else
  match op with
  | .connected => reduceBool red ms (ms.map connected)
  | .girth => reduceInt red ms (ms.map girth)
  | .diameter => reduceInt red ms (ms.map diameter)
  | .chromaticNumber => reduceInt red ms (ms.map chromaticNumber)
  | .cliqueNumber => reduceInt red ms (ms.map cliqueNumber)
  | .independenceNumber => reduceInt red ms (ms.map independenceNumber)
  | .isBipartite => reduceBool red ms (ms.map isBipartite)
  | .degreeSequence => reduceValues red (ms.map fun g => .degreeSequence (degreeSequence g))
  | .canonicalForm => reduceValues red (ms.map fun g => .matrix (graphCanonical g))

end Graphs
