import Lk.Reference

/-!
# hypergraphs: executable reference

A hypergraph is a nonempty list of edges. Every edge is a strictly increasing list of vertices,
and the edge list is strictly lexicographically increasing. The full vertex set is
`{0, ..., vertices - 1}`, so isolated vertices remain visible. Generated inputs normally have the
form `subsets_of(subsets_of(range(0, vertices), uniformity), edge_count)`.

The implementation uses only structural recursion and finite lists so `decide +kernel` can run it.
-/

namespace Hypergraphs

open Lk

abbrev Hypergraph := Mat

def naturalsPrime : Nat := 18446744073709551615

/-- Strict order on a list of naturals. -/
def strictlyIncreasing : List Nat → Bool
  | [] | [_] => true
  | x :: y :: xs => x < y && strictlyIncreasing (y :: xs)

/-- Strict lexicographic order, using the runtime's Boolean non-strict order. -/
def lexLt (a b : List Nat) : Bool := a != b && lexLe a b

def edgesIncreasing : List Vec → Bool
  | [] | [_] => true
  | e :: f :: rest => lexLt e f && edgesIncreasing (f :: rest)

def validHypergraph (vertices uniformity edgeCount : Nat) (h : Hypergraph) : Bool :=
  h.length = edgeCount && edgeCount > 0 && uniformity ≥ 2 && uniformity ≤ vertices &&
    h.all (fun e => e.length = uniformity && strictlyIncreasing e && e.all (· < vertices)) &&
    edgesIncreasing h

def naturalSource : Family → Bool
  | .explicit p _ | .subsets p _ _ => p = naturalsPrime
  | .subsetsOf inner _ => naturalSource inner
  | .range _ _ | .words _ _ => true
  | _ => false

def acceptedFamily : Family → Bool
  | .explicit _ _ | .subsets _ _ _ | .subsetsOf _ _ => true
  | _ => false

/-- Check the shared shape and canonical edge encoding once for the whole family. -/
def validFamily (vertices : Nat) (f : Family) (members : List Mat) : Bool :=
  match members with
  | [] => false
  | first :: _ =>
    match first with
    | [] => false
    | edge :: _ =>
      let uniformity := edge.length
      let edgeCount := first.length
      vertices ≤ 64 && acceptedFamily f && naturalSource f &&
        members.all (validHypergraph vertices uniformity edgeCount)

/-! ## Linearity -/

def intersectionSize (a b : Vec) : Nat := (a.filter (· ∈ b)).length

def edgePairs : List Vec → List (Vec × Vec)
  | [] => []
  | e :: rest => rest.map (e, ·) ++ edgePairs rest

def isLinear (h : Hypergraph) : Bool :=
  (edgePairs h).all fun (e, f) => intersectionSize e f ≤ 1

/-! ## Weak vertex colouring -/

def monochromatic (colours : Vec) : Vec → Bool
  | [] => true
  | v :: vs => vs.all fun w => colours.getD w 0 = colours.getD v 0

def properColouring (h : Hypergraph) (colours : Vec) : Bool :=
  h.all fun edge => !monochromatic colours edge

def qColourable (vertices : Nat) (h : Hypergraph) (q : Nat) : Bool :=
  (tuples q vertices).any (properColouring h)

def firstColourable (vertices : Nat) (h : Hypergraph) : List Nat → Nat
  | [] => vertices + 1
  | q :: qs => if qColourable vertices h q then q else firstColourable vertices h qs

def colouringNumber (vertices : Nat) (h : Hypergraph) : Nat :=
  firstColourable vertices h ((List.range vertices).map (· + 1))

/-! ## Berge cycles -/

/-- Ordered selections without replacement. -/
def picks (xs : List Nat) : Nat → List (List Nat)
  | 0 => [[]]
  | length + 1 => xs.flatMap fun x => (picks (xs.erase x) length).map (x :: ·)

def cycleIncidence (h : Hypergraph) (vertices edges : List Nat) : Bool :=
  let next := vertices.drop 1 ++ vertices.take 1
  (List.zipWith (fun pair edgeIndex =>
    let edge := h.getD edgeIndex []
    pair.1 ∈ edge && pair.2 ∈ edge) (List.zip vertices next) edges).all id

def hasBergeCycle (vertexCount length : Nat) (h : Hypergraph) : Bool :=
  if length < 2 || length > vertexCount || length > h.length then false
  else
    (picks (List.range h.length) length).any fun edges =>
      (picks (List.range vertexCount) length).any fun vertices => cycleIncidence h vertices edges

def bergeGirthFrom (vertexCount : Nat) (h : Hypergraph) : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, length =>
    if hasBergeCycle vertexCount length h then length
    else bergeGirthFrom vertexCount h fuel (length + 1)

def bergeGirth (vertexCount : Nat) (h : Hypergraph) : Nat :=
  let bound := min vertexCount h.length
  if bound < 2 then 0 else bergeGirthFrom vertexCount h (bound - 1) 2

/-! ## Complete subhypergraphs and two-colour Ramsey searches -/

def containsClique (vertexCount uniformity cliqueSize : Nat) (h : Hypergraph) (red : Bool) : Bool :=
  (combos (List.range vertexCount) cliqueSize).any fun vertices =>
    (combos vertices uniformity).all fun edge => if red then edge ∈ h else edge ∉ h

def isCliqueFree (vertexCount cliqueSize : Nat) (h : Hypergraph) : Bool :=
  !containsClique vertexCount (h.headD []).length cliqueSize h true

def isRamseyColouring (vertexCount redClique blueClique : Nat) (h : Hypergraph) : Bool :=
  let uniformity := (h.headD []).length
  !containsClique vertexCount uniformity redClique h true &&
    !containsClique vertexCount uniformity blueClique h false

/-! ## Operations -/

inductive Op
  | isLinear (vertices : Nat)
  | colouringNumber (vertices : Nat)
  | hasBergeCycle (length vertices : Nat)
  | bergeGirth (vertices : Nat)
  | isCliqueFree (cliqueSize vertices : Nat)
  | isRamseyColouring (blueClique redClique vertices : Nat)

deriving DecidableEq, Repr

inductive Value deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let members := f.members
  match op with
  | .isLinear vertices =>
    if validFamily vertices f members then reduceBool red members (members.map isLinear) else .invalid
  | .colouringNumber vertices =>
    if validFamily vertices f members then
      reduceInt red members (members.map (colouringNumber vertices))
    else .invalid
  | .hasBergeCycle length vertices =>
    if validFamily vertices f members && length ≥ 2 then
      reduceBool red members (members.map (hasBergeCycle vertices length))
    else .invalid
  | .bergeGirth vertices =>
    if validFamily vertices f members then reduceInt red members (members.map (bergeGirth vertices))
    else .invalid
  | .isCliqueFree cliqueSize vertices =>
    let uniformity := ((members.headD []).headD []).length
    if validFamily vertices f members && cliqueSize ≥ uniformity then
      reduceBool red members (members.map (isCliqueFree vertices cliqueSize))
    else .invalid
  | .isRamseyColouring blueClique redClique vertices =>
    let uniformity := ((members.headD []).headD []).length
    if validFamily vertices f members && redClique ≥ uniformity && blueClique ≥ uniformity then
      reduceBool red members (members.map (isRamseyColouring vertices redClique blueClique))
    else .invalid

end Hypergraphs
