import Lk.Reference

/-!
# graph_polynomials: executable reference

A graph is a `subsets` member drawn from a dictionary of rows `[u, v]` with
`u < v < vertices`. The labels are natural numbers and vertices absent from every edge remain
part of the graph. Coefficient vectors use ascending powers. Tutte coefficients are flattened
x-major, with the y exponent varying fastest.

The polynomial recurrences have an explicit fuel argument so Lean's kernel can evaluate every
oracle claim by structural recursion.
-/

namespace Graph_polynomials

open Lk

abbrev Edge := Nat × Nat
abbrev Poly := List Int

structure Graph where
  vertices : Nat
  edges : List Edge
  deriving DecidableEq, Repr

structure Term where
  x : Nat
  y : Nat
  coefficient : Int
  deriving DecidableEq, Repr


def edgeOf (row : Vec) : Edge := (row.getD 0 0, row.getD 1 0)

def edgesOf (m : Mat) : List Edge := m.map edgeOf

def validEdge (vertices : Nat) (row : Vec) : Bool :=
  decide (row.length = 2 ∧ row.getD 0 0 < row.getD 1 0 ∧ row.getD 1 0 < vertices)

def validDictionary (vertices : Nat) (d : List Vec) : Bool :=
  d.all (validEdge vertices) && decide d.Nodup


def zeros (length : Nat) : Poly := List.replicate length 0

def monomial (length degree : Nat) (coefficient : Int) : Poly :=
  (List.range length).map fun i => if i = degree then coefficient else 0

def polyAdd (length : Nat) (a b : Poly) : Poly :=
  (List.range length).map fun i => a.getD i 0 + b.getD i 0

def polySub (length : Nat) (a b : Poly) : Poly :=
  (List.range length).map fun i => a.getD i 0 - b.getD i 0


def hasLoop (edges : List Edge) : Bool := edges.any fun e => e.1 = e.2

def hasEdge (edges : List Edge) (u v : Nat) : Bool :=
  edges.any fun e => (e.1 = u ∧ e.2 = v) ∨ (e.1 = v ∧ e.2 = u)


def remapAfterContraction (u v x : Nat) : Nat :=
  let y := if x = v then u else x
  if v < y then y - 1 else y


def contractEdge (g : Graph) (e : Edge) (rest : List Edge) : Graph :=
  let mapped := rest.map fun f =>
    let a := remapAfterContraction e.1 e.2 f.1
    let b := remapAfterContraction e.1 e.2 f.2
    if a ≤ b then (a, b) else (b, a)
  ⟨g.vertices - 1, mapped⟩


def removeEndpoints (g : Graph) (e : Edge) (rest : List Edge) : Graph :=
  let kept := rest.filter fun f => f.1 ≠ e.1 ∧ f.2 ≠ e.1 ∧ f.1 ≠ e.2 ∧ f.2 ≠ e.2
  let lower := fun x => x - (if e.1 < x then 1 else 0) - (if e.2 < x then 1 else 0)
  ⟨g.vertices - 2, kept.map fun f => (lower f.1, lower f.2)⟩


def neighbours (edges : List Edge) (u : Nat) : List Nat :=
  edges.filterMap fun e => if e.1 = u then some e.2 else if e.2 = u then some e.1 else none


def reachable (vertices : Nat) (edges : List Edge) (u v : Nat) : Bool :=
  v ∈ saturate (neighbours edges) (vertices + 1) [u]


def chromaticRec : Nat → Graph → Poly
  | 0, g => zeros (g.vertices + 1)
  | fuel + 1, g =>
      if hasLoop g.edges then zeros (g.vertices + 1)
      else match g.edges with
        | [] => monomial (g.vertices + 1) g.vertices 1
        | e :: rest =>
            polySub (g.vertices + 1)
              (chromaticRec fuel ⟨g.vertices, rest⟩)
              (chromaticRec fuel (contractEdge g e rest))


def chromatic (vertices : Nat) (edges : List Edge) : Poly :=
  chromaticRec (edges.length + 1) ⟨vertices, edges⟩


def matchingRec : Nat → Graph → Poly
  | 0, g => zeros (g.vertices + 1)
  | fuel + 1, g =>
      match g.edges with
      | [] => monomial (g.vertices + 1) g.vertices 1
      | e :: rest =>
          polySub (g.vertices + 1)
            (matchingRec fuel ⟨g.vertices, rest⟩)
            (matchingRec fuel (removeEndpoints g e rest))


def matching (vertices : Nat) (edges : List Edge) : Poly :=
  matchingRec (edges.length + 1) ⟨vertices, edges⟩


def shiftX (terms : List Term) : List Term := terms.map fun t => { t with x := t.x + 1 }

def shiftY (terms : List Term) : List Term := terms.map fun t => { t with y := t.y + 1 }


def tutteTerms : Nat → Graph → List Term
  | 0, _ => []
  | fuel + 1, g =>
      match g.edges with
      | [] => [⟨0, 0, 1⟩]
      | e :: rest =>
          if e.1 = e.2 then shiftY (tutteTerms fuel ⟨g.vertices, rest⟩)
          else if !reachable g.vertices rest e.1 e.2 then
            shiftX (tutteTerms fuel (contractEdge g e rest))
          else
            tutteTerms fuel ⟨g.vertices, rest⟩ ++ tutteTerms fuel (contractEdge g e rest)


def tutte (vertices edgeCount : Nat) (edges : List Edge) : Poly :=
  let terms := tutteTerms (edges.length + 1) ⟨vertices, edges⟩
  (List.range (vertices + 1)).flatMap fun i =>
    (List.range (edgeCount + 1)).map fun j =>
      terms.foldl (fun coefficient t =>
        if t.x = i ∧ t.y = j then coefficient + t.coefficient else coefficient) 0


/-! ## Adjacency characteristic polynomial -/

def insertEverywhere (x : Nat) : List Nat → List (List Nat)
  | [] => [[x]]
  | y :: ys => (x :: y :: ys) :: (insertEverywhere x ys).map (fun zs => y :: zs)


def permutations : List Nat → List (List Nat)
  | [] => [[]]
  | x :: xs => (permutations xs).flatMap (insertEverywhere x)


def inversionCount (p : List Nat) : Nat :=
  (List.range p.length).foldl (fun total i =>
    total + ((List.range p.length).filter fun j => i < j ∧ p.getD j 0 < p.getD i 0).length) 0


def fixedCount (p : List Nat) : Nat :=
  ((List.range p.length).filter fun i => p.getD i 0 = i).length


def characteristicTermValid (edges : List Edge) (p : List Nat) : Bool :=
  (List.range p.length).all fun i => i = p.getD i 0 ∨ hasEdge edges i (p.getD i 0)


def negativeOnePow (n : Nat) : Int := if n % 2 = 0 then 1 else -1


def characteristic (vertices : Nat) (edges : List Edge) : Poly :=
  let ps := permutations (List.range vertices)
  (List.range (vertices + 1)).map fun degree =>
    ps.foldl (fun coefficient p =>
      if characteristicTermValid edges p && fixedCount p = degree then
        coefficient + negativeOnePow (inversionCount p + vertices - fixedCount p)
      else coefficient) 0


/-! ## Operations and values -/

inductive Op
  | chromatic (vertices : Nat)
  | tutte (vertices : Nat)
  | characteristic (vertices : Nat)
  | matching (vertices : Nat)

inductive Value
  | coefficients (values : List Int)
  deriving DecidableEq, Repr


def opVertices : Op → Nat
  | .chromatic n | .tutte n | .characteristic n | .matching n => n


def valueFor (op : Op) (edgeCount : Nat) (m : Mat) : Value :=
  let edges := edgesOf m
  match op with
  | .chromatic n => .coefficients (chromatic n edges)
  | .tutte n => .coefficients (tutte n edgeCount edges)
  | .characteristic n => .coefficients (characteristic n edges)
  | .matching n => .coefficients (matching n edges)


def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match f, red with
  | .subsets _ dictionary k, .all =>
      if validDictionary (opVertices op) dictionary then
        .values (f.members.map (valueFor op k))
      else .invalid
  | _, _ => .invalid

end Graph_polynomials
