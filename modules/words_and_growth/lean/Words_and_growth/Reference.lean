import Lk.Reference

/-!
# words_and_growth: executable reference

Letters `2*i` and `2*i+1` are generator `i` and its inverse. An encoded relation gives an
 equation between two words. The larger side in shortlex order rewrites to the smaller one, and
adjacent inverse pairs rewrite to the empty word. A presentation is accepted only when all
critical pairs join, so the terminating system has one shortlex normal form for each group
element.

The reference deliberately enumerates words. Oracle cases keep radii small; the native backend
uses a finite automaton for growth coefficients.
-/

namespace Words_and_growth

open Lk

abbrev Word := List Nat

structure Equation where
  left : Word
  right : Word
  deriving DecidableEq, Repr

structure Rule where
  lhs : Word
  rhs : Word
  deriving DecidableEq, Repr


def lexLt : Word → Word → Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | x :: xs, y :: ys => if x < y then true else if y < x then false else lexLt xs ys

def shortlexLt (a b : Word) : Bool :=
  if a.length < b.length then true else if b.length < a.length then false else lexLt a b

/-- `[left length, right length, left, right, padding]`, where every padding entry is `q`. -/
def decodeEquation (q : Nat) (row : Vec) : Option Equation :=
  if row.length < 2 then none else
  let nl := row.getD 0 0
  let nr := row.getD 1 0
  if row.length < 2 + nl + nr then none else
  let left := (row.drop 2).take nl
  let right := (row.drop (2 + nl)).take nr
  let padding := row.drop (2 + nl + nr)
  if (left ++ right).all (fun x => x < q) ∧ padding.all (· = q)
  then some { left, right }
  else none

def decodeEquations (q : Nat) : List Vec → Option (List Equation)
  | [] => some []
  | row :: rows =>
    match decodeEquation q row, decodeEquations q rows with
    | some e, some es => some (e :: es)
    | _, _ => none

/-- Orient an equation toward its shortlex-smaller side. Trivial equations add no rule. -/
def orient (e : Equation) : Option Rule :=
  if e.left = e.right then none
  else if shortlexLt e.left e.right then some { lhs := e.right, rhs := e.left }
  else some { lhs := e.left, rhs := e.right }

def inverseRules (g : Nat) : List Rule :=
  (List.range g).flatMap fun i =>
    [{ lhs := [2 * i, 2 * i + 1], rhs := [] },
     { lhs := [2 * i + 1, 2 * i], rhs := [] }]

def orientedRules (g : Nat) (equations : List Equation) : List Rule :=
  ((equations.filterMap orient) ++ inverseRules g).eraseDups


def firstSome (f : α → Option β) : List α → Option β
  | [] => none
  | x :: xs => match f x with
    | some y => some y
    | none => firstSome f xs

def replaceAt (w : Word) (pos len : Nat) (replacement : Word) : Word :=
  w.take pos ++ replacement ++ w.drop (pos + len)

def applyAt (w : Word) (pos : Nat) (r : Rule) : Option Word :=
  if (w.drop pos).take r.lhs.length = r.lhs
  then some (replaceAt w pos r.lhs.length r.rhs)
  else none

def rewriteOnce (rules : List Rule) (w : Word) : Option Word :=
  firstSome (fun pos => firstSome (applyAt w pos) rules) (List.range (w.length + 1))

/-- The number of words of length at most `n`; every rewrite drops at least one position in this
finite shortlex order. -/
def shortlexVolume (q n : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun total k => total + q ^ k) 0

def normaliseFuel (rules : List Rule) : Nat → Word → Option Word
  | 0, w => match rewriteOnce rules w with
    | none => some w
    | some _ => none
  | fuel + 1, w => match rewriteOnce rules w with
    | none => some w
    | some next => normaliseFuel rules fuel next

def normalise (q : Nat) (rules : List Rule) (w : Word) : Option Word :=
  normaliseFuel rules (shortlexVolume q w.length) w


def inclusions (outer inner : Rule) : List (Word × Word) :=
  if inner.lhs.length ≤ outer.lhs.length then
    (List.range (outer.lhs.length - inner.lhs.length + 1)).filterMap fun pos =>
      if (outer.lhs.drop pos).take inner.lhs.length = inner.lhs
      then some (outer.rhs, replaceAt outer.lhs pos inner.lhs.length inner.rhs)
      else none
  else []

def overlaps (left right : Rule) : List (Word × Word) :=
  ((List.range (min left.lhs.length right.lhs.length)).map (· + 1)).filterMap fun k =>
    if left.lhs.drop (left.lhs.length - k) = right.lhs.take k then
      some (left.rhs ++ right.lhs.drop k,
            left.lhs.take (left.lhs.length - k) ++ right.rhs)
    else none

def criticalPairs (rules : List Rule) : List (Word × Word) :=
  rules.flatMap fun left => rules.flatMap fun right => inclusions left right ++ overlaps left right

def rulesComplete (q : Nat) (rules : List Rule) : Bool :=
  (criticalPairs rules).all fun pair => normalise q rules pair.1 = normalise q rules pair.2

/-- Decode and validate a finite complete shortlex-decreasing rewriting presentation. -/
def buildRules (g : Nat) (encoded : List Vec) : Option (List Rule) :=
  if g = 0 ∨ 128 < g ∨ 256 < encoded.length ∨ encoded.any (fun row => 514 < row.length)
  then none
  else match decodeEquations (2 * g) encoded with
    | none => none
    | some equations =>
      let rules := orientedRules g equations
      if rulesComplete (2 * g) rules then some rules else none


def irreducible (rules : List Rule) (w : Word) : Bool := (rewriteOnce rules w).isNone

def isGeodesicWord (q : Nat) (rules : List Rule) (w : Word) : Bool :=
  match normalise q rules w with
  | some nf => nf.length = w.length
  | none => false

def sphereSize (q : Nat) (rules : List Rule) (n : Nat) : Nat :=
  ((tuples q n).filter (irreducible rules)).length

def ballSize (q : Nat) (rules : List Rule) (n : Nat) : Nat :=
  (List.range (n + 1)).foldl (fun total k => total + sphereSize q rules k) 0

def geodesicCount (q : Nat) (rules : List Rule) (n : Nat) : Nat :=
  ((tuples q n).filter (isGeodesicWord q rules)).length

def radius (m : Mat) : Nat := (m.headD []).headD 0

def word (m : Mat) : Word := m.headD []


inductive Op
  | ballSize (generators : Nat) (relations : List Vec)
  | sphereSize (generators : Nat) (relations : List Vec)
  | geodesicCount (generators : Nat) (relations : List Vec)
  | isGeodesic (generators : Nat) (relations : List Vec)

inductive Value deriving DecidableEq, Repr


def Op.generators : Op → Nat
  | .ballSize g _ | .sphereSize g _ | .geodesicCount g _ | .isGeodesic g _ => g

def Op.relations : Op → List Vec
  | .ballSize _ r | .sphereSize _ r | .geodesicCount _ r | .isGeodesic _ r => r

def run (op : Op) (family : Family) (red : Red) : Result Value :=
  let g := op.generators
  let q := 2 * g
  match buildRules g op.relations with
  | none => .invalid
  | some rules =>
    let members := family.members
    match op, family with
    | .ballSize _ _, .range _ _ => reduceInt red members (members.map fun m => ballSize q rules (radius m))
    | .sphereSize _ _, .range _ _ => reduceInt red members (members.map fun m => sphereSize q rules (radius m))
    | .geodesicCount _ _, .range _ _ => reduceInt red members (members.map fun m => geodesicCount q rules (radius m))
    | .isGeodesic _ _, .words alphabet _ =>
      if alphabet = q then reduceBool red members (members.map fun m => isGeodesicWord q rules (word m))
      else .invalid
    | _, _ => .invalid

end Words_and_growth
