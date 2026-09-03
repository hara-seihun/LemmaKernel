import Orbits.Reference

/-!
# burnside: executable reference

Permutation groups act on dictionary positions of `subsets` and `subsets_of`, and on coordinate
positions of `words`. Fixed subsets are unions of cycles. A fixed word is constant on every
cycle. The reference closes the generator group, computes cycle-count vectors, and applies those
two formulas. It never constructs a member of the acted-on family.

A cycle-count vector is `[c₁, ..., cₙ]`, where `cᵢ` is the number of cycles of length `i`.
Cycle-index terms are sorted lexicographically by this vector. Each term carries
`(group order, multiplicity, cycle-count vector)`.
-/

namespace Burnside

open Lk

/-- Follow the cycle containing `start`. `fuel = g.length` suffices for a permutation. -/
def cycleAux (g : Perm) (start : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | fuel + 1, x =>
    let y := g.getD x x
    x :: if y = start then [] else cycleAux g start fuel y

def bumpAt (xs : List Nat) (i : Nat) : List Nat :=
  (List.range xs.length).map fun j => xs.getD j 0 + if i = j then 1 else 0

/-- `[c₁, ..., cₙ]`, the cycle counts of `g`. -/
def cycleType (g : Perm) : List Nat :=
  let step := fun state i =>
    if i ∈ state.1 then state
    else
      let cyc := cycleAux g i g.length i
      (state.1 ++ cyc, bumpAt state.2 (cyc.length - 1))
  ((List.range g.length).foldl step ([], List.replicate g.length 0)).2

def validPerm (n : Nat) (g : Perm) : Bool :=
  g.length = n ∧ (List.range n).all fun i => (g.filter (· = i)).length = 1

def familyDegree : Family → Option Nat
  | .subsets _ dictionary _ => some dictionary.length
  | .subsetsOf inner _ => some inner.members.length
  | .words _ length => some length
  | _ => none

/-- Update coefficients of `product (1 + x^cycleLength)`, truncated after degree `k`. -/
def addSubsetCycle (k : Nat) (poly : List Nat) (cycleLength : Nat) : List Nat :=
  (List.range (k + 1)).map fun d =>
    poly.getD d 0 + if cycleLength ≤ d then poly.getD (d - cycleLength) 0 else 0

def cycleLengths (counts : List Nat) : List Nat :=
  (List.range counts.length).flatMap fun i => List.replicate (counts.getD i 0) (i + 1)

/-- Coefficient of `x^k` in `product_cycles (1 + x^cycleLength)`. -/
def fixedSubsets (k : Nat) (counts : List Nat) : Nat :=
  ((cycleLengths counts).foldl (addSubsetCycle k) (1 :: List.replicate k 0)).getD k 0

def fixedCount : Family → Perm → Option Nat
  | .subsets _ _ k, g => some (fixedSubsets k (cycleType g))
  | .subsetsOf _ k, g => some (fixedSubsets k (cycleType g))
  | .words alphabet _, g => some (alphabet ^ (cycleType g).foldl (· + ·) 0)
  | _, _ => none

def dedup [DecidableEq α] (xs : List α) : List α :=
  xs.foldl (fun out x => if x ∈ out then out else out ++ [x]) []

/-- Canonical sparse cycle-index terms `(denominator, multiplicity, cycle counts)`. -/
def cycleIndexTerms (elements : List Perm) : List (Nat × Nat × List Nat) :=
  let types := elements.map cycleType
  (sortBy lexLe (dedup types)).map fun t =>
    (elements.length, (types.filter (· = t)).length, t)

def groupElements (f : Family) : Orbits.Group → Option (List Perm)
  | .perms gens =>
    match familyDegree f with
    | some n => if !gens.isEmpty ∧ gens.all (validPerm n) then some (permElements gens) else none
    | none => none
  | .mats _ _ => none

inductive Op
  | orbitCount (group : Orbits.Group)
  | cycleIndex (group : Orbits.Group)
  | fixedCount (g : List Perm)

inductive Value
  | count (n : Nat)
  | cycleIndexTerm (term : Nat × Nat × List Nat)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match op, red with
  | .orbitCount group, .all =>
    match groupElements f group with
    | some elements =>
      let total := (elements.map fun g => (fixedCount f g).getD 0).foldl (· + ·) 0
      .values [.count (total / elements.length)]
    | none => .invalid
  | .cycleIndex group, .all =>
    match groupElements f group with
    | some elements => .values ((cycleIndexTerms elements).map fun t => .cycleIndexTerm t)
    | none => .invalid
  | .fixedCount gs, .all =>
    match familyDegree f, gs with
    | some n, [g] =>
      if validPerm n g then
        match fixedCount f g with
        | some count => .values [.count count]
        | none => .invalid
      else .invalid
    | _, _ => .invalid
  | _, _ => .invalid

end Burnside
