import Lk.Reference

/-!
# permutation_statistics: executable reference

Permutations are zero-based image lists. A family is either an `orbits.perms` batch wrapped by
`explicit`, or `groupElements` of permutation generators. The reference uses only structural
recursion and list folds so Lean's kernel can evaluate every oracle case.

The `cycleType` value is a dense code. Complete cycle-length partitions, including parts of size
one, are sorted in descending lexicographic order. The code is the zero-based position in that
list. Classical patterns use increasing position subsequences and zero-based relative order.
`bruhatLeq upper` uses the rank criterion for strong Bruhat order.
-/

namespace Permutation_statistics

open Lk


def validPerm (g : Perm) : Bool := sortBy (· ≤ ·) g = List.range g.length

def inversions (g : Perm) : Nat :=
  (List.range g.length).foldl (fun total i =>
    total + ((List.range g.length).filter fun j => i < j ∧ g.getD i 0 > g.getD j 0).length) 0

def descents (g : Perm) : Nat :=
  ((List.range (g.length - 1)).filter fun i => g.getD i 0 > g.getD (i + 1) 0).length

def majorIndex (g : Perm) : Nat :=
  (List.range (g.length - 1)).foldl (fun total i =>
    if g.getD i 0 > g.getD (i + 1) 0 then total + i + 1 else total) 0

/-- The cycle beginning at `start`, with `fuel` bounding traversal of malformed inputs. -/
def cyclePoints (g : Perm) (start : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | fuel + 1, x =>
    let y := g.getD x start
    if y = start then [x] else x :: cyclePoints g start fuel y

def cycleLengths (g : Perm) : List Nat :=
  let state := (List.range g.length).foldl (fun state i =>
    if i ∈ state.1 then state
    else
      let points := cyclePoints g i g.length i
      (state.1 ++ points, state.2 ++ [points.length])) ([], [])
  sortBy (· ≥ ·) state.2

/-- Partitions of `remaining` with parts at most `maximum`, in descending lexicographic order.
The first argument is structural fuel; `n + 1` is enough for a partition of `n`. -/
def partitionsAux : Nat → Nat → Nat → List (List Nat)
  | 0, _, _ => []
  | _ + 1, 0, _ => [[]]
  | fuel + 1, remaining + 1, maximum =>
    let top := min (remaining + 1) maximum
    (List.range top).reverse.flatMap fun d =>
      let part := d + 1
      (partitionsAux fuel (remaining + 1 - part) part).map (part :: ·)

def partitions (n : Nat) : List (List Nat) := partitionsAux (n + 1) n n

def cycleTypeCode (g : Perm) : Nat := (partitions g.length).idxOf (cycleLengths g)


def patternCompatible (pattern selected : List Nat) (value : Nat) : Bool :=
  (List.range selected.length).all fun j =>
    (selected.getD j 0 < value) == (pattern.getD j 0 < pattern.getD selected.length 0)

/-- Search increasing positions. Pairwise order checks prune a candidate as soon as it cannot
have the pattern's relative order. -/
def containsPatternAux (g pattern : Perm) : Nat → Nat → List Nat → Bool
  | 0, _, selected => selected.length = pattern.length
  | fuel + 1, start, selected =>
    if selected.length = pattern.length then true
    else
      (List.range (g.length - start)).any fun offset =>
        let i := start + offset
        let value := g.getD i 0
        patternCompatible pattern selected value &&
          containsPatternAux g pattern fuel (i + 1) (selected ++ [value])

def containsPattern (g pattern : Perm) : Bool :=
  containsPatternAux g pattern (g.length + 1) 0 []

def patternAvoids (g : Perm) (patterns : List Perm) : Bool :=
  patterns.all fun pattern => !containsPattern g pattern


def upperRank (g : Perm) (pref threshold : Nat) : Nat :=
  ((g.take pref).filter fun value => threshold ≤ value).length

/-- Strong Bruhat order by the northwest rank criterion, written using upper value thresholds. -/
def bruhatLeq (lower upper : Perm) : Bool :=
  lower.length = upper.length ∧
    (List.range (lower.length + 1)).all fun pref =>
      (List.range (lower.length + 1)).all fun threshold =>
        upperRank lower pref threshold ≤ upperRank upper pref threshold


def sameDegree (gs : List Perm) : Bool :=
  gs.all fun g => g.length = (gs.headD []).length

def supportedFamily : Family → Bool
  | .explicit p batch =>
    p = 0 ∧ sameDegree (batch.map (·.headD [])) ∧
      batch.all fun member => member.length = 1 ∧ validPerm (member.headD [])
  | .groupElements generators =>
    !generators.isEmpty ∧ sameDegree generators ∧ generators.all validPerm
  | _ => false

def familyPerms (f : Family) : List Perm := f.members.map (·.headD [])


inductive Op
  | inversions | descents | majorIndex | cycleType
  | patternAvoids (patterns : List Perm)
  | bruhatLeq (upper : List Perm)

inductive Value deriving DecidableEq, Repr


def run (op : Op) (f : Family) (red : Red) : Result Value :=
  if !supportedFamily f then .invalid
  else
    let members := f.members
    let perms := familyPerms f
    match op with
    | .inversions => reduceInt red members (perms.map inversions)
    | .descents => reduceInt red members (perms.map descents)
    | .majorIndex => reduceInt red members (perms.map majorIndex)
    | .cycleType => reduceInt red members (perms.map cycleTypeCode)
    | .patternAvoids patterns =>
      if !(patterns.all validPerm) ∨ !sameDegree patterns then .invalid
      else reduceBool red members (perms.map fun g => patternAvoids g patterns)
    | .bruhatLeq upper =>
      match upper with
      | [bound] =>
        if !validPerm bound ∨ !(perms.all fun g => g.length = bound.length) then .invalid
        else reduceBool red members (perms.map fun g => bruhatLeq g bound)
      | _ => .invalid

end Permutation_statistics
