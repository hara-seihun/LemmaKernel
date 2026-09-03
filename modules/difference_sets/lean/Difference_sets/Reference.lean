import Lk.Reference

/-!
# difference_sets: executable reference

The ambient group is a permutation group. Its elements use `Lk.permElements`, hence the same
lexicographic order as `group_elements`. A difference is the ordered right quotient `x*y⁻¹`,
represented by the function `x ∘ y⁻¹`. `differenceMultiset` includes all ordered pairs,
so its identity entry is `|D|`; the predicates inspect the other entries.
-/

namespace Difference_sets

open Lk

/-- The inverse image list of a permutation. -/
def inversePerm (g : Perm) : Perm :=
  (List.range g.length).map fun i => g.idxOf i

/-- The right quotient `x*y⁻¹`, represented by `x ∘ y⁻¹`. -/
def quotient (x y : Perm) : Perm := compose (inversePerm y) x

def quotients (d : List Perm) : List Perm :=
  d.flatMap fun x => d.map fun y => quotient x y

/-- Multiplicities of all ordered quotients, in the supplied ambient-element order. -/
def differenceMultiset (elems d : List Perm) : List Nat :=
  let qs := quotients d
  elems.map fun g => (qs.filter (· = g)).length

def identityIndex (elems : List Perm) : Nat :=
  elems.idxOf (identityPerm (elems.headD []).length)

def isDifferenceSet (elems d : List Perm) : Bool :=
  let v := elems.length
  let k := d.length
  if v ≤ 1 then false
  else if k * (k - 1) % (v - 1) ≠ 0 then false
  else
    let lam := k * (k - 1) / (v - 1)
    let counts := differenceMultiset elems d
    (List.range v).all fun i =>
      if i = identityIndex elems then counts.getD i 0 = k else counts.getD i 0 = lam

def inverseClosed (d : List Perm) : Bool :=
  d.all fun x => inversePerm x ∈ d

def isPds (elems d : List Perm) (lam mu : Nat) : Bool :=
  let v := elems.length
  let k := d.length
  let e := identityPerm (elems.headD []).length
  if e ∈ d then false
  else if !inverseClosed d then false
  else if k * (k - 1) ≠ k * lam + (v - 1 - k) * mu then false
  else
    let counts := differenceMultiset elems d
    (List.range v).all fun i =>
      let g := elems.getD i []
      if g = e then counts.getD i 0 = k
      else if g ∈ d then counts.getD i 0 = lam
      else counts.getD i 0 = mu

def validForbidden (elems forbidden : List Perm) : Bool :=
  let n := (elems.headD []).length
  if forbidden.isEmpty then false
  else forbidden.all (·.length = n) ∧ (permElements forbidden).all (· ∈ elems)

def isRelativeDifferenceSet (elems d forbidden : List Perm) : Bool :=
  let subgroup := permElements forbidden
  let outside := elems.length - subgroup.length
  let k := d.length
  if outside = 0 then false
  else if k * (k - 1) % outside ≠ 0 then false
  else
    let lam := k * (k - 1) / outside
    let e := identityPerm (elems.headD []).length
    let counts := differenceMultiset elems d
    (List.range elems.length).all fun i =>
      let g := elems.getD i []
      if g = e then counts.getD i 0 = k
      else if g ∈ subgroup then counts.getD i 0 = 0
      else counts.getD i 0 = lam

inductive Op
  | isDifferenceSet
  | differenceMultiset
  | isPds (lambda mu : Nat)
  | isRelativeDifferenceSet (forbidden : List Perm)

/-- One materialised natural matrix per member. `difference_multiset` uses one row. -/
inductive Value
  | naturals (m : Mat)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match f with
  | .subsetsOf (.groupElements gens) _ =>
    let elems := permElements gens
    let ms := f.members
    match op with
    | .isDifferenceSet => reduceBool red ms (ms.map fun d => isDifferenceSet elems d)
    | .differenceMultiset =>
      reduceValues red (ms.map fun d => .naturals [differenceMultiset elems d])
    | .isPds lam mu => reduceBool red ms (ms.map fun d => isPds elems d lam mu)
    | .isRelativeDifferenceSet forbidden =>
      if validForbidden elems forbidden then
        reduceBool red ms (ms.map fun d => isRelativeDifferenceSet elems d forbidden)
      else .invalid
  | _ => .invalid

end Difference_sets
