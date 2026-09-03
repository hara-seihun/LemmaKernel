import Lk.Reference

/-!
# posets: executable reference

Each family member denotes one finite poset. An `explicit` member is its square 0/1 order matrix.
A `subsets` or `subsets_of` member is a list of sets, represented by nonzero row supports and
ordered by inclusion. A `range` member `x` denotes the positive divisors of `x`, in increasing
order, under divisibility. Invalid presentations make the request invalid.

The definitions use structural recursion and exhaustive finite lists so Lean's kernel can evaluate
small oracle cases. Element labels always follow the presentation order. In particular, the
Möbius matrix is row-major in that order.
-/

namespace Posets

open Lk

abbrev Rel := List (List Bool)
abbrev ZMat := List (List Int)

def relates (r : Rel) (i j : Nat) : Bool := (r.getD i []).getD j false

def square (m : Mat) : Bool := m.all fun row => row.length = m.length

def relationFromMatrix (m : Mat) : Option Rel :=
  let n := m.length
  if !square m || !m.all (fun row => row.all fun x => x = 0 || x = 1) then none
  else
    let r := m.map fun row => row.map (· = 1)
    let refl := (List.range n).all fun i => relates r i i
    let antisymm := (List.range n).all fun i => (List.range n).all fun j =>
      i = j || !(relates r i j && relates r j i)
    let trans := (List.range n).all fun i => (List.range n).all fun j =>
      (List.range n).all fun k => !(relates r i j && relates r j k) || relates r i k
    if refl && antisymm && trans then some r else none

def supports (rows : Mat) : List (List Bool) := rows.map fun row => row.map (· ≠ 0)

def subsetRelation (rows : Mat) : Option Rel :=
  let ss := supports rows
  if !ss.Nodup then none
  else some (ss.map fun a => ss.map fun b => List.zipWith (fun x y => !x || y) a b |>.all id)

def divisors (x : Nat) : List Nat := ((List.range x).map (· + 1)).filter fun d => x % d = 0

def divisorRelation (x : Nat) : Option Rel :=
  if x = 0 then none
  else
    let ds := divisors x
    some (ds.map fun a => ds.map fun b => b % a = 0)

def collectOptions : List (Option α) → Option (List α)
  | [] => some []
  | none :: _ => none
  | some x :: xs => (collectOptions xs).map (x :: ·)

def relations : Family → Option (List Rel)
  | .explicit _ batch => collectOptions (batch.map relationFromMatrix)
  | f@(.subsets _ _ _) => collectOptions (f.members.map subsetRelation)
  | f@(.subsetsOf _ _) => collectOptions (f.members.map subsetRelation)
  | .range a b => collectOptions ((List.range (b - a)).map fun i => divisorRelation (a + i))
  | _ => none

/-! ## Poset operations -/

def selected (mask i : Nat) : Bool := mask / (2 ^ i) % 2 = 1

def selectedCount (n mask : Nat) : Nat := ((List.range n).filter fun i => selected mask i).length

def isChainMask (r : Rel) (mask : Nat) : Bool :=
  let n := r.length
  (List.range n).all fun i => (List.range n).all fun j =>
    !selected mask i || !selected mask j || i = j || relates r i j || relates r j i

def isAntichainMask (r : Rel) (mask : Nat) : Bool :=
  let n := r.length
  (List.range n).all fun i => (List.range n).all fun j =>
    !selected mask i || !selected mask j || i = j || !(relates r i j || relates r j i)

def height (r : Rel) : Nat :=
  (List.range (2 ^ r.length)).foldl
    (fun best mask => if isChainMask r mask then max best (selectedCount r.length mask) else best) 0

def width (r : Rel) : Nat :=
  (List.range (2 ^ r.length)).foldl
    (fun best mask => if isAntichainMask r mask then max best (selectedCount r.length mask) else best) 0

def strict (r : Rel) (i j : Nat) : Bool := i ≠ j && relates r i j

def linearExtensionsAux (r : Rel) : Nat → List Nat → Nat
  | 0, xs => if xs.isEmpty then 1 else 0
  | fuel + 1, xs =>
    if xs.isEmpty then 1
    else
      let mins := xs.filter fun x => xs.all fun y => !strict r y x
      mins.foldl (fun total x => total + linearExtensionsAux r fuel (xs.filter (· ≠ x))) 0

def linearExtensionCount (r : Rel) : Nat := linearExtensionsAux r r.length (List.range r.length)

def meet? (r : Rel) (a b : Nat) : Option Nat :=
  (List.range r.length).find? fun x =>
    relates r x a && relates r x b &&
      (List.range r.length).all fun y => !(relates r y a && relates r y b) || relates r y x

def join? (r : Rel) (a b : Nat) : Option Nat :=
  (List.range r.length).find? fun x =>
    relates r a x && relates r b x &&
      (List.range r.length).all fun y => !(relates r a y && relates r b y) || relates r x y

def isLattice (r : Rel) : Bool :=
  (List.range r.length).all fun a => (List.range r.length).all fun b =>
    (meet? r a b).isSome && (join? r a b).isSome

def distributiveAt (r : Rel) (x y z : Nat) : Bool :=
  match meet? r y z, join? r y z, meet? r x y, meet? r x z, join? r x y, join? r x z with
  | some yzMeet, some yzJoin, some xyMeet, some xzMeet, some xyJoin, some xzJoin =>
    meet? r x yzJoin = join? r xyMeet xzMeet &&
    join? r x yzMeet = meet? r xyJoin xzJoin
  | _, _, _, _, _, _ => false

def isDistributive (r : Rel) : Bool :=
  isLattice r && (List.range r.length).all fun x => (List.range r.length).all fun y =>
    (List.range r.length).all fun z => distributiveAt r x y z

def orderPolynomial (r : Rel) (t : Nat) : Nat :=
  ((tuples t r.length).filter fun values =>
    (List.range r.length).all fun i => (List.range r.length).all fun j =>
      !relates r i j || values.getD i 0 ≤ values.getD j 0).length

def mu (r : Rel) : Nat → Nat → Nat → Int
  | 0, i, j => if !relates r i j then 0 else if i = j then 1 else 0
  | fuel + 1, i, j =>
    if !relates r i j then 0
    else if i = j then 1
    else -((List.range r.length).foldl (fun total k =>
      if relates r i k && relates r k j && k ≠ j then total + mu r fuel i k else total) 0)

def mobiusFunction (r : Rel) : ZMat :=
  (List.range r.length).map fun i => (List.range r.length).map fun j => mu r r.length i j

/-! ## Operations and values -/

inductive Op
  | mobiusFunction | linearExtensionCount | isLattice | isDistributive | width | height
  | orderPolynomial (t : Nat)

inductive Value
  | mobius (m : ZMat)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match relations f with
  | none => .invalid
  | some rs => match op with
    | .mobiusFunction => reduceValues red (rs.map fun r => .mobius (mobiusFunction r))
    | .linearExtensionCount =>
      if rs.all fun r => r.length ≤ 24 then reduceInt red f.members (rs.map linearExtensionCount) else .invalid
    | .isLattice => reduceBool red f.members (rs.map isLattice)
    | .isDistributive => reduceBool red f.members (rs.map isDistributive)
    | .width => reduceInt red f.members (rs.map width)
    | .height => reduceInt red f.members (rs.map height)
    | .orderPolynomial t =>
      if rs.all fun r => r.length ≤ 24 then reduceInt red f.members (rs.map fun r => orderPolynomial r t) else .invalid

end Posets
