import Orbits.Reference

/-!
# cayley: executable reference

The group is the closure of permutation generators. A family member is a list of distinct,
nonidentity group elements, one permutation image list per row. The graph has the group elements
as vertices. It uses `SimpleGraph.fromRel` semantics: vertices `x` and `y` are adjacent when they
are distinct and either `x⁻¹y` or `y⁻¹x` belongs to the member. Thus arbitrary members are
symmetrised, while inverse-closed members give the usual undirected Cayley graph.

The graph algorithms and exhaustive isomorphism checks are structural recursion over lists. This
file is the test oracle, so its small cases favour direct definitions over fast algorithms.
-/

namespace Cayley

open Lk

abbrev Group := Orbits.Group


def groupGenerators : Group → Option (List Perm)
  | .perms gens => some gens
  | .mats _ _ => none

def isPermutation (n : Nat) (g : Perm) : Bool :=
  decide (g.length = n ∧ g.Nodup ∧ ∀ x ∈ g, x < n)

def groupWellFormed (gens : List Perm) : Bool :=
  !gens.isEmpty ∧ gens.all (isPermutation (gens.headD []).length)

def supportedFamily : Family → Bool
  | .explicit _ _ | .subsets _ _ _ | .subsetsOf _ _ => true
  | _ => false


def mulIndex (es : List Perm) (a b : Nat) : Nat :=
  es.idxOf (compose (es.getD a []) (es.getD b []))

def identityIndex (es : List Perm) : Nat :=
  es.idxOf (identityPerm (es.headD []).length)

def inverseIndex (es : List Perm) (a : Nat) : Nat :=
  let e := identityIndex es
  (List.range es.length).find? (fun b => mulIndex es a b = e ∧ mulIndex es b a = e) |>.getD e


def memberIndices (es : List Perm) (s : Mat) : List Nat := s.map es.idxOf

def validMember (es : List Perm) (s : Mat) : Bool :=
  let e := identityPerm (es.headD []).length
  decide (s.Nodup ∧ ∀ x ∈ s, x ∈ es ∧ x ≠ e)

def inverseClosed (es : List Perm) (s : List Nat) : Bool :=
  s.all fun x => inverseIndex es x ∈ s


def difference (es : List Perm) (x y : Nat) : Nat :=
  mulIndex es (inverseIndex es x) y

def adj (es : List Perm) (s : List Nat) (x y : Nat) : Bool :=
  x ≠ y ∧ (difference es x y ∈ s ∨ difference es y x ∈ s)

def neighbours (es : List Perm) (s : List Nat) (x : Nat) : List Nat :=
  (List.range es.length).filter (adj es s x)


def reachable (es : List Perm) (s : List Nat) (start : Nat) : List Nat :=
  saturate (neighbours es s) fuel [start]

def connected (es : List Perm) (s : List Nat) : Bool :=
  (reachable es s (identityIndex es)).length = es.length

def isRegularOfDegree (es : List Perm) (s : List Nat) (degree : Nat) : Bool :=
  (List.range es.length).all fun x => (neighbours es s x).length = degree


def addFresh (seen : List Nat) (xs : List Nat) : List Nat :=
  xs.foldl (fun acc x => if x ∈ acc then acc else acc ++ [x]) seen

def nextFrontier (es : List Perm) (s : List Nat) (blocked : Option (Nat × Nat))
    (frontier seen : List Nat) : List Nat :=
  let candidates := frontier.flatMap fun x =>
    (neighbours es s x).filter fun y =>
      match blocked with
      | none => true
      | some (a, b) => !((x = a ∧ y = b) ∨ (x = b ∧ y = a))
  (addFresh seen candidates).drop seen.length

def distanceAux (es : List Perm) (s : List Nat) (blocked : Option (Nat × Nat))
    (target : Nat) : Nat → Nat → List Nat → List Nat → Option Nat
  | 0, _, _, _ => none
  | fuel + 1, depth, frontier, seen =>
      if target ∈ frontier then some depth
      else
        let next := nextFrontier es s blocked frontier seen
        if next.isEmpty then none
        else distanceAux es s blocked target fuel (depth + 1) next (seen ++ next)

def distance (es : List Perm) (s : List Nat) (blocked : Option (Nat × Nat))
    (x y : Nat) : Option Nat :=
  distanceAux es s blocked y (es.length + 1) 0 [x] [x]


def girth (es : List Perm) (s : List Nat) : Nat :=
  let lengths := (List.range es.length).flatMap fun x =>
    (List.range es.length).filterMap fun y =>
      if x < y ∧ adj es s x y then
        (distance es s (some (x, y)) x y).map (· + 1)
      else none
  match lengths with
  | [] => 0
  | x :: xs => xs.foldl min x

def diameter (es : List Perm) (s : List Nat) : Nat :=
  let ds := (List.range es.length).flatMap fun x =>
    (List.range es.length).map fun y => distance es s none x y
  if ds.any Option.isNone then 0
  else ds.foldl (fun d x => max d (x.getD 0)) 0


/-! ## Brute-force graph and group isomorphisms -/

def insertEverywhere (x : α) : List α → List (List α)
  | [] => [[x]]
  | y :: ys => (x :: y :: ys) :: (insertEverywhere x ys).map (y :: ·)

def permutations : List α → List (List α)
  | [] => [[]]
  | x :: xs => (permutations xs).flatMap (insertEverywhere x)

def preservesGraph (es : List Perm) (s t : List Nat) (p : Perm) : Bool :=
  (List.range es.length).all fun x =>
    (List.range es.length).all fun y =>
      adj es s x y == adj es t (p.getD x 0) (p.getD y 0)

def graphIsomorphic (es : List Perm) (s t : List Nat) : Bool :=
  (permutations (List.range es.length)).any (preservesGraph es s t)

def autOrder (es : List Perm) (s : List Nat) : Nat :=
  ((permutations (List.range es.length)).filter (preservesGraph es s s)).length


def preservesGroup (es : List Perm) (p : Perm) : Bool :=
  p.getD (identityIndex es) 0 = identityIndex es ∧
  (List.range es.length).all fun x =>
    (List.range es.length).all fun y =>
      p.getD (mulIndex es x y) 0 = mulIndex es (p.getD x 0) (p.getD y 0)

def sameSet (n : Nat) (s t : List Nat) : Bool :=
  (List.range n).all fun x => (x ∈ s) == (x ∈ t)

def automorphic (es : List Perm) (s t : List Nat) : Bool :=
  (permutations (List.range es.length)).any fun p =>
    preservesGroup es p ∧ sameSet es.length (s.map (p.getD · 0)) t


def inverseAtoms (es : List Perm) : List (List Nat) :=
  let e := identityIndex es
  ((List.range es.length).filter fun x => x ≠ e ∧ x ≤ inverseIndex es x).map fun x =>
    let y := inverseIndex es x
    if x = y then [x] else [x, y]

def chooseAtoms : List (List Nat) → List (List Nat)
  | [] => [[]]
  | a :: atoms =>
      let rest := chooseAtoms atoms
      rest.map (a ++ ·) ++ rest

def connectionSets (es : List Perm) (k : Nat) : List (List Nat) :=
  (chooseAtoms (inverseAtoms es)).filter fun s => s.length = k

def isCISet (es : List Perm) (s : List Nat) : Bool :=
  if !inverseClosed es s then false
  else (connectionSets es s.length).all fun t =>
    if graphIsomorphic es s t then automorphic es s t else true


/-! ## Operations and reductions -/

inductive Op
  | connected (group : Group)
  | isRegularOfDegree (degree : Nat) (group : Group)
  | girth (group : Group)
  | diameter (group : Group)
  | autOrder (group : Group)
  | isCiSet (group : Group)

inductive Value deriving DecidableEq, Repr


def Op.group : Op → Group
  | .connected g | .isRegularOfDegree _ g | .girth g | .diameter g | .autOrder g | .isCiSet g => g

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match groupGenerators op.group with
  | none => .invalid
  | some gens =>
    let es := permElements gens
    let ms := f.members
    if !groupWellFormed gens || !supportedFamily f || !(ms.all (validMember es)) then .invalid
    else
      let sets := ms.map (memberIndices es)
      match op with
      | .connected _ => reduceBool red ms (sets.map (connected es))
      | .isRegularOfDegree d _ => reduceBool red ms (sets.map fun s => isRegularOfDegree es s d)
      | .girth _ => reduceInt red ms (sets.map (girth es))
      | .diameter _ => reduceInt red ms (sets.map (diameter es))
      | .autOrder _ => reduceInt red ms (sets.map (autOrder es))
      | .isCiSet _ => reduceBool red ms (sets.map (isCISet es))

end Cayley
