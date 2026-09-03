import Orbits.Reference

/-!
# designs: executable reference

A block on `v` labelled points is a `k x v` matrix whose rows are distinct standard basis
vectors in increasing point order. A family of such matrices is one indexed block family;
repeated blocks remain repeated indexed blocks. Every operation examines the whole family and
returns one natural-number matrix.
-/

namespace Designs

open Lk

abbrev Block := List Nat

structure BlockFamily where
  v : Nat
  k : Nat
  blocks : List Block
  deriving DecidableEq, Repr

def unitVec (v i : Nat) : Vec := (List.range v).map fun j => if i = j then 1 else 0

def pointOf (v : Nat) (row : Vec) : Option Nat :=
  (List.range v).find? fun i => row = unitVec v i

def strictlyIncreasing : List Nat → Bool
  | [] | [_] => true
  | x :: y :: xs => x < y && strictlyIncreasing (y :: xs)

def sequenceOptions : List (Option α) → Option (List α)
  | [] => some []
  | none :: _ => none
  | some x :: xs => (sequenceOptions xs).map (x :: ·)

def blockOf (v k : Nat) (m : Mat) : Option Block :=
  if m.length ≠ k then none
  else
    match sequenceOptions (m.map (pointOf v)) with
    | some b => if strictlyIncreasing b then some b else none
    | none => none

def parseFamily (f : Family) : Option BlockFamily :=
  let ms := f.members
  let v := ((ms.headD []).headD []).length
  let k := (ms.headD []).length
  if ms.isEmpty || v = 0 || k = 0 then none
  else (sequenceOptions (ms.map (blockOf v k))).map fun bs => { v := v, k := k, blocks := bs }

def contains (small block : Block) : Bool := small.all (· ∈ block)

def multiplicity (blocks : List Block) (s : Block) : Nat := (blocks.filter (contains s)).length

def lambdaVector (a : BlockFamily) (t : Nat) : List Nat :=
  (combos (List.range a.v) t).map (multiplicity a.blocks)

def uniform (xs : List Nat) : Bool := xs.all (· = xs.headD 0)

def isDesignData (a : BlockFamily) (t : Nat) : Mat :=
  let counts := lambdaVector a t
  [[if uniform counts then 1 else 0, counts.headD 0]]

def intersectionSize (a b : Block) : Nat := (a.filter (· ∈ b)).length

def pairIntersections : List Block → List Nat
  | [] => []
  | b :: bs => bs.map (intersectionSize b) ++ pairIntersections bs

def intersectionNumbers (a : BlockFamily) : List Nat :=
  let xs := pairIntersections a.blocks
  (List.range (a.k + 1)).map fun j => (xs.filter (· = j)).length

def dualDesignData (a : BlockFamily) : Mat :=
  let replications := (List.range a.v).map fun x => (a.blocks.filter fun b => x ∈ b).length
  let intersections := pairIntersections a.blocks
  let flag := a.blocks.length ≥ 2 && replications.headD 0 ≥ 2 && uniform replications && uniform intersections
  [[if flag then 1 else 0, intersections.headD 0]]

def coversPoints (a : BlockFamily) (c : List Nat) : Bool :=
  (List.range a.v).all fun x =>
    (c.filter fun i => x ∈ a.blocks.getD i []).length = 1

def parallelClasses (a : BlockFamily) (classSize : Nat) : List (List Nat) :=
  (combos (List.range a.blocks.length) classSize).filter (coversPoints a)

def firstSome (xs : List α) (f : α → Option β) : Option β :=
  xs.foldl (fun found x => match found with | some y => some y | none => f x) none

def resolutionSearch (classes : List (List Nat)) : Nat → List Nat → Option (List (List Nat))
  | 0, _ => none
  | _ + 1, [] => some []
  | fuel + 1, remaining@(_ :: _) =>
    let first := remaining.headD 0
    let candidates := classes.filter fun c => first ∈ c && c.all (· ∈ remaining)
    firstSome candidates fun c =>
      match resolutionSearch classes fuel (remaining.filter (· ∉ c)) with
      | some rest => some (c :: rest)
      | none => none

def resolution (a : BlockFamily) : Mat :=
  if a.v % a.k ≠ 0 then []
  else
    let classSize := a.v / a.k
    if classSize = 0 || a.blocks.isEmpty || a.blocks.length % classSize ≠ 0 then []
    else
      let classes := parallelClasses a classSize
      (resolutionSearch classes (a.blocks.length + 1) (List.range a.blocks.length)).getD []

/-! ## Kramer-Mesner matrices -/

def actSubset (g : Perm) (s : Block) : Block := sortBy (· ≤ ·) (s.map fun i => g.getD i 0)

def subsetOrbit (gens : List Perm) (s : Block) : List Block :=
  saturate (fun x => gens.map fun g => actSubset g x) fuel [s]

def subsetOrbits (gens : List Perm) : Nat → List Block → List (List Block)
  | 0, _ => []
  | _ + 1, [] => []
  | fuel + 1, x :: xs =>
    let orbit := subsetOrbit gens x
    orbit :: subsetOrbits gens fuel (xs.filter (· ∉ orbit))

def kramerMesner (v k t : Nat) (gens : List Perm) : Mat :=
  let ts := combos (List.range v) t
  let ks := combos (List.range v) k
  let rows := subsetOrbits gens (ts.length + 1) ts
  let cols := subsetOrbits gens (ks.length + 1) ks
  rows.map fun rowOrbit =>
    let representative := rowOrbit.headD []
    cols.map fun colOrbit => (colOrbit.filter (contains representative)).length

/-! ## Operations and values -/

inductive Op
  | isDesign (t : Nat)
  | lambdaVector (t : Nat)
  | isResolvable
  | dualIsDesign
  | intersectionNumbers
  | kramerMesnerMatrix (group : Orbits.Group) (t : Nat)

inductive Value
  | matrix (m : Mat)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match red with
  | .all =>
    match op with
    | .kramerMesnerMatrix (.perms gens) t =>
      match f with
      | .subsets _ d k =>
        if gens.isEmpty || t > k || k > d.length || !(gens.all fun g => g.length = d.length) then .invalid
        else .values [.matrix (kramerMesner d.length k t gens)]
      | _ => .invalid
    | .kramerMesnerMatrix (.mats _ _) _ => .invalid
    | _ =>
      match parseFamily f with
      | none => .invalid
      | some a =>
        match op with
        | .isDesign t => if t > a.k then .invalid else .values [.matrix (isDesignData a t)]
        | .lambdaVector t => if t > a.k then .invalid else .values [.matrix [lambdaVector a t]]
        | .isResolvable => .values [.matrix (resolution a)]
        | .dualIsDesign => .values [.matrix (dualDesignData a)]
        | .intersectionNumbers => .values [.matrix [intersectionNumbers a]]
        | .kramerMesnerMatrix _ _ => .invalid
  | _ => .invalid

end Designs
