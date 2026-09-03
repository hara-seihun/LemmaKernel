import Lk.Reference

/-!
# set_systems: executable reference

A member of `subsets_of(words(2, n), m)` is an `m`-set system on `0, ..., n-1`. Its rows are
binary incidence words. This file defines every operation by structural recursion so Lean's
kernel can evaluate it as the test oracle.
-/

namespace Set_systems

open Lk

/-- Apply `pred` to every unordered pair of distinct list entries. -/
def allPairs (pred : α → α → Bool) : List α → Bool
  | [] => true
  | x :: xs => xs.all (pred x) && allPairs pred xs

/-- Intersection of two incidence words. -/
def intersection (a b : Vec) : Vec :=
  List.zipWith (fun x y => if x = 1 && y = 1 then 1 else 0) a b

/-- Whether the set encoded by `a` is contained in the set encoded by `b`. -/
def isSubset (a b : Vec) : Bool :=
  (List.zipWith (fun x y => x ≠ 1 || y = 1) a b).all id

def isIntersecting (system : Mat) : Bool :=
  allPairs (fun a b => (intersection a b).any (· = 1)) system

def isAntichain (system : Mat) : Bool :=
  allPairs (fun a b => !isSubset a b && !isSubset b a) system

/-- A list of at least two petals is a sunflower when every pair has the same intersection. -/
def isSunflower : List Vec → Bool
  | a :: b :: rest =>
    let petals := a :: b :: rest
    let core := intersection a b
    allPairs (fun x y => intersection x y = core) petals
  | _ => false

def isSunflowerFree (k : Nat) (system : Mat) : Bool :=
  !((combos system k).any isSunflower)

def setSize (s : Vec) : Nat := (s.filter (· = 1)).length

def maxDegree (n : Nat) (system : Mat) : Nat :=
  (List.range n).foldl (fun best x => max best ((system.filter fun s => s.getD x 0 = 1).length)) 0

/-- Remove coordinate `i` from the represented set, leaving the incidence word length unchanged. -/
def removePoint (s : Vec) (i : Nat) : Vec :=
  (List.range s.length).map fun j => if i = j then 0 else s.getD j 0

/-- Deduplicate a list. Only its length is observed by `shadowSize`. -/
def unique [DecidableEq α] : List α → List α
  | [] => []
  | x :: xs => if x ∈ xs then unique xs else x :: unique xs

def lowerShadow (system : Mat) : List Vec :=
  unique (system.flatMap fun s => ((List.range s.length).filter fun i => s.getD i 0 = 1).map (removePoint s))

def shadowSize (system : Mat) : Nat := (lowerShadow system).length

/-- Binomial coefficient, written recursively so the kernel does not need Mathlib. -/
def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)

def isUniformOfSize (r : Nat) (system : Mat) : Bool := system.all fun s => setSize s = r

def isEkrExtremal (n : Nat) (system : Mat) : Bool :=
  let r := setSize (system.headD [])
  0 < r && 2 * r ≤ n && isUniformOfSize r system && isIntersecting system && system.length = choose (n - 1) (r - 1)

def isSpernerExtremal (n : Nat) (system : Mat) : Bool :=
  isAntichain system && system.length = choose n (n / 2)

inductive Op
  | isIntersecting
  | isAntichain
  | isSunflowerFree (k : Nat)
  | maxDegree
  | shadowSize
  | isEkrExtremal
  | isSpernerExtremal

abbrev Value := Unit

private def runOn (op : Op) (n : Nat) (systems : List Mat) (red : Red) : Result Value :=
  match op with
  | .isIntersecting => reduceBool red systems (systems.map isIntersecting)
  | .isAntichain => reduceBool red systems (systems.map isAntichain)
  | .isSunflowerFree k =>
    if k < 2 then .invalid else reduceBool red systems (systems.map (isSunflowerFree k))
  | .maxDegree => reduceInt red systems (systems.map (maxDegree n))
  | .shadowSize => reduceInt red systems (systems.map shadowSize)
  | .isEkrExtremal => reduceBool red systems (systems.map (isEkrExtremal n))
  | .isSpernerExtremal => reduceBool red systems (systems.map (isSpernerExtremal n))

/-- Only `subsets_of(words(2,n),m)` represents set systems for this module. -/
def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match f with
  | .subsetsOf (.words 2 n) _ => runOn op n f.members red
  | _ => .invalid

end Set_systems
