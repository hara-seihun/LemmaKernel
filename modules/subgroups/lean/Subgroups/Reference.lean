import Lk.Reference

/-!
# subgroups executable reference

A family member is a list of permutation generators. The reference closes the parent group, then
builds every subgroup by repeatedly adjoining one parent element to a known subgroup. A subgroup
is its complete lexicographically sorted element list, so equality and every output choice are
independent of the supplied generators.

Materialised results replace each permutation by its index in the parent's lexicographically
sorted element list. Conjugacy-class representatives are the least such lists in their classes.
The outer list uses the same lexicographic order.
-/

namespace Subgroups

open Lk

inductive Group
  | perms (gens : List Perm)
  | mats (p : Nat) (gens : List Mat)


def canonNats (xs : List Nat) : List Nat := (sortBy (· ≤ ·) xs).eraseDups

def isPerm (n : Nat) (g : Perm) : Bool := g.length = n ∧ canonNats g = List.range n

def validGenerators (n : Nat) (gens : List Perm) : Bool := n > 0 ∧ gens.all (isPerm n)

/-- The degree and generator rows of every group member. -/
def familyGenerators : Family → Option (Nat × List (List Perm))
  | f@(.subsets p dictionary _) =>
    let n := (dictionary.headD []).length
    if p = 0 ∧ !dictionary.isEmpty ∧ dictionary.all (isPerm n) then some (n, f.members) else none
  | f@(.explicit p batch) =>
    let n := ((batch.headD []).headD []).length
    if p = 0 ∧ n > 0 ∧ batch.all (fun gens => gens.all (isPerm n)) then some (n, f.members) else none
  | _ => none


def inversePerm (g : Perm) : Perm := (List.range g.length).map fun i => g.idxOf i

def generated (n : Nat) (gens : List Perm) : List Perm :=
  sortBy lexLe (saturate (fun g => gens.map (compose g)) fuel [identityPerm n])

/-- Boolean lexicographic order on complete subgroup element lists. -/
def subgroupLe (a b : List Perm) : Bool := lexLe a.flatten b.flatten

/-- Every subgroup, ordered by its complete element list. Starting from the trivial subgroup,
every extension adjoins one parent element and closes it with the known subgroup. -/
def allSubgroups (n : Nat) (gens : List Perm) : List (List Perm) :=
  let elements := generated n gens
  let trivial := [identityPerm n]
  sortBy subgroupLe <| saturate (fun h =>
    elements.filterMap fun x => if x ∈ h then none else some (generated n (x :: h))) fuel [trivial]


def conjugate (g : Perm) (h : Perm) : Perm := compose (compose (inversePerm g) h) g

def conjugateSubgroup (g : Perm) (h : List Perm) : List Perm :=
  sortBy lexLe (h.map (conjugate g))


def leastSubgroup : List (List Perm) → List Perm
  | [] => []
  | x :: xs => xs.foldl (fun best h => if subgroupLe h best then h else best) x


def canonicalConjugate (elements : List Perm) (h : List Perm) : List Perm :=
  leastSubgroup (elements.map fun g => conjugateSubgroup g h)


def conjugacyClasses (n : Nat) (gens : List Perm) : List (List Perm) :=
  let elements := generated n gens
  (allSubgroups n gens).filter fun h => canonicalConjugate elements h = h


def subset (a b : List Perm) : Bool := a.all (· ∈ b)

def isMaximal (whole : List Perm) (subgroups : List (List Perm)) (h : List Perm) : Bool :=
  h.length < whole.length ∧ !(subgroups.any fun k =>
    h.length < k.length ∧ k.length < whole.length ∧ subset h k)


def maximalSubgroups (n : Nat) (gens : List Perm) : List (List Perm) :=
  let elements := generated n gens
  let subgroups := allSubgroups n gens
  subgroups.filter fun h => isMaximal elements subgroups h ∧ canonicalConjugate elements h = h


def elementIndices (elements h : List Perm) : List Nat := h.map elements.idxOf

def subgroupIndices (elements : List Perm) (subgroups : List (List Perm)) : List (List Nat) :=
  subgroups.map (elementIndices elements)


def normalIn (n : Nat) (parentGens candidateGens : List Perm) : Bool :=
  let parent := generated n parentGens
  let candidate := generated n candidateGens
  subset candidate parent ∧ parent.all fun g => conjugateSubgroup g candidate = candidate


inductive Op
  | subgroupCount
  | conjugacyClasses
  | maximalSubgroups
  | isNormal (group : Group)

inductive Value
  | subgroups (lists : List (List Nat))
  deriving DecidableEq, Repr


def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match familyGenerators f with
  | none => .invalid
  | some (n, groups) =>
    let members := f.members
    match op with
    | .subgroupCount => reduceInt red members (groups.map fun gens => (allSubgroups n gens).length)
    | .conjugacyClasses =>
      reduceValues red (groups.map fun gens =>
        let elements := generated n gens
        .subgroups (subgroupIndices elements (conjugacyClasses n gens)))
    | .maximalSubgroups =>
      reduceValues red (groups.map fun gens =>
        let elements := generated n gens
        .subgroups (subgroupIndices elements (maximalSubgroups n gens)))
    | .isNormal group =>
      match group with
      | .perms parent =>
        if !validGenerators n parent then .invalid
        else reduceBool red members (groups.map (normalIn n parent))
      | .mats _ _ => .invalid

end Subgroups
