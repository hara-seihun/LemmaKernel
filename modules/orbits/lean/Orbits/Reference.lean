import Gfp.Reference

/-!
# orbits: executable reference

A group given by generators acts on a family by permuting its member indices; the canonical
representative of an orbit is the member with the least index, in the order `Lk.Reference`
defines for the family. Orbits are computed by saturating under the generators, groups by
saturating under multiplication. Everything is plain structural recursion so that
`decide +kernel` can evaluate it; a kernel answer is correct when
`example : run op family red = answer := by decide` is accepted.

A permutation of `n` points is a list of `n` images. Generators of a matrix group are `n x n`
matrices over `F_p`; they act on the right (`M ↦ M A`), on Grassmannian members up to rref.
-/

namespace Orbits

open Lk

def matElements (p : Nat) (gens : List Mat) : List Mat :=
  let n := (gens.headD []).length
  saturate (fun a => gens.map (matmul p a)) fuel [identity n]

/-! ## Actions -/

/-- A permutation of dictionary positions acting on a subset given by its sorted positions, each
position written as a one-entry row so that subsets and matrices share the `Mat` type. -/
def actPerm (g : Perm) (m : Mat) : Mat :=
  (sortBy (· ≤ ·) (m.map fun row => g.getD (row.headD 0) 0)).map ([·])

/-- Positions of a `subsets` member, in the family's order. -/
def subsetKeys (d : List Vec) (k : Nat) : List Mat := (combos (List.range d.length) k).map (·.map ([·]))

inductive Group
  | perms (gens : List Perm)
  | mats (p : Nat) (gens : List Mat)

def Group.size : Group → Nat
  | .perms gens => gens.length
  | .mats _ gens => gens.length

def Group.order : Group → Nat
  | .perms gens => (permElements gens).length
  | .mats p gens => (matElements p gens).length

/-- Whether the group's generators act on the family at all: right shapes, and on a
Grassmannian every generator invertible. -/
def actsOn : Group → Family → Bool
  | .perms gens, .subsets _ d _ => gens.all fun g => g.length = d.length
  | .mats p gens, .grassmannian p' n _ => p = p' ∧ gens.all fun a => a.length = n ∧ a.all (·.length = n) ∧ Gfp.rank p a = n
  | .mats p gens, .allMatrices p' _ n => p = p' ∧ gens.all fun a => a.length = n ∧ a.all (·.length = n)
  | _, _ => false

/-- What the group permutes: subset members as their dictionary positions (so that duplicate
dictionary rows remain distinct members), matrix family members as themselves. -/
def keys : Group → Family → List Mat
  | .perms _, .subsets _ d k => subsetKeys d k
  | _, f => f.members

/-- Image of key `m` under generator `k`, canonicalised within the family. -/
def image (g : Group) (f : Family) (k : Nat) (m : Mat) : Mat :=
  match g, f with
  | .perms gens, .subsets _ _ _ => actPerm (gens.getD k []) m
  | .mats p gens, .grassmannian _ _ _ => Gfp.rref p (matmul p m (gens.getD k []))
  | .mats p gens, _ => matmul p m (gens.getD k [])
  | _, _ => m

/-- Indices of the orbit of member `i`. -/
def orbit (g : Group) (f : Family) (i : Nat) : List Nat :=
  let ks := keys g f
  saturate (fun j => (List.range g.size).map fun k => ks.idxOf (image g f k (ks.getD j []))) fuel [i]

def leastOf (xs : List Nat) : Nat := xs.foldl min (xs.headD 0)

/-- Scale a nonzero vector so its first nonzero entry is 1. -/
def normalise (p : Nat) (v : Vec) : Vec :=
  match v.find? (· ≠ 0) with
  | some a => scale p v (inv p a)
  | none => v

/-! ## Operations and values -/

inductive Op
  | isCanonical (g : Group) | canonicalIndex (g : Group) | orbitSize (g : Group) | stabilizerOrder (g : Group)
  | fixedPoints (on : Family)
  | projectiveAction (points : List Vec)

/-- One materialised output per member: `projective_action` yields a permutation. -/
inductive Value
  | perm (g : Perm)
  deriving DecidableEq, Repr

def orbitOp (g : Group) (f : Family) (red : Red) (value : List Nat → Nat) (bool : Bool) : Result Value :=
  if !actsOn g f then .invalid else
  let ms := f.members
  let orbits := (List.range ms.length).map fun i => orbit g f i
  if bool then reduceBool red ms (List.zipWith (fun o i => decide (leastOf o = i)) orbits (List.range ms.length))
  else reduceInt red (orbits.map value)

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match op, f with
  | .isCanonical g, _ => orbitOp g f red (fun _ => 0) true
  | .canonicalIndex g, _ => orbitOp g f red leastOf false
  | .orbitSize g, _ => orbitOp g f red (·.length) false
  | .stabilizerOrder g, _ => orbitOp g f red (fun o => g.order / o.length) false
  | .fixedPoints on, .groupElements gens =>
    match on with
    | .subsets _ d on_k =>
      if !gens.all (·.length = d.length) then .invalid else
      let ks := subsetKeys d on_k
      reduceInt red ((permElements gens).map fun g => (ks.filter fun m => actPerm g m = m).length)
    | _ => .invalid
  | .projectiveAction pts, .explicit p batch =>
    let images := batch.map fun a => pts.map fun v => normalise p ((matmul p [v] a).headD [])
    if images.all (·.all (· ∈ pts)) then reduceValues red (images.map fun g => .perm (g.map pts.idxOf))
    else .invalid
  | _, _ => .invalid

end Orbits
