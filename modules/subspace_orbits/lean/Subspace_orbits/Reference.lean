import Gfp.Reference

/-!
# subspace_orbits: executable reference

A matrix group acts on each member's row space on the right. Every intermediate subspace is
stored as its reduced row echelon basis with zero rows removed. Among an orbit, the canonical
subspace is the one with the least index in `grassmannianMembers p n rank`.

`projective = 0` closes the supplied matrices in GL. `projective = 1` closes them after scaling
each matrix so its first nonzero entry is one. This is the image in PGL. The runtime only has
prime fields, whose field automorphism group is trivial, so the PGammaL action is the PGL action.
-/

namespace Subspace_orbits

open Lk

inductive Group
  | perms (gens : List Perm)
  | mats (p : Nat) (gens : List Mat)

/-- A family made from one Grassmannian by any sequence of `transform` and `stack`. -/
def grassmannianDerived : Family → Bool
  | .grassmannian _ _ _ => true
  | .transform inner _ | .stack inner _ => grassmannianDerived inner
  | _ => false

/-- Number of ambient coordinates in each member. -/
def familyCols : Family → Nat
  | .grassmannian _ n _ => n
  | .transform _ c => (c.headD []).length
  | .stack inner _ => familyCols inner
  | f => (((f.members).headD []).headD []).length

/-- Canonical rref basis of a row space. -/
def subspaceRref (p : Nat) (m : Mat) : Mat :=
  (Gfp.rref p m).filter fun row => row.any (· ≠ 0)

/-- Scale a nonzero matrix so its first nonzero entry is one. -/
def normaliseMatrix (p : Nat) (m : Mat) : Mat :=
  match m.flatten.find? (· ≠ 0) with
  | some a => m.map fun row => scale p row (inv p a)
  | none => m

/-- Matrix closure. Projective closure identifies nonzero scalar multiples. -/
def matrixElements (p projective : Nat) (gens : List Mat) : List Mat :=
  let n := (gens.headD []).length
  saturate (fun a => gens.map fun g =>
    let b := matmul p a g
    if projective = 1 then normaliseMatrix p b else b) fuel [identity n]

/-- The canonical Grassmannian index of a subspace of `F_p^n`. -/
def subspaceIndex (p n : Nat) (m : Mat) : Nat :=
  (grassmannianMembers p n (Gfp.rank p m)).idxOf (subspaceRref p m)

/-- Orbit under the generator matrices, with no repeated row spaces. -/
def orbit (p : Nat) (gens : List Mat) (m : Mat) : List Mat :=
  saturate (fun w => gens.map fun a => subspaceRref p (matmul p w a)) fuel [subspaceRref p m]

/-- Least Grassmannian index represented in an orbit. -/
def leastOrbitIndex (p n : Nat) (o : List Mat) : Nat :=
  (o.map (subspaceIndex p n)).foldl min ((o.head?.map (subspaceIndex p n)).getD 0)

/-- The group argument has the right field and shape, and every generator is invertible. -/
def actsOn (g : Group) (projective : Nat) (f : Family) : Bool :=
  match g with
  | .perms _ => false
  | .mats p gens =>
    let n := familyCols f
    decide (projective ≤ 1) && grassmannianDerived f && decide (p = f.p) && !gens.isEmpty &&
      gens.all fun a => decide (a.length = n) && a.all (fun row => decide (row.length = n)) &&
        decide (Gfp.rank p a = n)

inductive Op
  | isCanonical (group : Group) (projective : Nat)
  | canonicalIndex (group : Group) (projective : Nat)
  | orbitSize (group : Group) (projective : Nat)
  | stabilizerOrder (group : Group) (projective : Nat)

inductive Value
  | unused
  deriving DecidableEq, Repr

inductive IntOrbitOp
  | canonicalIndex | orbitSize | stabilizerOrder


def runOrbitInt (which : IntOrbitOp) (g : Group) (projective : Nat) (f : Family) (red : Red) : Result Value :=
  if !actsOn g projective f then .invalid else
  match g with
  | .perms _ => .invalid
  | .mats p gens =>
    let n := familyCols f
    let ms := f.members
    let os := ms.map (orbit p gens)
    let order := (matrixElements p projective gens).length
    let values := os.map fun o =>
      match which with
      | .canonicalIndex => leastOrbitIndex p n o
      | .orbitSize => o.length
      | .stabilizerOrder => order / o.length
    reduceInt red ms values


def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match op with
  | .isCanonical g projective =>
    if !actsOn g projective f then .invalid else
    match g with
    | .perms _ => .invalid
    | .mats p gens =>
      let n := familyCols f
      let ms := f.members
      reduceBool red ms (ms.map fun m =>
        subspaceIndex p n m = leastOrbitIndex p n (orbit p gens m))
  | .canonicalIndex g projective => runOrbitInt .canonicalIndex g projective f red
  | .orbitSize g projective => runOrbitInt .orbitSize g projective f red
  | .stabilizerOrder g projective => runOrbitInt .stabilizerOrder g projective f red

end Subspace_orbits
