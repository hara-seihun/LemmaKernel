import Gfp.Reference

/-!
# projective_sets: executable reference

A point is a nonzero homogeneous coordinate row over `F_p`. Callers give a set as one member of
a `subsets` family whose dictionary contains distinct rows scaled so their first nonzero entry is
one. Hyperplanes are kernels of similarly normalised nonzero rows. Lines are the two-dimensional
subspaces in the runtime's canonical Grassmannian enumeration.

The definitions use structural list operations throughout so `decide +kernel` can evaluate the
small oracle cases.
-/

namespace Projective_sets

open Lk


def dot (p : Nat) (a b : Vec) : Nat :=
  (List.zipWith (fun x y => x * y) a b).foldl (· + ·) 0 % p

def normalise (p : Nat) (v : Vec) : Vec :=
  match v.find? (· ≠ 0) with
  | some a => scale p v (inv p a)
  | none => v

/-- One representative of every projective point, obtained by normalising all nonzero rows. -/
def projectivePoints (p d : Nat) : List Vec :=
  (((tuples p d).filter fun v => v.any (· ≠ 0)).map (normalise p)).eraseDups

def validPoint (p d : Nat) (v : Vec) : Bool :=
  v.length = d ∧ v.any (· ≠ 0) ∧ normalise p v = v

def validDictionary (p : Nat) (dictionary : List Vec) : Bool :=
  let d := (dictionary.headD []).length
  dictionary.all (validPoint p d) ∧ dictionary.eraseDups.length = dictionary.length

/-- Number of selected points in the hyperplane whose equation is `form · x = 0`. -/
def hyperplaneIntersection (p : Nat) (form : Vec) (points : Mat) : Nat :=
  (points.filter fun v => dot p form v = 0).length

/-- An arc in `PG(d-1,p)`: no hyperplane contains `d` selected points. -/
def isArc (p : Nat) (points : Mat) : Bool :=
  let d := (points.headD []).length
  (projectivePoints p d).all fun form => hyperplaneIntersection p form points < d

/-- Canonical bases of every projective line. -/
def projectiveLines (p d : Nat) : List Mat := grassmannianMembers p d 2

def lineIntersection (p : Nat) (line points : Mat) : Nat :=
  (points.filter fun v => Gfp.rank p (line ++ [v]) = 2).length

def lineIntersections (p : Nat) (points : Mat) : List Nat :=
  let d := (points.headD []).length
  (projectiveLines p d).map fun line => lineIntersection p line points

def isCap (p : Nat) (points : Mat) : Bool :=
  (lineIntersections p points).all (· ≤ 2)

def isBlockingSet (p : Nat) (points : Mat) : Bool :=
  let d := (points.headD []).length
  (projectivePoints p d).all fun form => hyperplaneIntersection p form points ≠ 0

def isHyperoval (p : Nat) (points : Mat) : Bool :=
  p = 2 ∧ (points.headD []).length = 3 ∧ points.length = 4 ∧ isCap p points

def isOvoid (p : Nat) (points : Mat) : Bool :=
  (points.headD []).length = 4 ∧ points.length = p * p + 1 ∧ isCap p points

def maxCollinear (p : Nat) (points : Mat) : Nat :=
  (lineIntersections p points).foldl max 0

def lineCount (p target : Nat) (points : Mat) : Nat :=
  ((lineIntersections p points).filter (· = target)).length

inductive Op
  | isArc | isCap | isBlockingSet | isHyperoval | isOvoid
  | maxCollinear | spannedRank | secantCount | tangentCount | passantCount

inductive Value
  | unit
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match f with
  | .subsets p dictionary _ =>
    if !validDictionary p dictionary then .invalid else
    let ms := f.members
    match op with
    | .isArc => reduceBool red ms (ms.map (isArc p))
    | .isCap => reduceBool red ms (ms.map (isCap p))
    | .isBlockingSet => reduceBool red ms (ms.map (isBlockingSet p))
    | .isHyperoval => reduceBool red ms (ms.map (isHyperoval p))
    | .isOvoid => reduceBool red ms (ms.map (isOvoid p))
    | .maxCollinear => reduceInt red ms (ms.map (maxCollinear p))
    | .spannedRank => reduceInt red ms (ms.map (Gfp.rank p))
    | .secantCount => reduceInt red ms (ms.map (lineCount p 2))
    | .tangentCount => reduceInt red ms (ms.map (lineCount p 1))
    | .passantCount => reduceInt red ms (ms.map (lineCount p 0))
  | _ => .invalid

end Projective_sets
