import Gfp.Reference

/-!
# quadratic_forms: executable reference

An `n x n` symmetric matrix `A` over an odd prime field represents both the symmetric bilinear
form `B(x,y)=x^T A y` and the quadratic form `Q(x)=x^T A x`. Symmetric congruence elimination
produces a diagonal form. Its nonzero diagonal entries determine rank, discriminant square class,
type, Witt index, and the closed formula for the number of isotropic projective points.

The radical uses the same canonical nullspace basis as `gfp`. Type codes are 0 for hyperbolic,
1 for elliptic, and 2 for parabolic. Degenerate forms are classified through their nonsingular
quotient. Characteristic two and nonsymmetric inputs are invalid.
-/

namespace Quadratic_forms

open Lk

def entry (m : Mat) (i j : Nat) : Nat := (m.getD i []).getD j 0

/-- Swap two basis coordinates, applying the same swap to rows and columns. -/
def swapCoordinates (m : Mat) (i j : Nat) : Mat :=
  let rows := Gfp.swapRows m i j
  rows.map fun row =>
    (List.range row.length).map fun k =>
      if k = i then row.getD j 0 else if k = j then row.getD i 0 else row.getD k 0

/-- Replace basis vector `i` by `e_i + e_j`, by the corresponding row and column additions. -/
def addCoordinate (p : Nat) (m : Mat) (i j : Nat) : Mat :=
  let withRow := m.set i (List.zipWith (fun x y => (x + y) % p) (m.getD i []) (m.getD j []))
  (List.range withRow.length).map fun r =>
    let row := withRow.getD r []
    row.set i ((row.getD i 0 + row.getD j 0) % p)

def findDiagonal (m : Mat) (k : Nat) : Option Nat :=
  ((List.range m.length).drop k).find? fun i => entry m i i ≠ 0

def offDiagonalPairs (n k : Nat) : List (Nat × Nat) :=
  ((List.range n).drop k).flatMap fun i =>
    ((List.range n).drop (i + 1)).map fun j => (i, j)

def findOffDiagonal (m : Mat) (k : Nat) : Option (Nat × Nat) :=
  (offDiagonalPairs m.length k).find? fun ij => entry m ij.1 ij.2 ≠ 0

/-- Put a nonzero diagonal entry in position `k`. An off-diagonal entry `a` supplies the
nonzero diagonal entry `2a`, which is why the construction requires odd characteristic. -/
def preparePivot (p : Nat) (m : Mat) (k : Nat) : Option Mat :=
  match findDiagonal m k with
  | some i => some (swapCoordinates m k i)
  | none => match findOffDiagonal m k with
    | none => none
    | some (i, j) => some (swapCoordinates (addCoordinate p m i j) k i)

/-- Clear the row and column of a diagonal pivot by taking its Schur complement. -/
def clearPivot (p : Nat) (m : Mat) (k : Nat) : Mat :=
  let dInv := inv p (entry m k k)
  (List.range m.length).map fun i =>
    (List.range m.length).map fun j =>
      if i = k ∨ j = k then
        if i = k ∧ j = k then entry m k k else 0
      else if k < i ∧ k < j then
        let correction := entry m i k * dInv % p * entry m k j % p
        (entry m i j + p - correction) % p
      else entry m i j

/-- Nonzero entries of a diagonal form congruent to `m`. -/
def diagonalEntries (p : Nat) : Nat → Nat → Mat → Vec
  | 0, _, _ => []
  | fuel + 1, k, m =>
      if m.length ≤ k then []
      else match preparePivot p m k with
        | none => []
        | some prepared =>
            entry prepared k k :: diagonalEntries p fuel (k + 1) (clearPivot p prepared k)

def diagonal (p : Nat) (m : Mat) : Vec := diagonalEntries p m.length 0 m

def rank (p : Nat) (m : Mat) : Nat := (diagonal p m).length

/-- Discriminant of the nonsingular quotient, represented by the product of the nonzero diagonal
entries. Its square class does not depend on the congruence elimination choices. -/
def discriminant (p : Nat) (m : Mat) : Nat :=
  (diagonal p m).foldl (fun product d => product * d % p) (1 % p)

def squareSign (p a : Nat) : Nat := powMod p a 64 ((p - 1) / 2)

def hyperbolic (p r disc : Nat) : Bool :=
  let signed := if (r / 2) % 2 = 0 then disc else (p - disc) % p
  squareSign p signed = 1

/-- 0 hyperbolic, 1 elliptic, 2 parabolic. -/
def formType (p : Nat) (m : Mat) : Nat :=
  let ds := diagonal p m
  let r := ds.length
  let disc := ds.foldl (fun product d => product * d % p) (1 % p)
  if r % 2 = 1 then 2 else if hyperbolic p r disc then 0 else 1

/-- Number of hyperbolic planes in the nonsingular quotient. -/
def wittIndex (p : Nat) (m : Mat) : Nat :=
  let ds := diagonal p m
  let r := ds.length
  let disc := ds.foldl (fun product d => product * d % p) (1 % p)
  if r % 2 = 1 then r / 2 else if hyperbolic p r disc then r / 2 else r / 2 - 1

/-- Exact isometry, including the discriminant square class in odd rank. -/
def isIsometric (p : Nat) (m other : Mat) : Bool :=
  let left := diagonal p m
  let right := diagonal p other
  left.length = right.length &&
    squareSign p (left.foldl (fun product d => product * d % p) (1 % p)) =
      squareSign p (right.foldl (fun product d => product * d % p) (1 % p))

/-- `1 + p + ... + p^(n-1)`, the number of points in `PG(n-1,p)`. -/
def projectivePointCount (p : Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => 1 + p * projectivePointCount p n

/-- Number of projective zeros of `Q`. For even positive rank the correction has sign determined
by the hyperbolic or elliptic type; odd rank has the parabolic count. -/
def isotropicPointCount (p : Nat) (m : Mat) : Nat :=
  let n := m.length
  let ds := diagonal p m
  let r := ds.length
  if r = 0 then projectivePointCount p n
  else
    let base := projectivePointCount p (n - 1)
    if r % 2 = 1 then base
    else
      let disc := ds.foldl (fun product d => product * d % p) (1 % p)
      let correction := p ^ (n - r + r / 2 - 1)
      if hyperbolic p r disc then base + correction else base - correction

def squareSymmetric (m : Mat) : Bool :=
  let n := m.length
  m.all (fun row => row.length = n) &&
    (List.range n).all fun i =>
      (List.range n).all fun j => entry m i j = entry m j i

def supportedFamily : Family → Bool
  | .explicit _ _ | .symmetricMatrices _ _ => true
  | _ => false

inductive Op
  | formType | rank | radical | wittIndex
  | isIsometric (other : Mat)
  | isotropicPointCount

inductive Value
  | basis (rows : List Vec)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let p := f.p
  let ms := f.members
  if !supportedFamily f || p < 3 || p % 2 = 0 || !ms.all squareSymmetric then .invalid
  else match op with
    | .formType => reduceInt red ms (ms.map (formType p))
    | .rank => reduceInt red ms (ms.map (rank p))
    | .radical => reduceValues red (ms.map fun m => .basis (Gfp.nullspace p m))
    | .wittIndex => reduceInt red ms (ms.map (wittIndex p))
    | .isIsometric other =>
        if !squareSymmetric other || other.length ≠ (ms.headD []).length then .invalid
        else reduceBool red ms (ms.map fun m => isIsometric p m other)
    | .isotropicPointCount => reduceInt red ms (ms.map (isotropicPointCount p))

end Quadratic_forms
