import Gfp.Reference

/-!
# linear_codes: executable reference

A matrix denotes the linear code spanned by its rows over `F_p`. The reference first takes the
nonzero rows of its canonical rref, so duplicate and dependent generator rows never duplicate
codewords. Codewords then come from every coefficient tuple in lexicographic order. The native
backend uses a different enumeration order internally, but every operation here depends only on
counts, minima, maxima, or row spaces.
-/

namespace Linear_codes

open Lk

/-- The unique nonzero rref basis of the row span. -/
def codeBasis (p : Nat) (m : Mat) : Mat :=
  let g := Gfp.gaussJordan p m
  g.rows.take g.r

/-- A linear combination of `basis`, with an explicit length for the zero code. -/
def linearCombination (p n : Nat) (coeff : Vec) (basis : Mat) : Vec :=
  (List.range n).map fun j =>
    (List.zipWith (fun a row => a * row.getD j 0) coeff basis).foldl (· + ·) 0 % p

/-- Every distinct codeword. -/
def codewords (p : Nat) (m : Mat) : List Vec :=
  let n := (m.headD []).length
  let basis := codeBasis p m
  (tuples p basis.length).map fun coeff => linearCombination p n coeff basis

def hammingWeight (v : Vec) : Nat := (v.filter (· ≠ 0)).length

def hammingDistance : Vec → Vec → Nat
  | x :: xs, y :: ys => (if x = y then 0 else 1) + hammingDistance xs ys
  | _, _ => 0

/-- `[A_0, ..., A_n]`, where `A_w` counts codewords of weight `w`. -/
def weightEnumerator (p : Nat) (m : Mat) : List Nat :=
  let n := (m.headD []).length
  let weights := (codewords p m).map hammingWeight
  (List.range (n + 1)).map fun w => (weights.filter (· = w)).length

/-- The zero code has minimum distance zero. -/
def minimumDistance (p : Nat) (m : Mat) : Nat :=
  match ((codewords p m).map hammingWeight).filter (· ≠ 0) with
  | [] => 0
  | w :: ws => ws.foldl min w

/-- Canonical basis of the Euclidean dual. -/
def dual (p : Nat) (m : Mat) : Mat := Gfp.nullspace p m

def isSelfDual (p : Nat) (m : Mat) : Bool := codeBasis p m = codeBasis p (dual p m)

def nearestDistance (v : Vec) (code : List Vec) : Nat :=
  match code.map (hammingDistance v) with
  | [] => v.length
  | d :: ds => ds.foldl min d

def coveringRadius (p : Nat) (m : Mat) : Nat :=
  let n := (m.headD []).length
  let code := codewords p m
  (tuples p n).foldl (fun r v => max r (nearestDistance v code)) 0

def isMds (p : Nat) (m : Mat) : Bool :=
  let n := (m.headD []).length
  let k := (codeBasis p m).length
  k ≠ 0 ∧ minimumDistance p m = n - k + 1

/-- Every insertion position for `x`, used to enumerate `S_n` structurally. -/
def insertions (x : Nat) : List Nat → List (List Nat)
  | [] => [[x]]
  | y :: ys => (x :: y :: ys) :: (insertions x ys).map (y :: ·)

def permutations : List Nat → List (List Nat)
  | [] => [[]]
  | x :: xs => (permutations xs).flatMap (insertions x)

def permuteCoordinates (v perm : Vec) : Vec := perm.map fun j => v.getD j 0

def isAutomorphism (p : Nat) (basis : Mat) (perm : Vec) : Bool :=
  codeBasis p (basis.map fun row => permuteCoordinates row perm) = basis

/-- Order of the coordinate-permutation automorphism group. -/
def autOrder (p : Nat) (m : Mat) : Nat :=
  let basis := codeBasis p m
  ((permutations (List.range (m.headD []).length)).filter (isAutomorphism p basis)).length

inductive Op
  | minimumDistance | weightEnumerator | dual | isSelfDual | coveringRadius | isMds | autOrder

inductive Value
  | weightEnumerator (coefficients : List Nat)
  | basis (rows : List Vec)
  deriving DecidableEq, Repr

def supportedFamily : Family → Bool
  | .explicit _ _ | .subsets _ _ _ | .subsetsOf _ _ | .grassmannian _ _ _
  | .allMatrices _ _ _ | .symmetricMatrices _ _ | .transform _ _ | .stack _ _ => true
  | _ => false

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  if !supportedFamily f then .invalid else
  let p := f.p
  let ms := f.members
  match op with
  | .minimumDistance => reduceInt red ms (ms.map (minimumDistance p))
  | .weightEnumerator => reduceValues red (ms.map fun m => .weightEnumerator (weightEnumerator p m))
  | .dual => reduceValues red (ms.map fun m => .basis (dual p m))
  | .isSelfDual => reduceBool red ms (ms.map (isSelfDual p))
  | .coveringRadius => reduceInt red ms (ms.map (coveringRadius p))
  | .isMds => reduceBool red ms (ms.map (isMds p))
  | .autOrder => reduceInt red ms (ms.map (autOrder p))

end Linear_codes
