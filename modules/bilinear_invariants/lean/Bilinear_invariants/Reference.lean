import Gfp.Reference

/-!
# bilinear_invariants: executable reference

A valid member is the Gram matrix of a symmetric or alternating bilinear form over a prime field.
The congruence label uses the standard finite-field classification. Alternating forms are classified
by rank. Over odd fields, symmetric forms are classified by rank and the square class of the
nondegenerate quotient's determinant. Over `F_2`, a symmetric form is classified by rank and
whether it is alternating.
-/

namespace Bilinear_invariants

open Lk

/-! ## Matrix and field predicates -/

def isPrime (p : Nat) : Bool :=
  2 ≤ p && (List.range (p - 2)).all fun i => p % (i + 2) ≠ 0

def isSymmetric (a : Mat) : Bool :=
  let n := a.length
  (List.range n).all fun i => (List.range n).all fun j =>
    (a.getD i []).getD j 0 = (a.getD j []).getD i 0

def isAlternating (p : Nat) (a : Mat) : Bool :=
  let n := a.length
  (List.range n).all fun i =>
    (a.getD i []).getD i 0 = 0 &&
      (List.range n).all fun j => ((a.getD i []).getD j 0 + (a.getD j []).getD i 0) % p = 0

def validForm (p : Nat) (a : Mat) : Bool := isSymmetric a || isAlternating p a

/-! ## Rank, determinant, and square classes -/

def minor (a : Mat) (j : Nat) : Mat :=
  a.drop 1 |>.map fun row =>
    (List.range row.length).filterMap fun i => if i = j then none else some (row.getD i 0)

def detAux (p : Nat) : Nat → Mat → Nat
  | 0, _ => 1 % p
  | n + 1, a =>
    let row := a.headD []
    (List.range row.length).foldl (fun total j =>
      let term := (row.getD j 0 * detAux p n (minor a j)) % p
      if j % 2 = 0 then (total + term) % p else (total + p - term) % p) 0

def determinant (p : Nat) (a : Mat) : Nat := detAux p a.length a

def squareClass (p x : Nat) : Nat :=
  if x % p = 0 then 0 else if powMod p (x % p) 64 ((p - 1) / 2) = 1 then 1 else 2

def principalSubmatrix (a : Mat) (indices : List Nat) : Mat :=
  indices.map fun i => indices.map fun j => (a.getD i []).getD j 0

/-- Determinant of the first full-rank principal submatrix. Its square class does not depend on
which full-rank principal submatrix is chosen. -/
def discriminant (p : Nat) (a : Mat) : Nat :=
  let r := Gfp.rank p a
  if r = 0 then 0 else
    ((combos (List.range a.length) r).map fun indices =>
      determinant p (principalSubmatrix a indices)).find? (· ≠ 0) |>.getD 0

def discriminantClass (p : Nat) (a : Mat) : Nat := squareClass p (discriminant p a)

def leastNonsquare (p : Nat) : Nat :=
  (List.range p).find? (fun x => squareClass p x = 2) |>.getD 1

/-! ## Canonical congruence representative -/

def alternatingLabel (p n rank : Nat) : Mat :=
  (List.range n).map fun i => (List.range n).map fun j =>
    if i < rank ∧ i % 2 = 0 ∧ j = i + 1 then 1
    else if j < rank ∧ j % 2 = 0 ∧ i = j + 1 then (p - 1) % p
    else 0

def diagonalLabel (n rank last : Nat) : Mat :=
  (List.range n).map fun i => (List.range n).map fun j =>
    if i = j ∧ i < rank then if i + 1 = rank then last else 1 else 0

def congruenceLabel (p : Nat) (a : Mat) : Mat :=
  let n := a.length
  let r := Gfp.rank p a
  if isAlternating p a then alternatingLabel p n r
  else if p = 2 then diagonalLabel n r 1
  else
    let last := if discriminantClass p a = 2 then leastNonsquare p else 1
    diagonalLabel n r last

/-! ## Operations -/

inductive Op
  | rank | radicalDimension | determinant | determinantClass | discriminantClass
  | isNondegenerate | isAlternating | congruenceLabel

inductive Value
  | matrix (m : Mat)
  deriving DecidableEq, Repr

def supportedFamily : Family → Bool
  | .explicit _ _ | .symmetricMatrices _ _ | .alternatingMatrices _ _ => true
  | _ => false

def run (op : Op) (family : Family) (red : Red) : Result Value :=
  let p := family.p
  let ms := family.members
  let n := (ms.headD []).length
  if !supportedFamily family || !isPrime p || n = 0 ||
      ms.any (fun a => a.length ≠ n || a.any (fun row => row.length ≠ n)) ||
      ms.any (fun a => !validForm p a) then .invalid
  else match op with
  | .rank => reduceInt red ms (ms.map (Gfp.rank p))
  | .radicalDimension => reduceInt red ms (ms.map fun a => n - Gfp.rank p a)
  | .determinant => reduceInt red ms (ms.map (determinant p))
  | .determinantClass => reduceInt red ms (ms.map fun a => squareClass p (determinant p a))
  | .discriminantClass => reduceInt red ms (ms.map (discriminantClass p))
  | .isNondegenerate => reduceBool red ms (ms.map fun a => Gfp.rank p a = n)
  | .isAlternating => reduceBool red ms (ms.map (isAlternating p))
  | .congruenceLabel => reduceValues red (ms.map fun a => .matrix (congruenceLabel p a))

end Bilinear_invariants
