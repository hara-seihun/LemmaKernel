import Lk.Reference

/-!
# Integral lattices: executable reference

A member is a positive-definite symmetric integral Gram matrix. Signed entries use the runtime's
ZigZag encoding inside `Lk.Mat`; this file decodes them before doing arithmetic. Enumeration is
plain structural recursion over a finite coordinate box. The box follows from
`x_i^2 <= (x^T G x) (G^-1)_{ii}` and is intentionally loose because oracle cases are small.
The generic backend uses Fincke-Pohst instead.
-/

namespace Lattices_small

open Lk

abbrev IMat := List (List Int)
abbrev IVec := List Int

def decodeGram (g : Mat) : IMat := g.map fun row => row.map decodeInt

def eraseAt (j : Nat) (row : List α) : List α :=
  ((List.range row.length).zip row).filterMap fun (i, x) => if i = j then none else some x

def minor (m : IMat) (i j : Nat) : IMat :=
  (((List.range m.length).zip m).filterMap fun (r, row) => if r = i then none else some row).map (eraseAt j)

/-- Laplace expansion with explicit fuel, so kernel reduction stays structural. -/
def detAux : Nat → IMat → Int
  | 0, _ => 0
  | _ + 1, [] => 1
  | fuel + 1, row :: rows =>
      ((List.range row.length).zip row).foldl (fun total pair =>
        let (j, a) := pair
        let sign : Int := if j % 2 = 0 then 1 else -1
        total + sign * a * detAux fuel (rows.map (eraseAt j))) 0

def det (m : IMat) : Int := detAux (m.length + 1) m

def square (m : IMat) : Bool :=
  let n := m.length
  n > 0 ∧ m.all (·.length = n)

def symmetric (m : IMat) : Bool :=
  (List.range m.length).all fun i =>
    (List.range m.length).all fun j => (m.getD i []).getD j 0 = (m.getD j []).getD i 0

def leading (m : IMat) (k : Nat) : IMat := (m.take k).map (·.take k)

def positiveDefinite (m : IMat) : Bool :=
  square m ∧ symmetric m ∧ ((List.range m.length).map (· + 1)).all fun k => det (leading m k) > 0

def validGram (g : Mat) : Bool := positiveDefinite (decodeGram g)

def qform (g : IMat) (x : IVec) : Int :=
  (List.range g.length).foldl (fun total i =>
    (List.range g.length).foldl (fun total j =>
      total + x.getD i 0 * (g.getD i []).getD j 0 * x.getD j 0) total) 0

/-- Integers `-b,...,-1,0,1,...,b`, in increasing order. -/
def signedRange (b : Nat) : List Int :=
  (List.range b).reverse.map (fun i => -Int.ofNat (i + 1)) ++ (List.range (b + 1)).map Int.ofNat

def vectorTuples (values : List Int) : Nat → List IVec
  | 0 => [[]]
  | n + 1 => values.flatMap fun x => (vectorTuples values n).map (x :: ·)

def vectorBox : List Nat → List IVec
  | [] => [[]]
  | b :: bs => (signedRange b).flatMap fun x => (vectorBox bs).map (x :: ·)

def isZero (x : IVec) : Bool := x.all (· = 0)

/-- Exact coordinate bounds from `x_i² det(G) ≤ bound * cofactor(i,i)`. -/
def coordinateBounds (g : IMat) (bound : Nat) : List Nat :=
  let determinant := (det g).natAbs
  (List.range g.length).map fun i => Nat.sqrt (bound * (det (minor g i i)).natAbs / determinant)

/-- Canonical short vectors. `vectorBox` emits signed lexicographic order. -/
def shortVectors (g : Mat) (bound : Nat) : List IVec :=
  let gi := decodeGram g
  (vectorBox (coordinateBounds gi bound)).filter fun x =>
    !isZero x ∧ qform gi x ≤ Int.ofNat bound

def minimum (g : Mat) : Nat :=
  let gi := decodeGram g
  let upper := ((List.range gi.length).map fun i => ((gi.getD i []).getD i 0).natAbs).foldl min
    (((gi.headD []).headD 1).natAbs)
  (shortVectors g upper).foldl (fun best x => min best (qform gi x).natAbs) upper

def kissingNumber (g : Mat) : Nat :=
  let m := minimum g
  let gi := decodeGram g
  ((shortVectors g m).filter fun x => qform gi x = Int.ofNat m).length

def thetaSeries (g : Mat) (bound : Nat) : List Nat :=
  let gi := decodeGram g
  let vectors := vectorBox (coordinateBounds gi bound)
  (List.range (bound + 1)).map fun k => (vectors.filter fun x => qform gi x = Int.ofNat k).length

def isUnimodular (g : Mat) : Bool := det (decodeGram g) = 1

def isEven (g : Mat) : Bool :=
  let gi := decodeGram g
  (List.range gi.length).all fun i => (gi.getD i []).getD i 0 % 2 = 0

inductive Op
  | minimum
  | kissingNumber
  | thetaSeries (bound : Nat)
  | isUnimodular
  | isEven
  | shortVectors (bound : Nat)

inductive Value
  | thetaSeries (coefficients : List Nat)
  | shortVectors (vectors : List IVec)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let ms := f.members
  if f.p ≠ gramTag ∨ !ms.all validGram then .invalid else
  match op with
  | .minimum => reduceInt red ms (ms.map minimum)
  | .kissingNumber => reduceInt red ms (ms.map kissingNumber)
  | .thetaSeries bound => reduceValues red (ms.map fun g => .thetaSeries (thetaSeries g bound))
  | .isUnimodular => reduceBool red ms (ms.map isUnimodular)
  | .isEven => reduceBool red ms (ms.map isEven)
  | .shortVectors bound => reduceValues red (ms.map fun g => .shortVectors (shortVectors g bound))

end Lattices_small
