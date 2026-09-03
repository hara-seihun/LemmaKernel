import Lk.Reference

/-!
# Boolean functions: executable reference

A member is a binary truth table. Its columns are inputs `0, ..., 2^n - 1` in numeric order,
and row `i` is coordinate function `i`. Variable `x_i` is bit `i` of the column index.
The definitions use bounded structural recursion so Lean's kernel can evaluate the oracle cases.
-/

namespace Boolean_functions

open Lk

def logTwo : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, k => if k ≤ 1 then 0 else 1 + logTwo fuel (k / 2)

def inputDimension (size : Nat) : Nat := logTwo 64 size

def validSize (size : Nat) : Bool := 0 < size && 2 ^ inputDimension size = size

def xorNat : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | fuel + 1, a, b => (a % 2 + b % 2) % 2 + 2 * xorNat fuel (a / 2) (b / 2)

def bitDotParity : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | fuel + 1, a, b =>
      (a % 2 * (b % 2) + bitDotParity fuel (a / 2) (b / 2)) % 2

def popcount : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, x => x % 2 + popcount fuel (x / 2)

def isSubmask : Nat → Nat → Nat → Bool
  | 0, _, _ => true
  | fuel + 1, x, mask =>
      x % 2 ≤ mask % 2 && isSubmask fuel (x / 2) (mask / 2)

/-! ## Walsh transform and scalar invariants -/

def walshCoefficient (row : Vec) (u : Nat) : Int :=
  (List.range row.length).foldl (fun acc x =>
    if (row.getD x 0 + bitDotParity 64 u x) % 2 = 0 then acc + 1 else acc - 1) (0 : Int)

def walsh (row : Vec) : List Int :=
  (List.range row.length).map (walshCoefficient row)

def largestWalshMagnitude (row : Vec) : Nat :=
  (walsh row).foldl (fun best z => max best z.natAbs) 0

def nonlinearity (row : Vec) : Nat :=
  (row.length - largestWalshMagnitude row) / 2

def isBent (row : Vec) : Bool :=
  let n := inputDimension row.length
  n % 2 = 0 && (walsh row).all fun z => z.natAbs = 2 ^ (n / 2)

/-! ## Algebraic normal form -/

def anfCoefficient (row : Vec) (mask : Nat) : Nat :=
  (List.range row.length).foldl (fun acc x =>
    if isSubmask 64 x mask then (acc + row.getD x 0) % 2 else acc) 0

def coordinateDegree (row : Vec) : Nat :=
  (List.range row.length).foldl (fun degree mask =>
    if anfCoefficient row mask = 1 then max degree (popcount 64 mask) else degree) 0

def algebraicDegree (table : Mat) : Nat :=
  table.foldl (fun degree row => max degree (coordinateDegree row)) 0

/-! ## Differential uniformity -/

def derivativeValue (table : Mat) (a x : Nat) : Nat :=
  table.foldl (fun value row =>
    value * 2 + (row.getD x 0 + row.getD (xorNat 64 x a) 0) % 2) 0

def derivativeValues (table : Mat) (a : Nat) : List Nat :=
  let size := (table.headD []).length
  (List.range size).map (derivativeValue table a)

def hasMultiplicityAtMostTwo (xs : List Nat) : Bool :=
  xs.all fun y => (xs.filter (· = y)).length ≤ 2

def isAPN (table : Mat) : Bool :=
  let size := (table.headD []).length
  (List.range (size - 1)).all fun i => hasMultiplicityAtMostTwo (derivativeValues table (i + 1))

/-! ## Affine-equivalence canonical form -/

/-- All linear permutations of `F_2^n`, represented by the images of `0, ..., 2^n - 1`.
At each step, a vector outside the current span becomes the image of the next basis vector. -/
def linearOrders (size : Nat) : Nat → List Vec
  | 0 => [[0]]
  | steps + 1 =>
      (linearOrders size steps).flatMap fun span =>
        ((List.range size).filter fun v => v ∉ span).map fun v =>
          span ++ span.map (xorNat 64 v)

/-- All affine permutations `x ↦ A x + b` of the truth-table columns. -/
def affineOrders (size : Nat) : List Vec :=
  let linear := linearOrders size (inputDimension size)
  (List.range size).flatMap fun b => linear.map fun order => order.map (xorNat 64 b)

def reindexRow (row order : Vec) : Vec := order.map fun x => row.getD x 0

/-- The row-major lexicographically least truth table obtained by an invertible affine change of
input variables. This is the canonical representative of domain-affine equivalence. -/
def affineClass (row : Vec) : Vec :=
  let candidates := (affineOrders row.length).map (reindexRow row)
  candidates.foldl (fun best candidate => if lexLe candidate best then candidate else best)
    (candidates.headD row)

/-! ## Operations and values -/

inductive Op
  | nonlinearity | algebraicDegree | walshSpectrum | isBent | isApn | affineClass

inductive Value
  | spectrum (values : List (List Int))
  | matrix (value : Mat)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let ms := f.members
  let rows := (ms.headD []).length
  let cols := ((ms.headD []).headD []).length
  if f.p != 2 || !validSize cols then .invalid else
  match op with
  | .nonlinearity =>
      if rows != 1 then .invalid
      else reduceInt red ms (ms.map fun m => nonlinearity (m.headD []))
  | .algebraicDegree => reduceInt red ms (ms.map algebraicDegree)
  | .walshSpectrum =>
      if rows != 1 then .invalid
      else reduceValues red (ms.map fun m => .spectrum [walsh (m.headD [])])
  | .isBent =>
      if rows != 1 then .invalid
      else reduceBool red ms (ms.map fun m => isBent (m.headD []))
  | .isApn => reduceBool red ms (ms.map isAPN)
  | .affineClass =>
      if rows != 1 then .invalid
      else reduceValues red (ms.map fun m => .matrix [affineClass (m.headD [])])

end Boolean_functions
