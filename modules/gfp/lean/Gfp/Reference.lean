import Lk.Reference

/-!
# gfp: executable reference

The meaning of every gfp operation, written as plain structural recursion so that Lean's kernel
can evaluate it. Families, reductions and the result shape come from `Lk.Reference`. This is the
oracle: a kernel answer is correct when `example : run op family red = answer := by decide` is
accepted. The canonical choice in every output (which nullspace basis, which pivots) is *defined*
here; backends reproduce it bit for bit.

Matrices are lists of rows of naturals, every entry already reduced modulo the prime `p`.
-/

namespace Gfp

open Lk


/-! ## Gauss-Jordan elimination

`gaussJordan p rows` returns `(R, T, pivots)` with `T * rows = R`, `R` in reduced row echelon
form, `T` a product of the elementary operations, and `pivots` the pivot columns in increasing
order. Pivoting picks the first row at or below the current one with a nonzero entry. -/

structure GJ where
  rows : Mat
  t : Mat
  r : Nat
  pivots : List Nat

/-- Index of the first row `i ≥ r` with a nonzero entry in column `c`. -/
def findPivot (rows : Mat) (c r : Nat) : Option Nat :=
  ((List.range rows.length).drop r).find? fun i => (rows.getD i []).getD c 0 ≠ 0

def swapRows (m : Mat) (i j : Nat) : Mat :=
  (List.range m.length).map fun k =>
    if k = i then m.getD j [] else if k = j then m.getD i [] else m.getD k []

def gjColumn (p : Nat) (s : GJ) (c : Nat) : GJ :=
  match findPivot s.rows c s.r with
  | none => s
  | some i =>
    let rows := swapRows s.rows i s.r
    let t := swapRows s.t i s.r
    let f := inv p ((rows.getD s.r []).getD c 0)
    let prow := scale p (rows.getD s.r []) f
    let ptrow := scale p (t.getD s.r []) f
    let factor := fun k => (rows.getD k []).getD c 0
    { rows := (List.range rows.length).map fun k =>
        if k = s.r then prow else subMul p (rows.getD k []) prow (factor k)
      t := (List.range t.length).map fun k =>
        if k = s.r then ptrow else subMul p (t.getD k []) ptrow (factor k)
      r := s.r + 1
      pivots := s.pivots ++ [c] }

def gaussJordan (p : Nat) (m : Mat) : GJ :=
  let cols := (m.headD []).length
  (List.range cols).foldl (gjColumn p) { rows := m, t := identity m.length, r := 0, pivots := [] }

def rref (p : Nat) (m : Mat) : Mat := (gaussJordan p m).rows
def pivots (p : Nat) (m : Mat) : List Nat := (gaussJordan p m).pivots
def rank (p : Nat) (m : Mat) : Nat := (pivots p m).length
def nullity (p : Nat) (m : Mat) : Nat := (m.headD []).length - rank p m

/-- Canonical nullspace basis: one vector per free column `f` in increasing order, with
`x_f = 1`, `x_{pivot_i} = -R[i][f]`, and zeros elsewhere. -/
def nullspace (p : Nat) (m : Mat) : List Vec :=
  let g := gaussJordan p m
  let n := (m.headD []).length
  ((List.range n).filter fun f => f ∉ g.pivots).map fun f =>
    (List.range n).map fun j =>
      if j = f then 1
      else match g.pivots.idxOf? j with
        | some i => (p - (g.rows.getD i []).getD f 0) % p
        | none => 0

def inSpan (p : Nat) (m : Mat) (target : Vec) : Bool := rank p m = rank p (m ++ [target])

/-- `x` with `m xᵀ = rhsᵀ`, free coordinates zero; `none` when inconsistent. -/
def solve (p : Nat) (m : Mat) (rhs : Vec) : Option Vec :=
  let n := (m.headD []).length
  let g := gaussJordan p (List.zipWith (fun row b => row ++ [b]) m rhs)
  if n ∈ g.pivots then none
  else some ((List.range n).map fun j =>
    match g.pivots.idxOf? j with
    | some i => (g.rows.getD i []).getD n 0
    | none => 0)

def inverse (p : Nat) (m : Mat) : Option Mat :=
  let n := m.length
  let g := gaussJordan p (List.zipWith (· ++ ·) m (identity n))
  if g.pivots = List.range n then some (g.rows.map fun row => row.drop n) else none

/-- `(R, T)` with `T * m = R`. -/
def witness (p : Nat) (m : Mat) : Mat × Mat := let g := gaussJordan p m; (g.rows, g.t)

/-! ## Operations and values -/

inductive Op
  | rank | nullity | fullRowRank | fullColRank
  | inSpan (target : Vec)
  | rref | nullspace
  | solve (rhs : List Vec) | inverse | rrefWitness

/-- One materialised output per member. -/
inductive Value
  | matrix (m : Mat)
  | basis (b : List Vec)
  | solution (x : Option Vec)
  | inverse (m : Option Mat)
  | witness (w : Mat × Mat)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let p := f.p
  let ms := f.members
  let rows := (ms.headD []).length
  let cols := ((ms.headD []).headD []).length
  match op, red with
  | .rank, _ => reduceInt red ms (ms.map (rank p))
  | .nullity, _ => reduceInt red ms (ms.map (nullity p))
  | .fullRowRank, _ => reduceBool red ms (ms.map fun m => rank p m = rows)
  | .fullColRank, _ => reduceBool red ms (ms.map fun m => rank p m = cols)
  | .inSpan t, _ => reduceBool red ms (ms.map fun m => inSpan p m t)
  | .rref, _ => reduceValues red (ms.map fun m => .matrix (rref p m))
  | .nullspace, _ => reduceValues red (ms.map fun m => .basis (nullspace p m))
  | .solve rhs, _ => match f with
    | .explicit _ _ => reduceValues red (List.zipWith (fun m b => .solution (solve p m b)) ms rhs)
    | _ => .invalid
  | .inverse, _ => match f with
    | .explicit _ _ => reduceValues red (ms.map fun m => .inverse (inverse p m))
    | _ => .invalid
  | .rrefWitness, _ => match f with
    | .explicit _ _ => reduceValues red (ms.map fun m => .witness (witness p m))
    | _ => .invalid

end Gfp
