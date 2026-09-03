import Lk.Reference

/-!
# gfq executable reference

The family tag is `q = p^d`. A field element in `0..q-1` is the base-`p`, little-endian
coefficient vector of a polynomial of degree below `d`. `modulus = [c₀, ..., c_{d-1}, 1]`
presents the field as `F_p[x] / (modulus)`.

Every function uses structural recursion and finite list folds so Lean's kernel can evaluate it.
-/

namespace Gfq

open Lk

/-! ## Polynomial-basis field arithmetic -/

def smallestDivisor (q : Nat) : Nat :=
  match (List.range (q - 1)).find? fun i => q % (i + 2) = 0 with
  | some i => i + 2
  | none => q

def isPrime (p : Nat) : Bool :=
  2 ≤ p && (List.range (p - 2)).all fun i => p % (i + 2) ≠ 0

def digits (p : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | d + 1, a => a % p :: digits p d (a / p)

def encode (p : Nat) (cs : List Nat) : Nat :=
  cs.reverse.foldl (fun a c => a * p + c) 0

/-- Cancel coefficient `k` using a monic polynomial of degree `d`. -/
def reduceAt (p d : Nat) (modulus cs : List Nat) (k : Nat) : List Nat :=
  let lead := cs.getD k 0
  (List.range d).foldl (fun out j =>
    let i := k - d + j
    out.set i ((out.getD i 0 + p - lead * modulus.getD j 0 % p) % p)) cs

def polynomialRemainder (p : Nat) (f g : List Nat) : List Nat :=
  let n := f.length - 1
  let d := g.length - 1
  if d = 0 ∨ n < d then f
  else
    let high := (List.range (n - d + 1)).reverse
    let reduced := high.foldl (fun cs t => reduceAt p d g cs (d + t)) f
    reduced.take d

def polynomialDivides (p : Nat) (f g : List Nat) : Bool :=
  (polynomialRemainder p f g).all (· = 0)

def irreducible (p : Nat) (modulus : List Nat) : Bool :=
  let d := modulus.length - 1
  (List.range (d / 2)).all fun z =>
    let k := z + 1
    (List.range (p ^ k)).all fun code =>
      !polynomialDivides p modulus (digits p k code ++ [1])

structure Field where
  q : Nat
  p : Nat
  d : Nat
  modulus : List Nat


def field? (q : Nat) (modulus : List Nat) : Option Field :=
  let p := smallestDivisor q
  let d := modulus.length - 1
  if 2 ≤ q ∧ 1 ≤ d ∧ isPrime p ∧ p ^ d = q ∧ modulus.length = d + 1 ∧
      modulus.getD d 0 = 1 ∧ modulus.all (· < p) ∧ irreducible p modulus then
    some { q, p, d, modulus }
  else none

def validField (q : Nat) (modulus : List Nat) : Bool := (field? q modulus).isSome

def Field.add (K : Field) (a b : Nat) : Nat :=
  encode K.p (List.zipWith (fun x y => (x + y) % K.p) (digits K.p K.d a) (digits K.p K.d b))

def Field.neg (K : Field) (a : Nat) : Nat :=
  encode K.p ((digits K.p K.d a).map fun x => (K.p - x) % K.p)

def Field.mul (K : Field) (a b : Nat) : Nat :=
  let raw := (List.range (2 * K.d - 1)).map fun k =>
    (List.range (k + 1)).foldl (fun acc i =>
      if i < K.d ∧ k - i < K.d then
        (acc + (digits K.p K.d a).getD i 0 * (digits K.p K.d b).getD (k - i) 0) % K.p
      else acc) 0
  let high := (List.range (K.d - 1)).reverse
  let reduced := high.foldl (fun cs t => reduceAt K.p K.d K.modulus cs (K.d + t)) raw
  encode K.p (reduced.take K.d)

/-- Repeated squaring with an explicit base. -/
def Field.powFrom (K : Field) (base : Nat) : Nat → Nat → Nat
  | 0, _ => 1
  | fuel + 1, e =>
    if e = 0 then 1
    else
      let h := K.powFrom (K.mul base base) fuel (e / 2)
      if e % 2 = 1 then K.mul h base else h
termination_by fuel _ => fuel

def Field.inv (K : Field) (a : Nat) : Nat := K.powFrom a 64 (K.q - 2)

def Field.subMul (K : Field) (row other : Vec) (c : Nat) : Vec :=
  List.zipWith (fun a b => K.add a (K.mul (K.neg c) b)) row other

def Field.scale (K : Field) (row : Vec) (c : Nat) : Vec := row.map (K.mul · c)

/-- Public arithmetic names used by the Mathlib contract. Invalid presentations return zero. -/
def add (q : Nat) (modulus : List Nat) (a b : Nat) : Nat :=
  match field? q modulus with | some K => K.add a b | none => 0

def mul (q : Nat) (modulus : List Nat) (a b : Nat) : Nat :=
  match field? q modulus with | some K => K.mul a b | none => 0

/-! ## Gauss-Jordan elimination -/

structure GJ where
  rows : Mat
  r : Nat
  pivots : List Nat


def findPivot (rows : Mat) (c r : Nat) : Option Nat :=
  ((List.range rows.length).drop r).find? fun i => (rows.getD i []).getD c 0 ≠ 0

def swapRows (m : Mat) (i j : Nat) : Mat :=
  (List.range m.length).map fun k =>
    if k = i then m.getD j [] else if k = j then m.getD i [] else m.getD k []

def gjColumn (K : Field) (s : GJ) (c : Nat) : GJ :=
  match findPivot s.rows c s.r with
  | none => s
  | some i =>
    let rows := swapRows s.rows i s.r
    let factor := K.inv ((rows.getD s.r []).getD c 0)
    let pivot := K.scale (rows.getD s.r []) factor
    { rows := (List.range rows.length).map fun k =>
        if k = s.r then pivot
        else K.subMul (rows.getD k []) pivot ((rows.getD k []).getD c 0)
      r := s.r + 1
      pivots := s.pivots ++ [c] }

def gaussJordan (K : Field) (m : Mat) : GJ :=
  let cols := (m.headD []).length
  (List.range cols).foldl (gjColumn K) { rows := m, r := 0, pivots := [] }

def rref (K : Field) (m : Mat) : Mat := (gaussJordan K m).rows

def pivots (K : Field) (m : Mat) : List Nat := (gaussJordan K m).pivots

def rank (K : Field) (m : Mat) : Nat := (pivots K m).length

def nullspace (K : Field) (m : Mat) : List Vec :=
  let g := gaussJordan K m
  let n := (m.headD []).length
  ((List.range n).filter fun f => f ∉ g.pivots).map fun f =>
    (List.range n).map fun j =>
      if j = f then 1
      else match g.pivots.idxOf? j with
        | some i => K.neg ((g.rows.getD i []).getD f 0)
        | none => 0

def inSpan (K : Field) (m : Mat) (target : Vec) : Bool :=
  rank K m = rank K (m ++ [target])

def solve (K : Field) (m : Mat) (rhs : Vec) : Option Vec :=
  let n := (m.headD []).length
  let g := gaussJordan K (List.zipWith (fun row b => row ++ [b]) m rhs)
  if n ∈ g.pivots then none
  else some ((List.range n).map fun j =>
    match g.pivots.idxOf? j with
    | some i => (g.rows.getD i []).getD n 0
    | none => 0)

def inverse (K : Field) (m : Mat) : Option Mat :=
  let n := m.length
  let g := gaussJordan K (List.zipWith (· ++ ·) m (identity n))
  if g.pivots = List.range n then some (g.rows.map fun row => row.drop n) else none

/-! ## Operations -/

inductive Op
  | rank (modulus : Vec)
  | inSpan (modulus target : Vec)
  | rref (modulus : Vec)
  | nullspace (modulus : Vec)
  | solve (modulus : Vec) (rhs : List Vec)
  | inverse (modulus : Vec)

def Op.modulus : Op → Vec
  | .rank f | .rref f | .nullspace f | .inverse f => f
  | .inSpan f _ | .solve f _ => f

inductive Value
  | matrix (m : Mat)
  | basis (b : List Vec)
  | solution (x : Option Vec)
  | inverse (m : Option Mat)
  deriving DecidableEq, Repr

def supportedFamily : Family → Bool
  | .explicit _ _ | .subsets _ _ _ | .grassmannian _ _ _ | .allMatrices _ _ _ |
    .symmetricMatrices _ _ => true
  | .subsetsOf inner _ | .stack inner _ => supportedFamily inner
  | _ => false

def run (op : Op) (family : Family) (red : Red) : Result Value :=
  if !supportedFamily family then .invalid
  else match field? family.p op.modulus with
  | none => .invalid
  | some K =>
    let ms := family.members
    let rows := (ms.headD []).length
    let cols := ((ms.headD []).headD []).length
    match op with
    | .rank _ => reduceInt red ms (ms.map (rank K))
    | .inSpan _ target =>
      if target.length = cols then reduceBool red ms (ms.map fun m => inSpan K m target) else .invalid
    | .rref _ => reduceValues red (ms.map fun m => .matrix (rref K m))
    | .nullspace _ => reduceValues red (ms.map fun m => .basis (nullspace K m))
    | .solve _ rhs => match family with
      | .explicit _ _ =>
        if rhs.length = ms.length ∧ rhs.all (·.length = rows) then
          reduceValues red (List.zipWith (fun m b => .solution (solve K m b)) ms rhs)
        else .invalid
      | _ => .invalid
    | .inverse _ => match family with
      | .explicit _ _ =>
        if rows = cols then reduceValues red (ms.map fun m => .inverse (inverse K m)) else .invalid
      | _ => .invalid

end Gfq
