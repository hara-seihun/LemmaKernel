import Gfp.Reference

/-!
# strongly_regular: executable reference

Each family member is a square matrix over F_2, read as an adjacency matrix. The reference checks
simplicity, symmetry, regularity, and every pair's common-neighbour count directly. Its recursion
is all through lists, so `decide +kernel` can use it as the test oracle.

The spectrum is exact. `(k, neg, absDelta, D, f, g)` means eigenvalue `k` with multiplicity one,
followed by `(delta + sqrt D) / 2` with multiplicity `f` and `(delta - sqrt D) / 2` with
multiplicity `g`. Here `delta` is `-absDelta` when `neg = 1`, and `absDelta` otherwise.
-/

namespace Strongly_regular

open Lk

abbrev Parameters := Nat × Nat × Nat × Nat
abbrev SpectrumData := Nat × Nat × Nat × Nat × Nat × Nat

def commonCount (m : Mat) (i j : Nat) : Nat :=
  (List.zipWith (fun a b => decide (a = 1 ∧ b = 1)) (m.getD i []) (m.getD j [])).filter id |>.length

def squareZeroSymmetric (m : Mat) : Bool :=
  let n := m.length
  (List.range n).all fun i =>
    (m.getD i []).length = n ∧ (m.getD i []).getD i 0 = 0 ∧
      (List.range n).all fun j => (m.getD i []).getD j 0 = (m.getD j []).getD i 0

def degree (m : Mat) (i : Nat) : Nat := ((m.getD i []).filter (· = 1)).length

def pairCounts (m : Mat) (adjacent : Bool) : List Nat :=
  let n := m.length
  (List.range n).flatMap fun i =>
    ((List.range n).filter fun j => i < j ∧ decide ((m.getD i []).getD j 0 = 1) = adjacent).map
      (commonCount m i)

def constantValue : List Nat → Option Nat
  | [] => none
  | x :: xs => if xs.all (· = x) then some x else none

def srgParams (m : Mat) : Option Parameters :=
  let v := m.length
  if !squareZeroSymmetric m then none else
  let k := degree m 0
  if k = 0 ∨ v ≤ k + 1 then none else
  if !(List.range v).all (fun i => degree m i = k) then none else
  match constantValue (pairCounts m true), constantValue (pairCounts m false) with
  | some lam, some mu => some (v, k, lam, mu)
  | _, _ => none

def exactNonnegativeQuotient (a b : Int) : Option Nat :=
  if a < 0 ∨ b ≤ 0 ∨ a % b ≠ 0 then none else some (a / b).toNat

def spectrumOf (p : Parameters) : Option SpectrumData :=
  let (v, k, lam, mu) := p
  let delta : Int := (lam : Int) - (mu : Int)
  let discriminantInt : Int := delta * delta + 4 * ((k : Int) - (mu : Int))
  if discriminantInt ≤ 0 then none else
  let discriminant := discriminantInt.toNat
  let imbalance : Int := 2 * (k : Int) + ((v - 1 : Nat) : Int) * delta
  let q := Nat.sqrt discriminant
  if q * q = discriminant then
    let base : Int := ((v - 1 : Nat) : Int) * (q : Int)
    let denominator : Int := 2 * (q : Int)
    match exactNonnegativeQuotient (base - imbalance) denominator,
          exactNonnegativeQuotient (base + imbalance) denominator with
    | some f, some g => some (k, if delta < 0 then 1 else 0, delta.natAbs, discriminant, f, g)
    | _, _ => none
  else if imbalance = 0 ∧ (v - 1) % 2 = 0 then
    some (k, if delta < 0 then 1 else 0, delta.natAbs, discriminant, (v - 1) / 2, (v - 1) / 2)
  else none

structure Quad where
  a : Int
  b : Int

def Quad.add (x y : Quad) : Quad := ⟨x.a + y.a, x.b + y.b⟩
def Quad.sub (x y : Quad) : Quad := ⟨x.a - y.a, x.b - y.b⟩
def Quad.scale (c : Int) (x : Quad) : Quad := ⟨c * x.a, c * x.b⟩
def Quad.mul (d : Nat) (x y : Quad) : Quad :=
  ⟨x.a * y.a + x.b * y.b * (d : Int), x.a * y.b + x.b * y.a⟩
def Quad.cube (d : Nat) (x : Quad) : Quad := (x.mul d x).mul d x

def Quad.nonnegative (d : Nat) (x : Quad) : Bool :=
  if 0 ≤ x.a then
    if 0 ≤ x.b then true else x.b * x.b * (d : Int) ≤ x.a * x.a
  else
    if x.b ≤ 0 then false else x.a * x.a ≤ x.b * x.b * (d : Int)

def kreinOne (p : Parameters) (positiveRoot : Bool) : Bool :=
  let (v, k, lam, mu) := p
  let delta : Int := (lam : Int) - (mu : Int)
  let d := (delta * delta + 4 * ((k : Int) - (mu : Int))).toNat
  let x : Quad := ⟨delta, if positiveRoot then 1 else -1⟩
  let xPlusTwo : Quad := ⟨delta + 2, if positiveRoot then 1 else -1⟩
  let c : Int := ((v - k - 1) * (v - k - 1) : Nat)
  let kk : Int := (k * k : Nat)
  let lhs := (Quad.scale c ((Quad.cube d x).add ⟨8 * kk, 0⟩)).sub
    (Quad.scale kk (Quad.cube d xPlusTwo))
  lhs.nonnegative d

def kreinBound : Parameters → Bool
  | p => kreinOne p true ∧ kreinOne p false

def absoluteBound (p : Parameters) : Bool :=
  let (v, k, lam, mu) := p
  if mu = 0 ∨ v + lam = 2 * k then true else
  match spectrumOf p with
  | some (_, _, _, _, f, g) => 2 * v ≤ f * (f + 3) ∧ 2 * v ≤ g * (g + 3)
  | none => false

def adjacencyFamily (f : Family) : Bool :=
  f.p = 2 ∧ f.members.all fun m => m.all (·.length = m.length)

inductive Op
  | srgParams | spectrum | isSrg | kreinBound | absoluteBound

inductive Value
  | params (p : Option Parameters)
  | spectrum (s : Option SpectrumData)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  if !adjacencyFamily f then .invalid else
  let ms := f.members
  let ps := ms.map srgParams
  match op with
  | .srgParams => reduceValues red (ps.map .params)
  | .spectrum => reduceValues red (ps.map fun p => .spectrum (p.bind spectrumOf))
  | .isSrg => reduceBool red ms (ps.map Option.isSome)
  | .kreinBound => reduceBool red ms (ps.map fun p => p.any kreinBound)
  | .absoluteBound => reduceBool red ms (ps.map fun p => p.any absoluteBound)

end Strongly_regular
