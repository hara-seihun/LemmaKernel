import Lk.Reference

/-!
# residues: executable reference

The meaning of every `residues` operation, written as plain structural recursion so that Lean's
kernel can evaluate it. Families, reductions and the result shape come from `Lk.Reference`. This
is the oracle: a kernel answer is correct when `example : run op family red = answer := by decide`
is accepted.

Every definition here is the mathematics spelled out, not the algorithm the backend runs: an
order is found by multiplying up, a quadratic residue by looking for a square root, a discrete
logarithm by walking the powers, and a Jacobi symbol by multiplying Legendre symbols over the
prime factorisation. The fast backend has to agree with all of it bit for bit.

Members come from a `range` family, one natural per member. For most operations that natural is a
residue and the modulus is an argument; for `leastPrimitiveRoot` the member is the modulus.
-/

namespace Residues

open Lk

/-! ## Arithmetic in `Z/n` -/

def gcdLoop : Nat → Nat → Nat → Nat
  | 0, a, _ => a
  | fuel + 1, a, b => if b = 0 then a else gcdLoop fuel b (a % b)

/-- Euclid's algorithm; 96 divisions cover any pair below `2 ^ 64`. -/
def gcd (a b : Nat) : Nat := gcdLoop 96 a b

/-- The number of units of `Z/n`. -/
def totient (n : Nat) : Nat := ((List.range n).filter fun a => gcd a n = 1).length

/-- `k + 1` for the least `k + 1 ≥ 1` with `v * a ^ (k + 1) = 1`, walking the powers of `a` from
`v = a ^ k`; `0` once the fuel runs out, which happens exactly when `a` is not a unit. -/
def orderLoop (n a : Nat) : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | fuel + 1, k, v =>
    let w := v * a % n
    if w = 1 % n then k + 1 else orderLoop n a fuel (k + 1) w

/-- The multiplicative order of `a` in `Z/n`: the least `k ≥ 1` with `a ^ k = 1`, and `0` when
there is none. A unit has order at most `totient n < n`, so `n` steps decide it. -/
def order (n a : Nat) : Nat := orderLoop n a n 0 (1 % n)

/-- `a` generates the units of `Z/n`. -/
def isPrimitiveRoot (n a : Nat) : Bool := gcd a n = 1 ∧ order n a = totient n

/-- `a` is a unit of `Z/n` with a square root there. -/
def isQuadraticResidue (n a : Nat) : Bool :=
  gcd a n = 1 ∧ (List.range n).any fun x => x * x % n = a % n

/-- Walk `v = base ^ x` for `x = 0, 1, ...` and stop at the first `x` with `v = a`. -/
def dlogLoop (n base a : Nat) : Nat → Nat → Nat → Nat
  | 0, _, _ => n
  | fuel + 1, x, v => if v = a % n then x else dlogLoop n base a fuel (x + 1) (v * base % n)

/-- The least `x < n` with `base ^ x = a` in `Z/n`, and `n` when there is none. The powers of
`base` repeat within `n` steps, so nothing is missed. -/
def discreteLog (n base a : Nat) : Nat := dlogLoop n base a n 0 (1 % n)

/-! ## Factorisation and the quadratic symbols -/

/-- Trial division by every `d ≥ 2`: the prime factors of `n` with multiplicity, ascending. -/
def factorsLoop : Nat → Nat → Nat → List Nat
  | 0, n, _ => if n ≤ 1 then [] else [n]
  | fuel + 1, n, d =>
    if n ≤ 1 then []
    else if n < d * d then [n]
    else if n % d = 0 then d :: factorsLoop fuel (n / d) d
    else factorsLoop fuel n (d + 1)

/-- Each step either divides `n` (at most `log₂ n` times) or raises the trial divisor (at most
`√n` times), so `n + 2` steps are enough. -/
def factors (n : Nat) : List Nat := factorsLoop (n + 2) n 2

def isPrime (n : Nat) : Bool := 2 ≤ n ∧ factors n = [n]

/-- The Legendre symbol `(a / p)` of an odd prime `p`, encoded `0` for `0`, `1` for `1`, and `2`
for `-1`: zero when `p` divides `a`, one when `a` is a nonzero square, and `-1` otherwise. -/
def legendre (p a : Nat) : Nat :=
  if a % p = 0 then 0
  else if (List.range p).any (fun x => x * x % p = a % p) then 1 else 2

/-- The encoded symbols `{0, 1, 2}` multiply as `{0, 1, -1}` do. -/
def mulSymbol (x y : Nat) : Nat := if x = 0 ∨ y = 0 then 0 else if x = y then 1 else 2

/-- The Jacobi symbol `(a / n)` of an odd `n`: the product of the Legendre symbols of the prime
factors of `n`, with multiplicity, in the same encoding. The empty product gives `(a / 1) = 1`. -/
def jacobi (n a : Nat) : Nat := (factors n).foldl (fun s p => mulSymbol s (legendre p a)) 1

/-- Try `g = 1, 2, ...` in turn. -/
def leastRootLoop (n : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, g => if isPrimitiveRoot n g then g else leastRootLoop n fuel (g + 1)

/-- The least `g ≥ 1` that generates the units of `Z/n`, and `0` when `(Z/n)ˣ` is not cyclic.
When one exists it is at most `n`, so `n` candidates decide it. -/
def leastPrimitiveRoot (n : Nat) : Nat := leastRootLoop n n 1

/-! ## Operations and values -/

inductive Op
  | multiplicativeOrder (modulus : Nat)
  | isPrimitiveRoot (modulus : Nat)
  | isQuadraticResidue (modulus : Nat)
  | discreteLog (base modulus : Nat)
  | legendre (modulus : Nat)
  | jacobi (modulus : Nat)
  | leastPrimitiveRoot

/-- Every operation reduces to an integer or a boolean, so nothing is materialised per member. -/
inductive Value
  | unit
  deriving DecidableEq, Repr

/-- A modulus the runtime can work in: at least one, and below the width of a member entry. -/
def validModulus (n : Nat) : Bool := 1 ≤ n ∧ n < 4294967296

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match f with
  | .range lo _ =>
    let ms := f.members
    let xs := ms.map fun m => (m.headD []).headD 0
    match op with
    | .multiplicativeOrder n =>
      if validModulus n then reduceInt red ms (xs.map (order n)) else .invalid
    | .isPrimitiveRoot n =>
      if validModulus n then reduceBool red ms (xs.map (isPrimitiveRoot n)) else .invalid
    | .isQuadraticResidue n =>
      if validModulus n then reduceBool red ms (xs.map (isQuadraticResidue n)) else .invalid
    | .discreteLog b n =>
      if validModulus n then reduceInt red ms (xs.map (discreteLog n b)) else .invalid
    | .legendre n =>
      if validModulus n ∧ n % 2 = 1 ∧ isPrime n then reduceInt red ms (xs.map (legendre n))
      else .invalid
    | .jacobi n =>
      if validModulus n ∧ n % 2 = 1 then reduceInt red ms (xs.map (jacobi n)) else .invalid
    | .leastPrimitiveRoot =>
      if 1 ≤ lo then reduceInt red ms (xs.map leastPrimitiveRoot) else .invalid
  | _ => .invalid

end Residues
