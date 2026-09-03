import Lk.Reference

/-!
# continued_fractions_and_pell: executable reference

The meaning of every operation of the module, written as plain structural recursion so that
Lean's kernel can evaluate it. Families, reductions and the result shape come from
`Lk.Reference`. This is the oracle: a kernel answer is correct when
`example : run op family red = answer := by decide +kernel` is accepted.

A member is one natural number held as a `1 x 1` matrix: the members of a `range` family, or of
an `explicit` family over an `lk.naturals` batch. Two readings of that number are in play and
each operation fixes one. The continued-fraction, unit and Pell operations read it as a radicand
`d` and work in the real quadratic order `Z[sqrt d]`; `classNumber` reads it as `n` for the
negative discriminant `D = -n`.

Everything mathematics leaves open is decided here and nowhere else, and the backend and the
naive implementation reproduce it byte for byte:

* the expansion of `sqrt d` is `a_0 = floor (sqrt d)` followed by exactly one period
  `a_1 .. a_L`, the period ending at the first complete quotient with denominator `1`, where
  `a_L = 2 * a_0`;
* a perfect square `d`, including `0` and `1`, has expansion `[a_0]`, period length `0`, and
  neither a unit nor a Pell solution;
* the fundamental unit is `p_{L-1} + q_{L-1} sqrt d`, the least `x + y sqrt d > 1` with
  `x, y > 0` and `x^2 - d y^2 = ±1`; its norm is `(-1)^L`;
* the fundamental Pell solution is the least `x, y > 0` with `x^2 - d y^2 = 1`: the unit when
  `L` is even, its square when `L` is odd;
* `classNumber n` counts primitive reduced positive definite forms of discriminant `-n`.

Units and Pell solutions cross the boundary as pairs of `u64`, so a request whose answer does not
fit is `.invalid` here and refused by the runtime; nothing is truncated.
-/

namespace Continued_fractions_and_pell

open Lk

/-! ## The natural number a member carries -/

/-- The families this module is defined on: a `range`, or an explicit batch of `1 x 1`
natural-number matrices. -/
def supported : Family → Bool
  | .range _ _ => true
  | .explicit p batch => p == Lk.naturals && batch.all fun m => m.length == 1 ∧ (m.headD []).length == 1
  | _ => false

/-- The single entry of a `1 x 1` member. -/
def numberOf (m : Mat) : Nat := (m.headD []).headD 0

/-! ## Integer square root -/

/-- `floor (sqrt n)`, as the largest `x ≤ n` with `x * x ≤ n`. Written as a search rather than an
iteration because this file is the definition and not the algorithm; the oracle only ever runs it
on the small members a Lean claim can afford. -/
def isqrt (n : Nat) : Nat := ((List.range (n + 1)).filter fun x => x * x ≤ n).foldl max 0

def isSquare (n : Nat) : Bool := isqrt n * isqrt n = n

/-! ## The continued fraction of `sqrt d`

The complete quotient after `k` steps is `(m_k + sqrt d) / q_k`, whose integer part is
`(a_0 + m_k) / q_k`. From `(m_1, q_1) = (a_0, d - a_0^2)` the recurrence is
`m_{k+1} = q_k a_k - m_k` and `q_{k+1} = (d - m_{k+1}^2) / q_k`, and the period ends at the first
`k ≥ 1` with `q_k = 1`, where `a_k = 2 a_0`. -/

/-- The partial quotients from state `(m, q)` up to and including the one whose state has
`q = 1`. -/
def periodAux (d a0 : Nat) : Nat → Nat → Nat → List Nat
  | 0, _, _ => []
  | fuel + 1, m, q =>
    let a := (a0 + m) / q
    if q = 1 then [a]
    else
      let m' := q * a - m
      let q' := (d - m' * m') / q
      a :: periodAux d a0 fuel m' q'

/-- One period `a_1 .. a_L` of `sqrt d`, and `[]` when `d` is a perfect square. Every state has
`0 < m < sqrt d` and `0 < q < 2 sqrt d`, so there are fewer than `2 d` of them and the fuel is
never the reason the walk stops. -/
def periodTerms (d : Nat) : List Nat :=
  let a0 := isqrt d
  if a0 * a0 = d then [] else periodAux d a0 (2 * d + 2) a0 (d - a0 * a0)

/-- `a_0` followed by exactly one period: the canonical expansion `cfExpansion` returns. -/
def expansionTerms (d : Nat) : List Nat := isqrt d :: periodTerms d

def periodLength (d : Nat) : Nat := (periodTerms d).length

def periodMax (d : Nat) : Nat := (periodTerms d).foldl max 0

def periodSum (d : Nat) : Nat := (periodTerms d).foldl (· + ·) 0

/-- `x^2 - d y^2 = -1` is solvable in positive integers: `d` is not a perfect square and its
period has odd length. -/
def negativePell (d : Nat) : Bool := periodLength d % 2 = 1

/-! ## Units and Pell solutions -/

/-- Numerator and denominator of the finite continued fraction `xs`, from the usual seeds
`p_{-1}/q_{-1} = 1/0` and `p_{-2}/q_{-2} = 0/1`. -/
def convergent (xs : List Nat) : Nat × Nat :=
  (xs.foldl (fun (s : (Nat × Nat) × (Nat × Nat)) a =>
    ((a * s.1.1 + s.2.1, a * s.1.2 + s.2.2), s.1)) ((1, 0), (0, 1))).1

/-- The fundamental unit of `Z[sqrt d]` as `(norm is -1, x, y)`: the convergent
`p_{L-1}/q_{L-1}`, which solves `x^2 - d y^2 = (-1)^L`. None for a perfect square. -/
def unitOf (d : Nat) : Option (Nat × Nat × Nat) :=
  let period := periodTerms d
  if period.isEmpty then none
  else
    let (x, y) := convergent (isqrt d :: period.take (period.length - 1))
    some (if period.length % 2 = 1 then 1 else 0, x, y)

/-- The fundamental solution of `x^2 - d y^2 = 1`. When the unit has norm `-1` its square is the
least solution of norm `+1`: `(x + y sqrt d)^2 = (2 x^2 + 1) + 2 x y sqrt d`. -/
def pellOf (d : Nat) : Option (Nat × Nat × Nat) :=
  match unitOf d with
  | none => none
  | some (neg, x, y) => if neg = 1 then some (0, 2 * x * x + 1, 2 * x * y) else some (0, x, y)

/-- `x` and `y` cross the boundary as `u64`. -/
def fits : Option (Nat × Nat × Nat) → Bool
  | none => true
  | some (_, x, y) => x < 18446744073709551616 ∧ y < 18446744073709551616

/-! ## Class numbers of imaginary quadratic orders -/

def gcdLoop : Nat → Nat → Nat → Nat
  | 0, a, _ => a
  | fuel + 1, a, b => if b = 0 then a else gcdLoop fuel b (a % b)

/-- Euclid's algorithm; 96 divisions cover any pair below `2 ^ 64`. -/
def gcd (a b : Nat) : Nat := gcdLoop 96 a b

/-- The class number `h(-n)`: the number of primitive reduced positive definite binary quadratic
forms `(a, b, c)` of discriminant `b^2 - 4ac = -n`, that is those with `gcd (a, b, c) = 1`,
`-a < b ≤ a ≤ c`, and `b ≥ 0` when `a = c`. Reduction pairs `(a, b, c)` with `(a, -b, c)`, so a
form with `0 < b < a < c` is counted twice and the boundary cases `b = 0`, `b = a` and `a = c`
once. A reduced form has `n = 4ac - b^2 ≥ 3a^2`, which bounds the search. It is `0` when `-n` is
not a discriminant, that is when `n = 0` or `n % 4 ∈ {1, 2}`. -/
def classNumber (n : Nat) : Nat :=
  if n = 0 ∨ (n % 4 ≠ 0 ∧ n % 4 ≠ 3) then 0
  else
    ((List.range (isqrt (n / 3))).map (· + 1)).foldl (fun total a =>
      total + (List.range (a + 1)).foldl (fun row b =>
        let t := b * b + n
        if t % (4 * a) ≠ 0 then row
        else
          let c := t / (4 * a)
          if c < a ∨ gcd (gcd a b) c ≠ 1 then row
          else if b = 0 ∨ b = a ∨ c = a then row + 1
          else row + 2) 0) 0

/-! ## Operations and values -/

inductive Op
  | cfPeriod | cfPeriodMax | cfPeriodSum | cfExpansion
  | negativePell | fundamentalUnit | pellFundamental | classNumber

/-- The materialised outputs: one continued fraction, or one unit, per member. -/
inductive Value
  | expansion (terms : List Nat)
  | unit (u : Option (Nat × Nat × Nat))
  deriving DecidableEq, Repr

/-- A unit-valued operation refuses the whole request when any member's answer is too wide for
the interchange encoding, so that no member is ever truncated. -/
def units (red : Red) (us : List (Option (Nat × Nat × Nat))) : Result Value :=
  if us.all fits then reduceValues red (us.map Value.unit) else .invalid

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  if !supported f then .invalid
  else
    let ms := f.members
    let ns := ms.map numberOf
    match op with
    | .cfPeriod => reduceInt red ms (ns.map periodLength)
    | .cfPeriodMax => reduceInt red ms (ns.map periodMax)
    | .cfPeriodSum => reduceInt red ms (ns.map periodSum)
    | .classNumber => reduceInt red ms (ns.map classNumber)
    | .negativePell => reduceBool red ms (ns.map negativePell)
    | .cfExpansion => reduceValues red (ns.map fun d => .expansion (expansionTerms d))
    | .fundamentalUnit => units red (ns.map unitOf)
    | .pellFundamental => units red (ns.map pellOf)

end Continued_fractions_and_pell
