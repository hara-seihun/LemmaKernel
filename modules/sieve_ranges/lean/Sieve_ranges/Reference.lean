import Lk.Reference

/-!
# sieve_ranges: executable reference

Every operation of the module is a function of one natural number's prime factorisation, so the
factorisation is written once, by trial division, and everything else reads it. Trial division is
the slow way to factor a whole interval; that is the backend's problem, not the oracle's. What
matters here is that the definition is plain structural recursion the Lean kernel can evaluate on
the few dozen members of an oracle case.

The conventions at the ends of the domain are Mathlib's, and they are what makes every answer a
total function: `0` and `1` both have the empty factorisation, so `phi 0 = sigma 0 = tau 0 = 0`
(the empty product is corrected to zero for `0`, which has no divisors) and
`phi 1 = sigma 1 = tau 1 = 1`. Moebius comes back shifted by one, `mu n + 1 ∈ {0, 1, 2}`, because
a reduction carries unsigned values; the module's manifest says so and the backend agrees.

Members are single natural numbers: a `range` family, or an `explicit` family of 1 x 1
`lk.naturals` members. Any other family is `.invalid`.
-/

namespace Sieve_ranges

open Lk

/-! ## Factorisation by trial division -/

/-- Divide `d` out of `n` as often as it goes: the exponent and the remaining cofactor. -/
def peel : Nat → Nat → Nat → Nat × Nat
  | 0, _, n => (0, n)
  | fuel + 1, d, n =>
    if 1 < d ∧ 0 < n ∧ n % d = 0 then
      let (e, m) := peel fuel d (n / d)
      (e + 1, m)
    else (0, n)

/-- Trial division of `n` by the candidates `d, d+1, ...`. Once `d * d > n` the cofactor left is
either `1` or a prime, because every smaller factor has been divided out. `fuel` bounds the
recursion; `n` is always enough, since each step either raises `d` or shrinks `n`. -/
def factorFrom : Nat → Nat → Nat → List (Nat × Nat)
  | 0, _, _ => []
  | fuel + 1, d, n =>
    if n ≤ 1 then []
    else if n < d * d then [(n, 1)]
    else if n % d = 0 then
      let (e, m) := peel (fuel + 1) d n
      (d, e) :: factorFrom fuel (d + 1) m
    else factorFrom fuel (d + 1) n

/-- The prime factorisation of `n`: `(prime, exponent)` pairs with primes increasing and every
exponent at least one. `0` and `1` have none. -/
def factorisation (n : Nat) : List (Nat × Nat) := factorFrom n 2 n

/-! ## The arithmetic functions -/

/-- Number of distinct prime factors. -/
def omega (n : Nat) : Nat := (factorisation n).length

/-- Number of prime factors with multiplicity. -/
def bigOmega (n : Nat) : Nat := ((factorisation n).map Prod.snd).foldl (· + ·) 0

/-- Euler's totient: `∏ p^(e-1) (p-1)`, and `0` at `0`. -/
def totient (n : Nat) : Nat :=
  if n = 0 then 0
  else (factorisation n).foldl (fun acc pe => acc * pe.1 ^ (pe.2 - 1) * (pe.1 - 1)) 1

/-- Sum of divisors: `∏ (p^(e+1) - 1)/(p - 1)`, and `0` at `0`. -/
def sigma (n : Nat) : Nat :=
  if n = 0 then 0
  else (factorisation n).foldl (fun acc pe => acc * ((pe.1 ^ (pe.2 + 1) - 1) / (pe.1 - 1))) 1

/-- Number of divisors: `∏ (e+1)`, and `0` at `0`. -/
def divisorCount (n : Nat) : Nat :=
  if n = 0 then 0 else (factorisation n).foldl (fun acc pe => acc * (pe.2 + 1)) 1

/-- No square above one divides `n`. `1` is squarefree; `0` is not. -/
def isSquarefree (n : Nat) : Bool := n ≠ 0 ∧ (factorisation n).all fun pe => pe.2 = 1

/-- `n` is prime exactly when it is its own only prime factor, to the first power. -/
def isPrime (n : Nat) : Bool := factorisation n == [(n, 1)]

/-- Moebius, shifted: `0` for `mu = -1`, `1` for `mu = 0`, `2` for `mu = +1`. -/
def mobius (n : Nat) : Nat :=
  if isSquarefree n then (if omega n % 2 = 0 then 2 else 0) else 1

/-- The largest prime factor, or `0` when there is none. -/
def largestPrimeFactor (n : Nat) : Nat := ((factorisation n).map Prod.fst).foldl max 0

/-! ## Operations and values -/

inductive Op
  | isPrime | isSquarefree | factorisation
  | omega | bigOmega | totient | sigma | divisorCount | mobius | largestPrimeFactor

/-- One materialised output per member. -/
inductive Value
  | factorisation (fs : List (Nat × Nat))
  deriving DecidableEq, Repr

/-- The numbers a family presents, or `none` when its members are not single naturals. An
`explicit` family carries the runtime's `naturals` sentinel as its prime when its members are
natural numbers rather than elements of a field. -/
def numbers? : Family → Option (List Nat)
  | .range a b => some ((List.range (b - a)).map (a + ·))
  | .explicit p batch =>
    if p = Lk.naturals ∧ batch.all fun m => m.length = 1 ∧ (m.headD []).length = 1 then
      some (batch.map fun m => (m.headD []).headD 0)
    else none
  | _ => none

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match numbers? f with
  | none => .invalid
  | some ns =>
    let ms := f.members
    match op with
    | .isPrime => reduceBool red ms (ns.map isPrime)
    | .isSquarefree => reduceBool red ms (ns.map isSquarefree)
    | .factorisation => reduceValues red (ns.map fun n => .factorisation (factorisation n))
    | .omega => reduceInt red ms (ns.map omega)
    | .bigOmega => reduceInt red ms (ns.map bigOmega)
    | .totient => reduceInt red ms (ns.map totient)
    | .sigma => reduceInt red ms (ns.map sigma)
    | .divisorCount => reduceInt red ms (ns.map divisorCount)
    | .mobius => reduceInt red ms (ns.map mobius)
    | .largestPrimeFactor => reduceInt red ms (ns.map largestPrimeFactor)

end Sieve_ranges
