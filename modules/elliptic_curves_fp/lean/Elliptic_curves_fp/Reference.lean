import Lk.Reference

/-!
# elliptic_curves_fp: executable reference

The meaning of every elliptic_curves_fp operation, written as plain structural recursion so that
Lean's kernel can evaluate it. Families, reductions and the result shape come from `Lk.Reference`.
This is the oracle: a kernel answer is correct when `example : run op family red = answer := by
decide +kernel` is accepted.

A member is a `1 x 2` matrix over `F_p` carrying the coefficients `(a, b)` of the short
Weierstrass curve `y^2 = x^3 + a*x + b`. Short Weierstrass form needs `p > 3`, so smaller primes
and members of any other shape make `run` `.invalid`, matching what the runtime refuses.

Every choice mathematics leaves open is made here and nowhere else:

* `pointCount` counts the projective curve `y^2*z = x^3 + a*x*z^2 + b*z^3`, whose only point with
  `z = 0` is `[0:1:0]`, so it is the number of affine solutions plus one. It is defined for
  singular `(a, b)` too, where it counts the points of the singular cubic.
* `jInvariant` returns a representative in `0..p-1`, and `p` (outside `F_p`) for a singular pair.
* the isomorphism class of `(a, b)` is the orbit `{(u^4*a, u^6*b) : u ≠ 0}` and its canonical
  representative is the lexicographically least pair in it.
* `groupStructure` returns the invariant factors `(n1, n2)`, `n1 ∣ n2`, and `(0, 0)` when singular.
-/

namespace Elliptic_curves_fp

open Lk

/-! ## Members -/

/-- The coefficients `(a, b)` carried by a member. -/
def curveOf (m : Mat) : Nat × Nat := ((m.headD []).getD 0 0, (m.headD []).getD 1 0)

/-- Members are one row of two entries; anything else is not a curve. -/
def wellShaped (ms : List Mat) : Bool := ms.all fun m => m.length = 1 ∧ (m.headD []).length = 2

/-- Short Weierstrass form needs an odd prime that is not 3. -/
def supported (p : Nat) (ms : List Mat) : Bool := 3 < p ∧ wellShaped ms

/-! ## The curve -/

/-- `x^3 + a*x + b` in `F_p`. -/
def rhs (p a b x : Nat) : Nat := (x * x % p * x + a * x + b) % p

/-- How many `y ∈ F_p` satisfy `y^2 = v`, by Euler's criterion (`p` an odd prime). -/
def rootCount (p v : Nat) : Nat :=
  if v = 0 then 1 else if powMod p v 64 ((p - 1) / 2) = 1 then 2 else 0

/-- `#E(F_p)`: the affine solutions of `y^2 = x^3 + a*x + b`, plus the point at infinity. -/
def pointCount (p a b : Nat) : Nat :=
  (List.range p).foldl (fun acc x => acc + rootCount p (rhs p a b x)) 1

/-- `4*a^3 + 27*b^2` in `F_p`. The discriminant is `-16` times this, so for `p > 3` the pair is
singular exactly when this vanishes. -/
def singularForm (p a b : Nat) : Nat := (4 * (a * a % p * a % p) + 27 * (b * b % p)) % p

def nonsingular (p a b : Nat) : Bool := singularForm p a b ≠ 0

/-- Nonsingular with `#E(F_p) = p + 1`; for `p > 3` that is exactly a vanishing trace of
Frobenius, since `|a_p| ≤ 2√p < p`. -/
def supersingular (p a b : Nat) : Bool := nonsingular p a b ∧ pointCount p a b = p + 1

/-- `1728*4*a^3 / (4*a^3 + 27*b^2)` in `F_p`, and `p` for a singular pair. -/
def jInvariant (p a b : Nat) : Nat :=
  let d := singularForm p a b
  if d = 0 then p else 6912 % p * (a * a % p * a % p) % p * inv p d % p

/-! ## Isomorphism classes

Over `F_p` with `p > 3`, `y^2 = x^3 + a*x + b` and `y^2 = x^3 + a'*x + b'` are isomorphic exactly
when `a' = u^4*a` and `b' = u^6*b` for some `u ∈ F_p^*`, the isomorphism being
`(x, y) ↦ (u^2*x, u^3*y)`. The same orbit is taken of a singular pair. -/

/-- The orbit of `(a, b)`, listed by `u = 1, 2, ..., p-1`; repeats are kept. -/
def isoOrbit (p a b : Nat) : List (Nat × Nat) :=
  ((List.range p).drop 1).map fun u =>
    let u2 := u * u % p
    let u4 := u2 * u2 % p
    (u4 * a % p, u4 * u2 % p * b % p)

def pairLe (x y : Nat × Nat) : Bool := x.1 < y.1 ∨ (x.1 = y.1 ∧ x.2 ≤ y.2)

/-- Whether `(a, b)` is the lexicographically least pair of its isomorphism class. -/
def isCanonical (p a b : Nat) : Bool := (isoOrbit p a b).all (pairLe (a, b))

/-- The `u` that fix `(a, b)`: a subgroup of `F_p^*`. -/
def classStabiliser (p a b : Nat) : Nat :=
  (((List.range p).drop 1).filter fun u =>
    let u2 := u * u % p
    let u4 := u2 * u2 % p
    u4 * a % p = a ∧ u4 * u2 % p * b % p = b).length

/-- The number of distinct pairs in the class. The class is the coset space of the stabiliser, so
it has `(p-1)/|stabiliser|` elements; counting them that way costs one pass over `F_p^*` instead
of deduplicating the orbit, which the kernel would pay for quadratically. -/
def classSize (p a b : Nat) : Nat := (p - 1) / classStabiliser p a b

/-! ## The group of points

A point is `none` (the point at infinity, the identity) or `some (x, y)`. The chord-and-tangent
law is the usual one; `p > 3` keeps `2*y` and `3*x^2 + a` meaningful. -/

abbrev Point := Option (Nat × Nat)

/-- Affine points ordered by `x`, then by `y`. -/
def affinePoints (p a b : Nat) : List (Nat × Nat) :=
  (List.range p).flatMap fun x =>
    (List.range p).filterMap fun y => if y * y % p = rhs p a b x then some (x, y) else none

def addPoints (p a : Nat) : Point → Point → Point
  | none, q => q
  | q, none => q
  | some (x₁, y₁), some (x₂, y₂) =>
    if x₁ = x₂ ∧ (y₁ + y₂) % p = 0 then none
    else
      let l :=
        if x₁ = x₂ then (3 * (x₁ * x₁ % p) + a) % p * inv p (2 * y₁ % p) % p
        else (y₂ + p - y₁) % p * inv p ((x₂ + p - x₁) % p) % p
      let x₃ := (l * l % p + 2 * p - x₁ - x₂) % p
      let y₃ := (l * ((x₁ + p - x₃) % p) % p + p - y₁) % p
      some (x₃, y₃)

/-- Repeated addition; `k` counts the multiple `cur` already is. -/
def orderLoop (p a : Nat) (base : Point) : Nat → Point → Nat → Nat
  | 0, _, k => k
  | fuel + 1, cur, k => if cur.isNone then k else orderLoop p a base fuel (addPoints p a cur base) (k + 1)

/-- The least `m ≥ 1` with `m·P = O`; `fuel` bounds the search and `#E(F_p)` always suffices. -/
def pointOrder (p a : Nat) (P : Point) (fuel : Nat) : Nat := orderLoop p a P fuel P 1

/-- The exponent of `E(F_p)`: the least common multiple of the orders of its points. In a finite
abelian group that is the order of some element, so it is the larger invariant factor. -/
def groupExponent (p a b : Nat) : Nat :=
  let n := pointCount p a b
  ((affinePoints p a b).map fun q => pointOrder p a (some q) (n + 1)).foldl Nat.lcm 1

/-- `(n1, n2)` with `n1 ∣ n2` and `E(F_p) ≅ Z/n1 × Z/n2`; `(0, 0)` for a singular pair. -/
def groupStructure (p a b : Nat) : Nat × Nat :=
  if nonsingular p a b then
    let e := groupExponent p a b
    (pointCount p a b / e, e)
  else (0, 0)

/-! ## Operations and values -/

inductive Op
  | pointCount | nonsingular | supersingular | jInvariant | isCanonical | classSize | groupStructure

/-- One materialised output per member. -/
inductive Value
  | group (invariants : Nat × Nat)
  deriving DecidableEq, Repr

/-- Apply a function of the coefficients to every member. -/
def onCurves (ms : List Mat) (g : Nat → Nat → α) : List α :=
  ms.map fun m => g (curveOf m).1 (curveOf m).2

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let p := f.p
  let ms := f.members
  if !supported p ms then .invalid else
  match op with
  | .pointCount => reduceInt red ms (onCurves ms (pointCount p))
  | .nonsingular => reduceBool red ms (onCurves ms (nonsingular p))
  | .supersingular => reduceBool red ms (onCurves ms (supersingular p))
  | .jInvariant => reduceInt red ms (onCurves ms (jInvariant p))
  | .isCanonical => reduceBool red ms (onCurves ms (isCanonical p))
  | .classSize => reduceInt red ms (onCurves ms (classSize p))
  | .groupStructure => reduceValues red (onCurves ms fun a b => .group (groupStructure p a b))

end Elliptic_curves_fp
