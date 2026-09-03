import Lk.Reference

/-!
# circulants: executable reference

A member is a connection set in `Z_n`, read by flattening its natural-number matrix. Directed
circulants use that set as written. Undirected circulants use `(S ∪ -S) \ {0}`, matching
Mathlib's `SimpleGraph.circulantGraph` convention.

The spectrum is exact rather than floating point. Character `j` has eigenvalue
`sum (zeta_n ^ e)` over `e = j*s mod n`. Its returned row lists those exponents in increasing
order, retaining repeats. Characters use increasing residue order.

At corrected-Adam orders, graph isomorphism is exactly multiplication of the connection set by
a unit of `Z_n`. The undirected exceptional CI orders 8, 9 and 18 are included. All recursion is
structural so `decide +kernel` can evaluate test cases.
-/

namespace Circulants

open Lk


def insertNat (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: ys => if x ≤ y then x :: y :: ys else y :: insertNat x ys

def sortNats (xs : List Nat) : List Nat := xs.foldr insertNat []

def dedupSorted : List Nat → List Nat
  | [] => []
  | [x] => [x]
  | x :: y :: xs => if x = y then dedupSorted (y :: xs) else x :: dedupSorted (y :: xs)

def canonicalSet (xs : List Nat) : List Nat := dedupSorted (sortNats xs)

def rawConnection (m : Mat) : List Nat := m.flatten

def validConnection (n : Nat) (s : List Nat) : Bool :=
  decide (0 < n ∧ s.Nodup ∧ 0 ∉ s ∧ ∀ x ∈ s, x < n)

def effectiveSet (n directed : Nat) (s : List Nat) : List Nat :=
  if directed = 1 then sortNats s
  else (canonicalSet (s ++ s.map fun x => (n - x) % n)).filter (· ≠ 0)

def multiplierImage (n a : Nat) (s : List Nat) : List Nat :=
  sortNats (s.map fun x => a * x % n)

def units (n : Nat) : List Nat := (List.range n).filter fun a => Nat.gcd a n = 1

def multiplierEquivalent (n : Nat) (s t : List Nat) : Bool :=
  (units n).any fun a => multiplierImage n a s = t

def multiplierCanonical (n : Nat) (s : List Nat) : Bool :=
  (units n).all fun a => lexLe s (multiplierImage n a s)

/-- Search odd divisors `d` for a square dividing `n`. The fuel starts at `n`. -/
def noOddSquareDivisor (n : Nat) : Nat → Nat → Bool
  | 0, _ => true
  | fuel + 1, d =>
    if n / d < d then true
    else if d % 2 = 1 ∧ n % (d * d) = 0 then false
    else noOddSquareDivisor n fuel (d + 1)

/-- The corrected-Adam orders: no factor 8 and no odd prime square. Checking all odd squares is
equivalent and avoids a primality subroutine. -/
def adamOrder (n : Nat) : Bool :=
  if n = 0 then false else n % 8 ≠ 0 ∧ noOddSquareDivisor n n 3

def ciOrder (n directed : Nat) : Bool :=
  adamOrder n || (directed = 0 && (n = 8 || n = 9 || n = 18))

def exponentRow (n : Nat) (s : List Nat) (j : Nat) : List Nat :=
  sortNats (s.map fun x => j * x % n)

def spectrumExponents (n directed : Nat) (s : List Nat) : List (List Nat) :=
  let eff := effectiveSet n directed s
  (List.range n).map (exponentRow n eff)

def validMode (directed : Nat) : Bool := directed = 0 || directed = 1

def validConnectionFamily (n directed : Nat) (f : Family) : Bool :=
  validMode directed && 0 < n && n < (1 <<< 32) &&
    f.members.all fun m => validConnection n (rawConnection m)

inductive Op
  | spectrum (directed n : Nat)
  | isomorphic (directed n : Nat) (target : Vec)
  | isCanonical (directed n : Nat)
  | isCi (directed : Nat)

inductive Value
  | spectrum (exponents : List (List Nat))
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let ms := f.members
  match op, red with
  | .spectrum directed n, .all =>
    if validConnectionFamily n directed f then
      .values (ms.map fun m => .spectrum (spectrumExponents n directed (rawConnection m)))
    else .invalid
  | .spectrum _ _, _ => .invalid
  | .isomorphic directed n target, _ =>
    if validConnectionFamily n directed f && ciOrder n directed && validConnection n target then
      let t := effectiveSet n directed target
      reduceBool red ms (ms.map fun m =>
        multiplierEquivalent n (effectiveSet n directed (rawConnection m)) t)
    else .invalid
  | .isCanonical directed n, _ =>
    if validConnectionFamily n directed f && ciOrder n directed then
      reduceBool red ms (ms.map fun m =>
        let raw := sortNats (rawConnection m)
        let eff := effectiveSet n directed raw
        (directed = 1 || raw = eff) && multiplierCanonical n eff)
    else .invalid
  | .isCi directed, _ =>
    match f with
    | .range _ _ =>
      if validMode directed && ms.all fun m => 0 < (rawConnection m).headD 0 then
        reduceBool red ms (ms.map fun m => ciOrder ((rawConnection m).headD 0) directed)
      else .invalid
    | _ => .invalid

end Circulants
