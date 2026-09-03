import Lk.Reference

/-!
# polynomials_fq: executable reference

The meaning of every polynomials_fq operation, written as plain structural recursion so that
Lean's kernel can evaluate it. Families, reductions and the result shape come from
`Lk.Reference`. This is the oracle: a kernel answer is correct when
`example : run op family red = answer := by decide +kernel` is accepted.

A **member** is one row `[a₀, …, a_{d-1}]` of `d` entries of `F_p`, read as the monic polynomial
`x^d + a_{d-1} x^{d-1} + ⋯ + a₀`: entry `i` is the coefficient of `xⁱ` and the degree is the
number of columns. Inside, a polynomial is the full ascending coefficient list with no trailing
zero (`[]` is `0`, `[1]` is `1`, `[0, 1]` is `x`), because remainders and gcds are not monic of a
fixed degree. `polyOf` converts a member row into that form and `gcd` converts back by dropping
the leading `1`.

Everything the answers choose is chosen here: factor degrees come back sorted, roots come back
in increasing order, and a gcd is made monic.
-/

namespace Polynomials_fq

open Lk

/-! ## Polynomials as coefficient lists -/

/-- Coefficients in ascending degree with no trailing zero; `[]` is the zero polynomial. -/
abbrev Poly := List Nat

/-- Drop trailing zeros, so that the last entry of a nonzero polynomial is its leading one. -/
def trim : Poly → Poly
  | [] => []
  | a :: rest =>
    let t := trim rest
    if t.isEmpty && a == 0 then [] else a :: t

/-- Degree of a trimmed nonzero polynomial. -/
def deg (f : Poly) : Nat := f.length - 1

/-- Leading coefficient of a trimmed polynomial (`0` for the zero polynomial). -/
def lead (f : Poly) : Nat := f.getD (f.length - 1) 0

def padd (p : Nat) (a b : Poly) : Poly :=
  trim ((List.range (max a.length b.length)).map fun i => (a.getD i 0 + b.getD i 0) % p)

def psub (p : Nat) (a b : Poly) : Poly :=
  trim ((List.range (max a.length b.length)).map fun i => (a.getD i 0 + (p - b.getD i 0 % p)) % p)

def pscale (p : Nat) (c : Nat) (a : Poly) : Poly := trim (a.map fun x => x * c % p)

def pmul (p : Nat) (a b : Poly) : Poly :=
  if a.isEmpty || b.isEmpty then []
  else trim ((List.range (a.length + b.length - 1)).map fun k =>
    ((List.range (k + 1)).foldl (fun s i => s + a.getD i 0 * b.getD (k - i) 0) 0) % p)

/-- Multiplication by `x`. -/
def pshift (a : Poly) : Poly := if a.isEmpty then [] else 0 :: a

/-- `c * x^k`. -/
def monomial (c k : Nat) : Poly := if c == 0 then [] else List.replicate k 0 ++ [c]

/-- One step of long division: cancel the leading term of `r` with a multiple of `g`. -/
def divmodAux (p : Nat) (g : Poly) (ginv : Nat) : Nat → Poly → Poly → Poly × Poly
  | 0, q, r => (q, r)
  | fuel + 1, q, r =>
    if r.length < g.length then (q, r)
    else
      let m := monomial (lead r * ginv % p) (r.length - g.length)
      divmodAux p g ginv fuel (padd p q m) (psub p r (pmul p m g))

/-- Quotient and remainder of `a` by a nonzero `g`. -/
def divmod (p : Nat) (a g : Poly) : Poly × Poly :=
  let a := trim a
  if g.isEmpty then ([], a) else divmodAux p g (inv p (lead g)) (a.length + 1) [] a

def pmod (p : Nat) (a g : Poly) : Poly := (divmod p a g).2
def pdiv (p : Nat) (a g : Poly) : Poly := (divmod p a g).1

/-- Divide by the leading coefficient; the zero polynomial stays zero. -/
def monicise (p : Nat) (a : Poly) : Poly := if a.isEmpty then [] else pscale p (inv p (lead a)) a

def gcdAux (p : Nat) : Nat → Poly → Poly → Poly
  | 0, a, _ => a
  | fuel + 1, a, b => if b.isEmpty then a else gcdAux p fuel b (pmod p a b)

/-- The monic greatest common divisor; `gcd 0 0 = 0`. -/
def pgcd (p : Nat) (a b : Poly) : Poly :=
  monicise p (gcdAux p (a.length + b.length + 1) (trim a) (trim b))

/-- Horner from the leading coefficient down. -/
def peval (p : Nat) (f : Poly) (x : Nat) : Nat := f.foldr (fun c acc => (c + x * acc) % p) 0

/-! ## Factorisation by trial division

The obvious way, which the kernel can evaluate: look for a monic divisor of degree 1, then 2, and
so on. The first one found is irreducible, because a divisor of smaller degree would have been
found first. -/

/-- Every monic polynomial of degree `d`, in the order `Lk.tuples` gives their coefficients. -/
def monics (p d : Nat) : List Poly := (tuples p d).map fun t => t ++ [1]

def pdvd (p : Nat) (g f : Poly) : Bool := (pmod p f g).isEmpty

/-- A monic divisor of least degree, searching `1 ≤ d ≤ deg f / 2`; `none` when `f` is
irreducible or constant. -/
def firstFactor (p : Nat) (f : Poly) : Option Poly :=
  (List.range (f.length / 2)).findSome? fun i =>
    (monics p (i + 1)).find? fun g => pdvd p g f

def factorAux (p : Nat) : Nat → Poly → List Nat
  | 0, _ => []
  | fuel + 1, f =>
    if f.length ≤ 1 then []
    else match firstFactor p f with
      | none => [deg f]
      | some g => deg g :: factorAux p fuel (pdiv p f g)

/-- Degrees of the monic irreducible factors with multiplicity, non-decreasing. -/
def factorDegrees (p : Nat) (f : Poly) : List Nat :=
  sortBy (fun a b => decide (a ≤ b)) (factorAux p f.length f)

def irreducible (p : Nat) (f : Poly) : Bool := 2 ≤ f.length && (firstFactor p f).isNone

/-- The distinct roots in `F_p`, increasing. -/
def rootsOf (p : Nat) (f : Poly) : List Nat := (List.range p).filter fun a => peval p f a == 0

/-! ## The order of `x` -/

/-- Drop the largest power of `x` dividing `f`: the ascending list without its leading zeros. -/
def stripX : Poly → Poly
  | [] => []
  | a :: rest => if a == 0 then stripX rest else a :: rest

def orderAux (p : Nat) (g one : Poly) : Nat → Poly → Nat → Nat
  | 0, _, e => e
  | fuel + 1, cur, e => if cur == one then e else orderAux p g one fuel (pmod p (pshift cur) g) (e + 1)

/-- Writing `f = x^h g` with `g 0 ≠ 0`, the least `e ≥ 1` with `g ∣ x^e - 1`: the multiplicative
order of `x` in `F_p[x]/(g)`, and `1` when `g` is constant. -/
def polyOrder (p : Nat) (f : Poly) : Nat :=
  let g := stripX f
  orderAux p g (pmod p [1] g) (p ^ g.length + 1) (pmod p [0, 1] g) 1

/-- Irreducible, not divisible by `x`, and `x` generates the units of `F_p[x]/(f)`. -/
def primitive (p : Nat) (f : Poly) : Bool :=
  irreducible p f && f.getD 0 0 != 0 && polyOrder p f == p ^ deg f - 1

/-! ## Operations and values -/

inductive Op
  | isIrreducible | factorisationDegrees | isPrimitive | order | roots | rootCount
  | gcd (other : Vec)

/-- One materialised output per member. -/
inductive Value
  | elements (xs : List Nat)
  | degrees (ds : List Nat)
  deriving DecidableEq, Repr

/-- The monic polynomial a member row denotes: `[a₀, …, a_{d-1}]` is `x^d + ∑ aᵢ xⁱ`. -/
def polyOf (row : Vec) : Poly := row ++ [1]

/-- `order` and `is_primitive` walk the powers of `x`, so the residue ring must be small. -/
def walkable (p d : Nat) : Bool := p ^ d < 4294967296

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let p := f.p
  let ms := f.members
  if p < 2 then .invalid
  else if ms.any (fun m => m.length != 1) then .invalid
  else
    let d := (ms.headD []).headD [] |>.length
    let fs := ms.map fun m => polyOf (m.headD [])
    match op with
    | .isIrreducible => reduceBool red ms (fs.map (irreducible p))
    | .isPrimitive => if walkable p d then reduceBool red ms (fs.map (primitive p)) else .invalid
    | .order => if walkable p d then reduceInt red ms (fs.map (polyOrder p)) else .invalid
    | .rootCount => reduceInt red ms (fs.map fun g => (rootsOf p g).length)
    | .roots => reduceValues red (fs.map fun g => .elements (rootsOf p g))
    | .factorisationDegrees => reduceValues red (fs.map fun g => .degrees (factorDegrees p g))
    | .gcd other => reduceValues red (fs.map fun g => .elements (pgcd p g (polyOf other)).dropLast)

end Polynomials_fq
