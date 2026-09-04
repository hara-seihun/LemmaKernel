import Lk.Reference

/-!
# heat_dirichlet: executable reference

Rigorous fixed-point upper bounds on the summands of heat-weighted Dirichlet polynomials. Every
real quantity is an integer at scale `S = 2^48`, every rounding is a floor or a ceiling chosen so
that the final value bounds the real one from above, and the algorithm is the backend's and the
naive implementation's step for step, so that the three return the same integer.

Notation, for `t = tNum / tDen`, a lower bound `sigma` on `Re s`, the height `y = yNum / yDen`,
cutoffs `N ∈ [N₋, N₊]`, the mollifier primes with product `D`, and a constant `C`:

* `g(n) = exp((t/4) ln² n − sigma ln n)`, the heat weight `b_n = exp((t/4) ln² n)` times `n^{-sigma}`;
* `ρ_d(n) = exp(−(t/4) ln d (2 ln n − ln d)) = b_{n/d} / b_n` for `d ∣ n`, decreasing in `n`;
* `π_d(n) = min(1, n / (d N₋))^y`, the partner sum's weight, increasing in `n`;
* `λ_d = ∏_{p ∣ d} (−b_p)`;
* `β(n) = ∑ λ_d ρ_d(n)` and `α(n) = C ∑ λ_d π_d(n) ρ_d(n)`, both over the divisors
  `d ∣ gcd(n, D)` with `n / d ≤ N` (the polynomial is truncated at `N`);
* `r = (1 − C N₋^{-y}) / (1 + C N₋^{-y})`;
* `term(n) = max(|β − α|, r |β + α|) g(n)`.

Everything is at the one height `y`; that one value covers every greater height is the caller's
argument (the partner weight at a greater height is still at most `π_d(n)`, and `Re s` grows with
the height). `weightUpper` bounds `g(n)`; `mollifiedTermUpper` bounds `term(n)` over the cutoffs;
`blockTermUpper` bounds its sum over a block of consecutive `n`, using that `ρ_d` decreases in
`n`, `π_d` increases and `g` is quasi-convex; `sigmaLower` is the lower bound on `Re s` the others
consume.

`phaseBound` is the phase-aware bound. Every `n = m k` with `m` 7-smooth and `k` coprime to
`210`; `θ = (θ₂, θ₃, θ₅, θ₇)` are the phases of `p^{-iT}`, shared by every term, and `ψ` the phase
of the partner sum. With a mollifier `E(s) = ∑ μ_d d^{-s}` on 7-smooth `d`, the mollified
polynomial is `∑_k b_k k^{-sigma} e^{iφ_k} T_k(θ, ψ)` with free rough phases `φ_k`, where

* `T_k(θ, ψ) = ∑_m [cS_{m,k} + e^{iψ} cA_{m,k}] e^{iθ(m)}`, `θ(m) = ∑_p v_p(m) θ_p`;
* `cS_{m,k} = ∑_{d ∣ m, mk/d ≤ N} μ_d b_{m/d} (m/d)^{ε_k} m^{-sigma}`, `ε_k = (t/2) ln k`;
* `cA_{m,k} = C ∑_{d ∣ m, mk/d ≤ N} μ_d (mk/(d N₋))^y b_{m/d} (m/d)^{ε_k} m^{-sigma}`;

and `dist(E f, (−∞, 0]) ≥ F(θ, ψ) − |E| Z` with
`F(θ, ψ) = dist(T₁(θ, ψ), (−∞, 0]) − ∑_{k > 1} b_k k^{-sigma} |T_k(θ, ψ)|`. The rough `k` are
grouped into the caller's bins `[K_i, K_{i+1})`; a bin costs `W_i max |T|`, `W_i = ∑ b_k k^{-sigma}`
over its rough `k`, and its coefficients are taken at the centre `k_c = ⌊√(K_a K_b)⌋` and expanded
in `ε_k` about `ε_c` to Taylor order `order`: `(m/d)^{ε_k} = (m/d)^{ε_c} ∑_r (δ ln(m/d))^r / r!`
plus a remainder, so `|T_k| ≤ ∑_r |T^{(r)}|` with `T^{(r)}` the component whose terms carry
`(δ ln j)^r / r!`, `δ` the largest `|ε_k − ε_c|` over the bin; the remainder,
`(δ ln j)^{order+1} / (order+1)! e^{δ ln j}` times each term's size, is a loss. A term whose cutoff
condition `mk/d ≤ N` depends on `k` in the bin or on `N` is hulled with `0`. Terms with `m > m0`
are not evaluated: their `ℓ¹` mass is a loss too, as is any tail component whose weighted mass is
at most `prune` (scale `2^48`). A member is a `θ` box of the grid `g2 × g3 × g5 × g7` on the
torus, as a mixed-radix index or a 1 x 4 row; `ψ` is sampled at `npsi` boxes inside the operation
and the value is the minimum over them: a lower bound on `F` over the box from the centre value,
the gradient and the second-order remainder `|e^{ix} − 1 − ix| ≤ x²/2` of every component (the
head precisely, a tail component with the first-order term by the triangle inequality
`|∂_p T| ≤ |∂_p S| + |∂_p A|`, capped by its mass; a term whose arc over the box exceeds a radian is
enclosed by its rectangle instead), minus the losses, plus `offset`, at scale `2^scale`, clamped
at `0`.

Arguments to `exp` above `7` make an operation `.invalid`, as they do in the backend, whose
128-bit words this limit protects. Members are single natural numbers (a `range` family or an
`explicit` family of 1 x 1 `lk.naturals` members; `phaseBound` also takes 1 x 4 members);
anything else is `.invalid`.
-/

namespace Heat_dirichlet

open Lk

/-! ## Fixed-point arithmetic -/

def K : Nat := 48
def S : Int := 2 ^ 48
def lnTerms : Nat := 30
def expTerms : Nat := 24
def ln2L : Int := 195103586505167
def ln2U : Int := 195103586505168
def expLimit : Int := 7 * S
def primesAll : List Nat := [2, 3, 5, 7]

/-- Floor division for a positive divisor: `Int` division rounds toward negative infinity. -/
def fdiv (a b : Int) : Int := a / b
def cdiv (a b : Int) : Int := -((-a) / b)
def rdiv (lower : Bool) (a b : Int) : Int := if lower then fdiv a b else cdiv a b

/-- Number of bits of `n`, by structural recursion on a fuel that is always enough. -/
def bitLength : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, n => if n = 0 then 0 else 1 + bitLength fuel (n / 2)

/-- `∑_{k<T} z^(2k+1)/(2k+1)` with every step rounded one way; `remaining` terms are left, the
next has index `k`. -/
def lnLoop (lower : Bool) (z2 : Int) : Nat → Nat → Int → Int → Int
  | 0, _, _, acc => acc
  | remaining + 1, k, power, acc =>
    lnLoop lower z2 remaining (k + 1) (rdiv lower (power * z2) S) (acc + rdiv lower power (2 * k + 1))

def lnMantissa (m : Int) (lower : Bool) : Int :=
  let z := rdiv lower ((m - S) * S) (m + S)
  let z2 := rdiv lower (z * z) S
  lnLoop lower z2 lnTerms 0 z 0

/-- `(lower, upper)` bounds on `ln n` at scale `S`, for `n ≥ 1`. -/
def lnBounds (n : Nat) : Int × Int :=
  let e := bitLength 128 n - 1
  let m : Int := if e ≤ K then (n <<< K) >>> e else n >>> (e - K)
  let mHi : Int := if e ≤ K then m else m + (if n % (2 ^ (e - K)) = 0 then 0 else 1)
  (2 * lnMantissa m true + e * ln2L, 2 * lnMantissa mHi false + e * ln2U + 1)

/-- Taylor terms of `exp(r/S)` from index `k`, rounded one way. -/
def expLoop (lower : Bool) (r : Int) : Nat → Nat → Int → Int → Int
  | 0, _, _, acc => acc
  | remaining + 1, k, term, acc =>
    let term' := rdiv lower (term * r) (S * k)
    expLoop lower r remaining (k + 1) term' (acc + term')

/-- Upper bound on `exp(x/S)`, or `none` past the limit. -/
def expUpper (x : Int) : Option Int :=
  if x > expLimit then none else
  let ell := if x ≥ 0 then ln2L else ln2U
  let q := fdiv x ell
  let r := x - q * ell
  let acc := expLoop false r expTerms 1 S S + 1
  some (if q ≥ 0 then acc * 2 ^ q.toNat else cdiv acc (2 ^ (-q).toNat))

/-- Lower bound on `exp(x/S)`, or `none` past the limit. -/
def expLower (x : Int) : Option Int :=
  if x > expLimit then none else
  let ell := if x ≥ 0 then ln2U else ln2L
  let q := fdiv x ell
  let r := x - q * ell
  let acc := expLoop true r expTerms 1 S S
  some (if q ≥ 0 then acc * 2 ^ q.toNat else fdiv acc (2 ^ (-q).toNat))

/-! ## Signed intervals at scale `S` -/

abbrev Iv := Int × Int

def imul (a b : Iv) : Iv :=
  let ps := [a.1 * b.1, a.1 * b.2, a.2 * b.1, a.2 * b.2]
  (fdiv (ps.foldl min (a.1 * b.1)) S, cdiv (ps.foldl max (a.1 * b.1)) S)
def iadd (a b : Iv) : Iv := (a.1 + b.1, a.2 + b.2)
def isub (a b : Iv) : Iv := (a.1 - b.2, a.2 - b.1)
def iscale (a : Iv) (num den : Int) : Iv :=
  (fdiv (min (a.1 * num) (a.2 * num)) den, cdiv (max (a.1 * num) (a.2 * num)) den)
def iexp (a : Iv) : Option Iv := do
  let lo ← expLower a.1
  let hi ← expUpper a.2
  pure (lo, hi)
def absUpper (a : Iv) : Int := max (Int.natAbs a.1) (Int.natAbs a.2)

/-! ## Parameters -/

structure Params where
  tNum : Int
  tDen : Int
  sigmaNum : Int
  sigmaDen : Int
  yNum : Int
  yDen : Int
  nMinus : Nat
  nPlus : Nat
  primes : List Nat
  cNum : Int
  cDen : Int
  n0 : Nat
  width : Nat
  scale : Nat

def Params.D (P : Params) : Nat := P.primes.foldl (· * ·) 1

def Params.valid (P : Params) : Bool :=
  0 < P.tDen ∧ 0 < P.sigmaDen ∧ 0 < P.yDen ∧ 0 < P.cDen ∧
  0 < P.tNum ∧ 2 * P.tNum ≤ P.tDen ∧
  1 ≤ P.scale ∧ P.scale ≤ K ∧
  0 ≤ P.yNum ∧ P.yNum ≤ P.yDen ∧
  1 ≤ P.nMinus ∧ 1 ≤ P.width ∧ P.nMinus ≤ P.nPlus

/-- The primes selected by the bitmask, in increasing order. -/
def primesOf (mask : Nat) : List Nat :=
  (List.range primesAll.length).filterMap fun i =>
    if (mask >>> i) % 2 = 1 then some (primesAll.getD i 0) else none

/-- Divisors of `D` (squarefree) that divide `m`, increasing. -/
def Params.divisorsOf (P : Params) (m : Nat) : List Nat :=
  (List.range (P.D + 1)).filter fun d => 0 < d ∧ P.D % d = 0 ∧ m % d = 0

/-- Round a scale-`S` upper bound up to the output scale; it must fit 64 bits. -/
def Params.out (P : Params) (v : Int) : Option Nat :=
  let o := cdiv v (2 ^ (K - P.scale))
  if 0 ≤ o ∧ o < 2 ^ 64 then some o.toNat else none

/-- Exponent of `g`: `(t/4) ln² n − sigma ln n`. -/
def Params.gExponent (P : Params) (lnn : Iv) : Iv :=
  isub (iscale (imul lnn lnn) P.tNum (4 * P.tDen)) (iscale lnn P.sigmaNum P.sigmaDen)

def Params.gUpper (P : Params) (lnn : Iv) : Option Int := expUpper (P.gExponent lnn).2

/-- `b_p = exp((t/4) ln² p)`. -/
def Params.bInterval (P : Params) (p : Nat) : Option Iv :=
  let lnp := lnBounds p
  iexp (iscale (imul lnp lnp) P.tNum (4 * P.tDen))

/-- `λ_d = ∏_{p ∣ d} (−b_p)`. -/
def Params.lambdaInterval (P : Params) (d : Nat) : Option Iv :=
  P.primes.foldlM (fun acc p => do
    if d % p = 0 then
      let b ← P.bInterval p
      pure (imul acc (-b.2, -b.1))
    else pure acc) ((S, S) : Iv)

/-- `ρ_d(n) = exp(−(t/4) ln d (2 ln n − ln d))` from bounds on `ln n` and `ln d`. -/
def Params.rhoInterval (P : Params) (lnn lnd : Iv) : Option Iv :=
  iexp (iscale (imul lnd (isub (2 * lnn.1, 2 * lnn.2) lnd)) (-P.tNum) (4 * P.tDen))

/-- `π_d(n) = min(1, n / (d N₋))^y` from bounds on `ln n` and `ln d`. -/
def Params.piInterval (P : Params) (lnn lnd : Iv) : Option Iv :=
  let base := isub (isub lnn lnd) (lnBounds P.nMinus)
  iexp (iscale (min base.1 0, min base.2 0) P.yNum P.yDen)

/-- Upper bound on `r = (1 − C N₋^{-y}) / (1 + C N₋^{-y})`. -/
def Params.rUpper (P : Params) : Option Int := do
  let lnN := lnBounds P.nMinus
  let w ← iexp (iscale (-lnN.2, -lnN.1) P.yNum P.yDen)
  let wc := iscale w P.cNum P.cDen
  let a := max wc.1 0
  pure (if a < S then cdiv ((S - a) * S) (S + a) else 0)

/-- `max(|β − α|, r |β + α|)` over the divisor set `divs`, whose `ρ` and `π` enclosures are `rho`
and `pi`. -/
def Params.coefficientOf (P : Params) (divs : List Nat) (rho pi : List Iv) : Option Int := do
  let triples := divs.zip (rho.zip pi)
  let beta ← triples.foldlM (fun acc (d, r, _) => do
    let lam ← P.lambdaInterval d
    pure (iadd acc (imul lam r))) ((0, 0) : Iv)
  let alphaSum ← triples.foldlM (fun acc (d, r, q) => do
    let lam ← P.lambdaInterval d
    pure (iadd acc (imul (imul lam q) r))) ((0, 0) : Iv)
  let alpha := iscale alphaSum P.cNum P.cDen
  let rU ← P.rUpper
  pure (max (absUpper (isub beta alpha)) (cdiv (rU * absUpper (iadd beta alpha)) S))

/-- Upper bound on the coefficient over every `n ∈ [lo, hi]` and cutoff `N ∈ [N₋, N₊]`: the
truncation keeps the upper set `{d : d ≥ n / N}` of the divisors, whose cut ranges over
`[lo / N₊, hi / N₋]`, and the bound is the largest over the upper sets whose cut lies there. -/
def Params.coefficientUpper (P : Params) (divs : List Nat) (rho pi : List Iv)
    (lo hi : Nat) : Option Int :=
  (List.range (divs.length + 1)).foldlM (fun best j => do
    let below := if j = 0 then 0 else divs.getD (j - 1) 0
    let aboveOk := j = divs.length ∨ lo ≤ divs.getD j 0 * P.nPlus
    if aboveOk ∧ below * P.nMinus < hi then
      let c ← P.coefficientOf (divs.drop j) (rho.drop j) (pi.drop j)
      pure (max best c)
    else pure best) 0

/-- The summand bound for one member `n`. -/
def Params.termUpper (P : Params) (n : Nat) : Option Int := do
  let divs := P.divisorsOf n
  let lnn := lnBounds n
  let rho ← divs.mapM fun d => P.rhoInterval lnn (lnBounds d)
  let pi ← divs.mapM fun d => P.piInterval lnn (lnBounds d)
  let g ← P.gUpper lnn
  let coeff ← P.coefficientUpper divs rho pi n n
  pure (cdiv (coeff * g) S)

/-- Members of `[a, b)` in the class `c mod D`. -/
def classCount (a b c D : Nat) : Nat := (b + D - 1 - c) / D - (a + D - 1 - c) / D

/-- The summand bound summed over block `k`: `[n0 + k·width, n0 + (k+1)·width)`. -/
def Params.blockUpper (P : Params) (k : Nat) : Option Int := do
  let a := P.n0 + k * P.width
  let b := a + P.width
  if a < 1 then none else
  let lna := lnBounds a
  let lnb := lnBounds (b - 1)
  let ga ← P.gUpper lna
  let gb ← P.gUpper lnb
  let gMax := max ga gb
  -- one coefficient per gcd class, weighted by how many members of the block fall in it
  (P.divisorsOf P.D).foldlM (fun total e => do
    let count := (List.range P.D).foldl
      (fun acc c => if Nat.gcd c P.D = e then acc + classCount a b c P.D else acc) 0
    if count = 0 then pure total else
    let divs := P.divisorsOf e
    let rho ← divs.mapM fun d => do
      let lo ← P.rhoInterval lnb (lnBounds d)
      let hi ← P.rhoInterval lna (lnBounds d)
      pure ((lo.1, hi.2) : Iv)
    let pi ← divs.mapM fun d => do
      let lo ← P.piInterval lna (lnBounds d)
      let hi ← P.piInterval lnb (lnBounds d)
      pure ((lo.1, hi.2) : Iv)
    let coeff ← P.coefficientUpper divs rho pi a (b - 1)
    pure (total + count * cdiv (coeff * gMax) S)) 0

/-- `(1 + y)/2 + (t/4) ln(N² − 1) − t / (144 (N² − 1)²)`, rounded down. -/
def Params.sigmaLower (P : Params) (n : Nat) : Int :=
  let m : Int := n * n - 1
  let half := fdiv (S * (P.yDen + P.yNum)) (2 * P.yDen)
  let main := fdiv (P.tNum * (lnBounds (n * n - 1)).1) (4 * P.tDen)
  let corr := cdiv (S * P.tNum) (P.tDen * 144 * m * m)
  half + main - corr

/-! ## phase_bound -/

def piL : Int := 884279719003555
def piU : Int := 884279719003556
def trigTerms : Nat := 13

/-- Lower bound on `sqrt(a / S)` at scale `S`, for `a ≥ 0`. -/
def sqrtLower (a : Int) : Int := Nat.sqrt (a * S).toNat
def sqrtUpper (a : Int) : Int :=
  let r := sqrtLower a
  if r * r = a * S then r else r + 1

/-- The square of an interval, tight at `0`. -/
def isq (a : Iv) : Iv :=
  let top := max (a.1 * a.1) (a.2 * a.2)
  let bottom := if a.1 ≤ 0 ∧ 0 ≤ a.2 then 0 else min (a.1 * a.1) (a.2 * a.2)
  (fdiv bottom S, cdiv top S)

def absLower (a : Iv) : Int :=
  if a.1 ≤ 0 ∧ 0 ≤ a.2 then 0 else min (Int.natAbs a.1) (Int.natAbs a.2)

/-- `x^n / n!` at scale `S` for `x ≥ 0`, rounded one way. -/
def powfact (x : Int) (n : Nat) (lower : Bool) : Int :=
  (List.range n).foldl (fun term (i : Nat) => rdiv lower (term * x) (S * ((i : Int) + 1))) S

/-- The alternating series on `[0, π/2]` with each term rounded the safe way and the first
omitted term as the tail; `odd` selects `sin`. -/
def trigBounds (x : Int) (odd : Bool) : Iv :=
  let off := if odd then 1 else 0
  let (lo, hi) := (List.range trigTerms).foldl (fun (acc : Iv) k =>
    if k % 2 = 0 then (acc.1 + powfact x (2 * k + off) true, acc.2 + powfact x (2 * k + off) false)
    else (acc.1 - powfact x (2 * k + off) false, acc.2 - powfact x (2 * k + off) true)) (0, 0)
  let tail := powfact x (2 * trigTerms + off) false
  (lo - tail, hi + tail)

def cosBounds (x : Int) : Iv := trigBounds x false
def sinBounds (x : Int) : Iv := trigBounds x true

def ineg (a : Iv) : Iv := (-a.2, -a.1)

/-- Enclosures of `cos` and `sin` of `2πJ/M`, `M` a multiple of `4`, `0 ≤ J < M`: reduce to the
first quadrant, where `cos` decreases and `sin` increases, then rotate. -/
def unitCircle (M J : Nat) : Iv × Iv :=
  let quarter := M / 4
  let q := J / quarter
  let j : Int := J % quarter
  let xLo := fdiv (2 * piL * j) M
  let xHi := cdiv (2 * piU * j) M
  let c : Iv := ((cosBounds xHi).1, (cosBounds xLo).2)
  let s : Iv := ((sinBounds xLo).1, (sinBounds xHi).2)
  if q = 0 then (c, s) else if q = 1 then (ineg s, c) else if q = 2 then (ineg c, ineg s)
  else (s, ineg c)

/-- Powers of `p` up to `limit`. -/
def powersUpTo (p limit : Nat) : List Nat :=
  ((List.range (bitLength 128 limit + 1)).foldl (fun (acc : List Nat × Nat) _ =>
    if acc.2 ≤ limit then (acc.2 :: acc.1, acc.2 * p) else acc) ([], 1)).1

/-- Powers of `p` up to `limit`, with their exponents. -/
def powersUpTo' (p limit : Nat) : List (Nat × Nat) :=
  (powersUpTo p limit).reverse.zipIdx

/-- Every `2^a 3^b 5^c 7^e ≤ limit` with its exponent vector. -/
def smoothNumbers (limit : Nat) : List (Nat × List Nat) :=
  (powersUpTo' 2 limit).flatMap fun (a, ea) => (powersUpTo' 3 limit).flatMap fun (b, eb) =>
    (powersUpTo' 5 limit).flatMap fun (c, ec) => (powersUpTo' 7 limit).filterMap fun (e, ee) =>
      let m := a * b * c * e
      if m ≤ limit then some (m, [ea, eb, ec, ee]) else none

def isRough (k : Nat) : Bool := Nat.gcd k 210 = 1

/-- One term of a component: its exponent vector, both coefficient enclosures, the half-angle it
sweeps over a box (radians at scale `S`) and that half-angle in table steps. -/
structure Term where
  v : List Nat
  cS : Iv
  cA : Iv
  vh : Int
  X : Nat

/-- A component: its weight, its second-order constant, the `ℓ¹` mass of its terms (which `|T|`
never exceeds) and the terms over the low `m ≤ m0`. -/
structure Poly where
  W : Int
  Q : Int
  mass : Int
  terms : List Term

structure PhaseParams where
  base : Params
  sigmaHiNum : Int
  g : List Nat
  npsi : Nat
  m0 : Nat
  order : Nat
  prune : Int
  offset : Int
  mollifier : List (Nat × Int × Int)
  bins : List Nat

def PhaseParams.valid (P : PhaseParams) : Bool :=
  P.base.valid ∧ P.base.sigmaNum ≤ P.sigmaHiNum ∧ P.g.length = 4 ∧ P.g.all (0 < ·) ∧
  0 < P.npsi ∧ 0 < P.m0 ∧ P.order ≤ 3 ∧ 0 ≤ P.prune ∧ 0 ≤ P.offset ∧
  P.mollifier.all (fun (d, _, den) => 1 ≤ d ∧ d ≤ P.base.nPlus ∧ 0 < den ∧
    (smoothNumbers d).any fun (m, _) => m = d) ∧
  (P.mollifier.map (·.1)).Nodup ∧
  2 ≤ P.bins.length ∧ P.bins.headD 0 = 2 ∧ P.base.nPlus < P.bins.getLastD 0 ∧
  (P.bins.zip P.bins.tail).all fun (a, b) => a < b

def PhaseParams.sigmaIv (P : PhaseParams) : Iv :=
  (fdiv (S * P.base.sigmaNum) P.base.sigmaDen, cdiv (S * P.sigmaHiNum) P.base.sigmaDen)
/-- The five grid sizes: the four of `θ` and `npsi`. -/
def PhaseParams.g5 (P : PhaseParams) : List Nat := P.g ++ [P.npsi]
def PhaseParams.M (P : PhaseParams) : Nat := 4 * P.g5.foldl Nat.lcm 1
def PhaseParams.h (P : PhaseParams) : List Int := P.g5.map fun (x : Nat) => cdiv piU (Int.ofNat x)
def PhaseParams.dmax (P : PhaseParams) : Nat := P.mollifier.foldl (fun a (d, _, _) => max a d) 1
def PhaseParams.smooth (P : PhaseParams) : List (Nat × List Nat) :=
  smoothNumbers (P.dmax * P.base.nPlus)
def PhaseParams.lnN (P : PhaseParams) : Iv := lnBounds P.base.nMinus
def PhaseParams.circle (P : PhaseParams) (J : Nat) : Iv × Iv := unitCircle P.M J
/-- `ε_k = (t/2) ln k` enclosed. -/
def PhaseParams.eps (P : PhaseParams) (k : Nat) : Iv := iscale (lnBounds k) P.base.tNum (2 * P.base.tDen)

/-- The component with weight `W` at the bin centre `kc`: coefficient enclosures over the cell's
cutoffs (a term whose cutoff condition depends on `k ∈ [ka, kb]` or on `N` is hulled with `0`),
every term times `(δ ln j)^r / r!`; terms with `m > m0` and, in the `r = 0` pass of a tail bin,
the Taylor remainder go to the returned loss. `none` past the exponential limit or when the mass
exceeds `16`; `some (none, loss)` when the component is pruned. -/
def PhaseParams.polynomial (P : PhaseParams) (kc W : Int) (ka kb : Nat) (r : Nat) (head : Bool)
    (delta : Int) : Option (Option Poly × Int) := do
  let B := P.base
  let Lk : Iv := if head then (0, 0) else lnBounds kc.toNat
  let h4 := P.h.getD 4 0
  let (terms, massLow, massHigh, rem, Q) ← P.smooth.foldlM
    (fun (acc : List Term × Int × Int × Int × Int) (m, v) => do
    let Lm := lnBounds m
    let (cS, cA, rem) ← P.mollifier.foldlM (fun (c : Iv × Iv × Int) (d, num, den) => do
      if m % d ≠ 0 then pure c else
      let j := m / d
      if j * ka > B.nPlus then pure c else
      let certain := j * kb ≤ B.nMinus
      let Lj := lnBounds j
      let eMain := isub (iadd (iscale (imul Lj Lj) B.tNum (4 * B.tDen))
                              (iscale (imul Lj Lk) B.tNum (2 * B.tDen)))
                        (imul Lm P.sigmaIv)
      let ePart := iadd eMain (iscale (isub (iadd Lj Lk) P.lnN) B.yNum B.yDen)
      let main ← iexp eMain
      let main := iscale main num den
      let part ← iexp ePart
      let part := iscale (iscale part num den) B.cNum B.cDen
      let (main, part) := (List.range r).foldl (fun (mp : Iv × Iv) (q : Nat) =>
        let den : Int := S * (Int.ofNat q + 1)
        (iscale (imul mp.1 Lj) delta den, iscale (imul mp.2 Lj) delta den))
        (main, part)
      let main := if certain then main else (min main.1 0, max main.2 0)
      let part := if certain then part else (min part.1 0, max part.2 0)
      let rem ← if r = 0 ∧ !head ∧ 1 < j then do
          let x := cdiv (delta * Lj.2) S
          let ex ← expUpper x
          let factor := cdiv (powfact x (P.order + 1) false * ex) S
          pure (c.2.2 + cdiv ((absUpper main + absUpper part) * factor) S)
        else pure c.2.2
      pure (iadd c.1 main, iadd c.2.1 part, rem)) ((0, 0), (0, 0), acc.2.2.2.1)
    let (terms, massLow, massHigh, _, Q) := acc
    if cS = (0, 0) ∧ cA = (0, 0) then pure (terms, massLow, massHigh, rem, Q) else
    let aS := absUpper cS
    let aA := absUpper cA
    if m > P.m0 then pure (terms, massLow, massHigh + aS + aA, rem, Q) else
    let vh := ((v.zip P.h).map fun (vp, hp) => vp * hp).sum
    let X := ((v.zip P.g).map fun (vp, gx) => vp * (P.M / (2 * gx))).sum
    let Q := Q + (if vh ≤ S then cdiv (aS * cdiv (vh * vh) S) S else 0)
      + (if vh + h4 ≤ S then cdiv (aA * cdiv ((vh + h4) * (vh + h4)) S) S else 0)
    pure ({ v, cS, cA, vh, X } :: terms, massLow + aS + aA, massHigh, rem, Q)) ([], 0, 0, 0, 0)
  if massLow + massHigh > 16 * S then none else
  let loss := cdiv (W * (massHigh + rem)) S
  if !head ∧ cdiv (W * massLow) S ≤ P.prune then pure (none, loss + cdiv (W * massLow) S) else
  pure (some { W, Q := cdiv Q 2, mass := massLow, terms := terms.reverse }, loss)

/-- Enclosures of `cos` and `sin` over the arc of table steps `[J − X, J + X]`: an extremum lies
inside when the arc holds a step congruent to it, otherwise the ends are extreme. -/
def PhaseParams.arc (P : PhaseParams) (J X : Nat) : Iv × Iv :=
  let M : Int := P.M
  if 2 * X ≥ P.M then ((-S, S), (-S, S)) else
  let a : Int := (J : Int) - X
  let b : Int := (J : Int) + X
  let contains (r : Int) : Bool := fdiv (b - r) M ≥ cdiv (a - r) M
  let (ca, sa) := P.circle (a % M).toNat
  let (cb, sb) := P.circle (b % M).toNat
  let cosHi := if contains 0 then S else max ca.2 cb.2
  let cosLo := if contains (M / 2) then -S else min ca.1 cb.1
  let sinHi := if contains (M / 4) then S else max sa.2 sb.2
  let sinLo := if contains (3 * M / 4) then -S else min sa.1 sb.1
  ((cosLo, cosHi), (sinLo, sinHi))

/-- The `θ` box of a member: a mixed-radix index, or four coordinates. -/
def PhaseParams.boxOf (P : PhaseParams) (member : List Nat) : Option (List Nat) :=
  match member with
  | [i] =>
    let (js, rest) := P.g.reverse.foldl (fun (acc : List Nat × Nat) gx => (acc.2 % gx :: acc.1, acc.2 / gx)) ([], i)
    if rest = 0 then some js else none
  | js =>
    if js.length = 4 ∧ (js.zip P.g).all (fun (j, gx) => j < gx) then some js else none

/-- A complex interval and its four `θ` gradient components. -/
structure Part where
  T : Iv × Iv
  G : List (Iv × Iv)

/-- The `S` and `A` parts of a component on the `θ` box: centre values and gradients, the terms
whose arc over the box exceeds a radian enclosed by their rectangle (with no gradient). -/
def PhaseParams.evaluate (P : PhaseParams) (poly : Poly) (js : List Nat) : Part × Part :=
  let steps := (js.zip P.g).map fun (j, gx) => (2 * j + 1) * (P.M / (2 * gx))
  let h4 := P.h.getD 4 0
  let one (e : Part) (term : Term) (c : Iv) (J : Nat) (partner : Bool) : Part :=
    if c = (0, 0) then e else
    let taylor := if partner then term.vh + h4 ≤ S else term.vh ≤ S
    if !taylor then
      let (cs, sn) := P.arc J (if partner then term.X + P.M / (2 * P.npsi) else term.X)
      { e with T := (iadd e.T.1 (imul c cs), iadd e.T.2 (imul c sn)) }
    else
      let (cs, sn) := P.circle J
      let wre := imul c cs
      let wim := imul c sn
      let G := (List.range 4).map fun p =>
        let (g0, g1) := e.G.getD p ((0, 0), (0, 0))
        let vp : Int := term.v.getD p 0
        if vp = 0 then (g0, g1)
        else (iadd g0 (-vp * wim.2, -vp * wim.1), iadd g1 (vp * wre.1, vp * wre.2))
      { T := (iadd e.T.1 wre, iadd e.T.2 wim), G := G }
  let zero : Part := { T := ((0, 0), (0, 0)), G := List.replicate 4 ((0, 0), (0, 0)) }
  poly.terms.foldl (fun (e : Part × Part) term =>
    let J := ((term.v.zip steps).map fun (vp, sp) => vp * sp).sum % P.M
    (one e.1 term term.cS J false, one e.2 term term.cA J true)) (zero, zero)

def modUpper (z : Iv × Iv) : Int :=
  let ar := absUpper z.1
  let ai := absUpper z.2
  sqrtUpper (cdiv (ar * ar) S + cdiv (ai * ai) S)

/-- `z e^{iψ}` for the enclosure `(cs, sn)` of `e^{iψ}`. -/
def rotate (z : Iv × Iv) (cs sn : Iv) : Iv × Iv :=
  (isub (imul z.1 cs) (imul z.2 sn), iadd (imul z.1 sn) (imul z.2 cs))

/-- `e^{iψ}` at the centre of the `ψ` box `jpsi`. -/
def PhaseParams.psiCircle (P : PhaseParams) (jpsi : Nat) : Iv × Iv :=
  P.circle ((2 * jpsi + 1) * (P.M / (2 * P.npsi)))

/-- `(upper bound on |T|, lower bound on dist(T, (−∞, 0]))` of a component over the `θ` box and
the `ψ` box, precisely: the five gradient components of `T = S + e^{iψ} A` are
`∂_p S + e^{iψ} ∂_p A` and `i e^{iψ} A`. -/
def PhaseParams.bounds (P : PhaseParams) (poly : Poly) (parts : Part × Part) (jpsi : Nat) :
    Int × Int :=
  let (cs, sn) := P.psiCircle jpsi
  let Ar := rotate parts.2.T cs sn
  let Tre := iadd parts.1.T.1 Ar.1
  let Tim := iadd parts.1.T.2 Ar.2
  let G := ((List.range 4).map fun p =>
    let gs := parts.1.G.getD p ((0, 0), (0, 0))
    let ga := rotate (parts.2.G.getD p ((0, 0), (0, 0))) cs sn
    (iadd gs.1 ga.1, iadd gs.2 ga.2)) ++ [((-Ar.2.2, -Ar.2.1), Ar.1)]
  let hg := P.h.zip G
  let A := iadd (isq Tre) (isq Tim)
  let B := (hg.map fun (h, g) => cdiv (h * absUpper (iadd (imul Tre g.1) (imul Tim g.2))) S).sum
  let Cc := (hg.map fun (h, g) => cdiv (h * modUpper g) S).sum
  let upper := min (sqrtUpper (A.2 + 2 * B + cdiv (Cc * Cc) S) + poly.Q) poly.mass
  let lowerMod := sqrtLower (max 0 (A.1 - 2 * B)) - poly.Q
  let reMargin := Tre.1 - (hg.map fun (h, g) => cdiv (h * absUpper g.1) S).sum - poly.Q
  let imLower := absLower Tim - (hg.map fun (h, g) => cdiv (h * absUpper g.2) S).sum - poly.Q
  let dist := max 0 (max imLower (if reMargin ≥ 0 then lowerMod else 0))
  (upper, dist)

/-- Upper bound on `|T|` of a tail component over the `θ` box and the `ψ` box, the first-order
term by the triangle inequality so that only the centre value depends on `ψ`:
`|T(c)| + ∑_p h_p (|∂_p S| + |∂_p A|) + h_ψ |A(c)| + Q`, capped by the mass. -/
def PhaseParams.tailUpper (P : PhaseParams) (poly : Poly) (parts : Part × Part) (jpsi : Nat) : Int :=
  let (cs, sn) := P.psiCircle jpsi
  let Ar := rotate parts.2.T cs sn
  let T := (iadd parts.1.T.1 Ar.1, iadd parts.1.T.2 Ar.2)
  let first := ((List.range 4).map fun p =>
    cdiv (P.h.getD p 0 * (modUpper (parts.1.G.getD p ((0, 0), (0, 0))) + modUpper (parts.2.G.getD p ((0, 0), (0, 0))))) S).sum
    + cdiv (P.h.getD 4 0 * modUpper parts.2.T) S
  min (modUpper T + first + poly.Q) poly.mass

/-- The head and the bins' components with the total loss, computed once per request: for each
bin its rough range `[ka, kb]`, weight `W`, centre `kc` and `δ`, then one component per Taylor
order. -/
def PhaseParams.polys (P : PhaseParams) : Option (Poly × List Poly × Int) := do
  let (head, loss) ← P.polynomial 1 S 1 1 0 true 0
  let head ← head
  let (tail, loss) ← (P.bins.zip P.bins.tail).foldlM (fun (acc : List Poly × Int) (lo, hi) => do
    let ks := ((List.range (min hi (P.base.nPlus + 1) - lo)).map (lo + ·)).filter isRough
    match ks with
    | [] => pure acc
    | ka :: _ =>
      let kb := ks.getLastD ka
      let W ← ks.foldlM (fun w k => do
        let g ← P.base.gUpper (lnBounds k)
        pure (w + g)) (0 : Int)
      let kc := Nat.sqrt (ka * kb)
      let ec := P.eps kc
      let ea := P.eps ka
      let eb := P.eps kb
      let delta := max (max (ec.2 - ea.1) (eb.2 - ec.1)) 0
      (List.range (P.order + 1)).foldlM (fun (acc : List Poly × Int) r => do
        let (poly, l) ← P.polynomial kc W ka kb r false delta
        pure (match poly with | some q => q :: acc.1 | none => acc.1, acc.2 + l)) acc) ([], loss)
  if loss > 16 * S then none else
  pure (head, tail.reverse, loss)

def PhaseParams.phaseBound (P : PhaseParams) (head : Poly) (tail : List Poly) (loss : Int)
    (member : List Nat) : Option Nat := do
  let js ← P.boxOf member
  let hp := P.evaluate head js
  let tp := tail.map fun poly => (poly, P.evaluate poly js)
  let Fs := (List.range P.npsi).map fun jpsi =>
    tp.foldl (fun F (poly, parts) => F - cdiv (poly.W * P.tailUpper poly parts jpsi) S)
      (P.bounds head hp jpsi).2
  let best := Fs.foldl min (Fs.headD 0)
  let v := fdiv (best - loss + P.offset * S) (2 ^ (K - P.base.scale))
  if v < 0 then some 0 else if v ≥ 2 ^ 64 then none else some v.toNat

/-- The mollifier rows `(d, sign, num, den)` as `(d, ±num, den)`; `none` for a malformed row. -/
def readMollifier (rows : List (List Nat)) : Option (List (Nat × Int × Int)) :=
  rows.mapM fun row => match row with
    | [d, sign, num, den] =>
      if sign = 0 then some (d, (num : Int), (den : Int))
      else if sign = 1 then some (d, -(num : Int), (den : Int)) else none
    | _ => none

/-! ## Operations -/

inductive Op
  | weightUpper (tNum tDen sigmaNum sigmaDen scale : Nat)
  | mollifiedTermUpper (tNum tDen sigmaNum sigmaDen yNum yDen nMinus nPlus primes cNum cDen scale : Nat)
  | blockTermUpper (tNum tDen sigmaNum sigmaDen yNum yDen nMinus nPlus primes cNum cDen n0 width
      scale : Nat)
  | sigmaLower (tNum tDen yNum yDen scale : Nat)
  | phaseBound (tNum tDen sigmaNum sigmaHiNum sigmaDen yNum yDen nMinus nPlus cNum cDen
      g2 g3 g5 g7 npsi m0 order prune offset scale : Nat) (mollifier bins : List (List Nat))

/-- No operation materialises a per-member value; every reduction is over integers. -/
inductive Value
  deriving DecidableEq, Repr

def Op.params : Op → Params
  | .weightUpper tNum tDen sigmaNum sigmaDen scale =>
    { tNum, tDen, sigmaNum, sigmaDen, yNum := 0, yDen := 1, nMinus := 1, nPlus := 1,
      primes := [], cNum := 1, cDen := 1, n0 := 0, width := 1, scale }
  | .mollifiedTermUpper tNum tDen sigmaNum sigmaDen yNum yDen nMinus nPlus primes cNum cDen scale =>
    { tNum, tDen, sigmaNum, sigmaDen, yNum, yDen, nMinus, nPlus, primes := primesOf primes,
      cNum, cDen, n0 := 0, width := 1, scale }
  | .blockTermUpper tNum tDen sigmaNum sigmaDen yNum yDen nMinus nPlus primes cNum cDen n0 width scale =>
    { tNum, tDen, sigmaNum, sigmaDen, yNum, yDen, nMinus, nPlus, primes := primesOf primes,
      cNum, cDen, n0, width, scale }
  | .sigmaLower tNum tDen yNum yDen scale =>
    { tNum, tDen, sigmaNum := 0, sigmaDen := 1, yNum, yDen, nMinus := 1, nPlus := 1,
      primes := [], cNum := 1, cDen := 1, n0 := 0, width := 1, scale }
  | .phaseBound tNum tDen sigmaNum _ sigmaDen yNum yDen nMinus nPlus cNum cDen _ _ _ _ _ _ _ _ _ scale _ _ =>
    { tNum, tDen, sigmaNum, sigmaDen, yNum, yDen, nMinus, nPlus, primes := [], cNum, cDen,
      n0 := 0, width := 1, scale }

/-- The phase-aware request, or `none` when it is not one. -/
def Op.phaseParams : Op → Option PhaseParams
  | op@(.phaseBound _ _ _ sigmaHiNum _ _ _ _ _ _ _ g2 g3 g5 g7 npsi m0 order prune offset _ mollifier bins) => do
    let mollifier ← readMollifier mollifier
    let bins ← match bins with | [row] => some row | _ => none
    pure { base := op.params, sigmaHiNum, g := [g2, g3, g5, g7], npsi, m0, order, prune, offset,
           mollifier, bins }
  | _ => none

/-- The boxes a family presents: single naturals or rows of four. -/
def boxes? : Family → Option (List (List Nat))
  | .range a b => some ((List.range (b - a)).map fun i => [a + i])
  | .explicit p batch =>
    if p = Lk.naturals ∧ batch.all fun m => m.length = 1 ∧ ((m.headD []).length = 1 ∨ (m.headD []).length = 4) then
      some (batch.map fun m => m.headD [])
    else none
  | _ => none

/-- The numbers a family presents, or `none` when its members are not single naturals. -/
def numbers? : Family → Option (List Nat)
  | .range a b => some ((List.range (b - a)).map (a + ·))
  | .explicit p batch =>
    if p = Lk.naturals ∧ batch.all fun m => m.length = 1 ∧ (m.headD []).length = 1 then
      some (batch.map fun m => (m.headD []).headD 0)
    else none
  | _ => none

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match op with
  | .phaseBound .. =>
    match op.phaseParams, boxes? f with
    | some P, some bs =>
      if !P.valid then .invalid else
      match P.polys with
      | none => .invalid
      | some (head, tail, loss) =>
        match bs.mapM (P.phaseBound head tail loss) with
        | none => .invalid
        | some vs => reduceInt red f.members vs
    | _, _ => .invalid
  | _ =>
  match numbers? f with
  | none => .invalid
  | some ns =>
    let P := op.params
    if !P.valid then .invalid else
    let ms := f.members
    let values : Option (List Nat) :=
      match op with
      | .weightUpper .. =>
        if ns.any (· < 1) then none else
        ns.mapM fun n => P.gUpper (lnBounds n) >>= P.out
      | .mollifiedTermUpper .. =>
        if ns.any (· < 1) then none else
        ns.mapM fun n => P.termUpper n >>= P.out
      | .blockTermUpper .. => ns.mapM fun k => P.blockUpper k >>= P.out
      | .sigmaLower .. =>
        if ns.any (· < 2) then none else
        some (ns.map fun n => (max (P.sigmaLower n) 0).toNat >>> (K - P.scale))
      | .phaseBound .. => none
    match values with
    | none => .invalid
    | some vs => reduceInt red ms vs

end Heat_dirichlet
