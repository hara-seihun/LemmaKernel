import Lk.Reference

/-!
# heat_dirichlet: executable reference

Rigorous fixed-point upper bounds on the summands of heat-weighted Dirichlet polynomials. Every
real quantity is an integer at scale `S = 2^48`, every rounding is a floor or a ceiling chosen so
that the final value bounds the real one from above, and the algorithm is the backend's and the
naive implementation's step for step, so that the three return the same integer.

Notation, for `t = tNum / tDen`, a lower bound `sigma` on `Re s`, `y ∈ [yLo, yHi]`, cutoffs
`N ∈ [N₋, N₊]`, the mollifier primes with product `D`, and a constant `C`:

* `g(n) = exp((t/4) ln² n − sigma ln n)`, the heat weight `b_n = exp((t/4) ln² n)` times `n^{-sigma}`;
* `ρ_d(n) = exp(−(t/4) ln d (2 ln n − ln d)) = b_{n/d} / b_n` for `d ∣ n`, decreasing in `n`;
* `λ_d = ∏_{p ∣ d} (−b_p)`;
* `β(n) = ∑ λ_d ρ_d(n)` and `α(n, y) = C (n/N₋)^y ∑ λ_d d^{-y} ρ_d(n)`, both over the divisors
  `d ∣ gcd(n, D)` with `n / d ≤ N` (the polynomial is truncated at `N`);
* `r(y) = (1 − C N₋^{-y}) / (1 + C N₋^{-y})`;
* `term(n, y) = max(|β − α|, r |β + α|) g(n)`.

`weightUpper` bounds `g(n)`; `mollifiedTermUpper` bounds `term(n, y)` over the cell; `blockTermUpper`
bounds its sum over a block of consecutive `n`, using that `ρ_d` decreases in `n`, `(n/N₋)^y`
increases and `g` is quasi-convex; `sigmaLower` is the lower bound on `Re s` the others consume.

Arguments to `exp` above `7` make an operation `.invalid`, as they do in the backend, whose
128-bit words this limit protects. Members are single natural numbers (a `range` family or an
`explicit` family of 1 x 1 `lk.naturals` members); anything else is `.invalid`.
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
  yLoNum : Int
  yHiNum : Int
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
  0 ≤ P.yLoNum ∧ P.yLoNum ≤ P.yHiNum ∧ P.yHiNum ≤ P.yDen ∧
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

/-- `exp(y · base)` over `y ∈ [yLo, yHi]`. -/
def Params.ypowInterval (P : Params) (base : Iv) : Option Iv :=
  iexp (fdiv (min (base.1 * P.yLoNum) (base.1 * P.yHiNum)) P.yDen,
        cdiv (max (base.2 * P.yLoNum) (base.2 * P.yHiNum)) P.yDen)

/-- `d^{-y}` over the `y` interval. -/
def Params.divisorPowInterval (P : Params) (d : Nat) : Option Iv :=
  let lnd := lnBounds d
  P.ypowInterval (-lnd.2, -lnd.1)

/-- `(n / N₋)^y` from bounds on `ln n`. -/
def Params.ratioPowInterval (P : Params) (lnn : Iv) : Option Iv :=
  P.ypowInterval (isub lnn (lnBounds P.nMinus))

/-- Upper bound on `r = (1 − C N₋^{-y}) / (1 + C N₋^{-y})`. -/
def Params.rUpper (P : Params) : Option Int := do
  let lnN := lnBounds P.nMinus
  let w ← iexp (-(cdiv (lnN.2 * P.yHiNum) P.yDen), -(fdiv (lnN.1 * P.yLoNum) P.yDen))
  let wc := iscale w P.cNum P.cDen
  let a := max wc.1 0
  pure (if a < S then cdiv ((S - a) * S) (S + a) else 0)

/-- `max(|β − α|, r |β + α|)` over the divisor set `divs`, whose `ρ` enclosures are `rho`. -/
def Params.coefficientOf (P : Params) (divs : List Nat) (rho : List Iv) (ratio : Iv) : Option Int := do
  let pairs := divs.zip rho
  let beta ← pairs.foldlM (fun acc (d, r) => do
    let lam ← P.lambdaInterval d
    pure (iadd acc (imul lam r))) ((0, 0) : Iv)
  let alphaSum ← pairs.foldlM (fun acc (d, r) => do
    let lam ← P.lambdaInterval d
    let dp ← P.divisorPowInterval d
    pure (iadd acc (imul (imul lam dp) r))) ((0, 0) : Iv)
  let alpha := iscale (imul alphaSum ratio) P.cNum P.cDen
  let rU ← P.rUpper
  pure (max (absUpper (isub beta alpha)) (cdiv (rU * absUpper (iadd beta alpha)) S))

/-- Upper bound on the coefficient over every `n ∈ [lo, hi]` and cutoff `N ∈ [N₋, N₊]`: the
truncation keeps the upper set `{d : d ≥ n / N}` of the divisors, whose cut ranges over
`[lo / N₊, hi / N₋]`, and the bound is the largest over the upper sets whose cut lies there. -/
def Params.coefficientUpper (P : Params) (divs : List Nat) (rho : List Iv) (ratio : Iv)
    (lo hi : Nat) : Option Int :=
  (List.range (divs.length + 1)).foldlM (fun best j => do
    let below := if j = 0 then 0 else divs.getD (j - 1) 0
    let aboveOk := j = divs.length ∨ lo ≤ divs.getD j 0 * P.nPlus
    if aboveOk ∧ below * P.nMinus < hi then
      let c ← P.coefficientOf (divs.drop j) (rho.drop j) ratio
      pure (max best c)
    else pure best) 0

/-- The summand bound for one member `n`. -/
def Params.termUpper (P : Params) (n : Nat) : Option Int := do
  let divs := P.divisorsOf n
  let lnn := lnBounds n
  let rho ← divs.mapM fun d => P.rhoInterval lnn (lnBounds d)
  let ratio ← P.ratioPowInterval lnn
  let g ← P.gUpper lnn
  let coeff ← P.coefficientUpper divs rho ratio n n
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
  let rlo ← P.ratioPowInterval lna
  let rhi ← P.ratioPowInterval lnb
  let ratio : Iv := (rlo.1, rhi.2)
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
    let coeff ← P.coefficientUpper divs rho ratio a (b - 1)
    pure (total + count * cdiv (coeff * gMax) S)) 0

/-- `(1 + yLo)/2 + (t/4) ln(N² − 1) − t / (144 (N² − 1)²)`, rounded down. -/
def Params.sigmaLower (P : Params) (n : Nat) : Int :=
  let m : Int := n * n - 1
  let half := fdiv (S * (P.yDen + P.yLoNum)) (2 * P.yDen)
  let main := fdiv (P.tNum * (lnBounds (n * n - 1)).1) (4 * P.tDen)
  let corr := cdiv (S * P.tNum) (P.tDen * 144 * m * m)
  half + main - corr

/-! ## Operations -/

inductive Op
  | weightUpper (tNum tDen sigmaNum sigmaDen scale : Nat)
  | mollifiedTermUpper (tNum tDen sigmaNum sigmaDen yLoNum yHiNum yDen nMinus nPlus primes cNum cDen
      scale : Nat)
  | blockTermUpper (tNum tDen sigmaNum sigmaDen yLoNum yHiNum yDen nMinus nPlus primes cNum cDen n0
      width scale : Nat)
  | sigmaLower (tNum tDen yLoNum yHiNum yDen scale : Nat)

/-- No operation materialises a per-member value; every reduction is over integers. -/
inductive Value
  deriving DecidableEq, Repr

def Op.params : Op → Params
  | .weightUpper tNum tDen sigmaNum sigmaDen scale =>
    { tNum, tDen, sigmaNum, sigmaDen, yLoNum := 0, yHiNum := 0, yDen := 1, nMinus := 1, nPlus := 1,
      primes := [], cNum := 1, cDen := 1, n0 := 0, width := 1, scale }
  | .mollifiedTermUpper tNum tDen sigmaNum sigmaDen yLoNum yHiNum yDen nMinus nPlus primes cNum cDen scale =>
    { tNum, tDen, sigmaNum, sigmaDen, yLoNum, yHiNum, yDen, nMinus, nPlus, primes := primesOf primes,
      cNum, cDen, n0 := 0, width := 1, scale }
  | .blockTermUpper tNum tDen sigmaNum sigmaDen yLoNum yHiNum yDen nMinus nPlus primes cNum cDen n0 width scale =>
    { tNum, tDen, sigmaNum, sigmaDen, yLoNum, yHiNum, yDen, nMinus, nPlus, primes := primesOf primes,
      cNum, cDen, n0, width, scale }
  | .sigmaLower tNum tDen yLoNum yHiNum yDen scale =>
    { tNum, tDen, sigmaNum := 0, sigmaDen := 1, yLoNum, yHiNum, yDen, nMinus := 1, nPlus := 1,
      primes := [], cNum := 1, cDen := 1, n0 := 0, width := 1, scale }

/-- The numbers a family presents, or `none` when its members are not single naturals. -/
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
    match values with
    | none => .invalid
    | some vs => reduceInt red ms vs

end Heat_dirichlet
