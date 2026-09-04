import Mathlib
import Heat_dirichlet.Reference
import Lk.Contract

/-!
# heat_dirichlet: contract

What each output of `Reference.lean` bounds, in Mathlib's reals. A caller relies on these
inequalities: an integer `v` returned at scale `2^scale` satisfies `real quantity ≤ v / 2^scale`
(or `≥` for `sigmaLower`). They are stated, not proved: the backend is checked against
`Reference.lean` by `decide`, and `Reference.lean` is related to the reals here. A theorem with
`sorry` is an obligation written down and not yet discharged; nothing makes the native code
"verified", and we do not claim it is.

The intended proofs are the elementary directed-rounding arguments: each floor or ceiling moves in
the safe direction, the atanh series for `ln` and the Taylor series for `exp` are alternating-free
with explicit tails (`z < 1/3` after the binary reduction, `r < ln 2` after the power-of-two
reduction), and the cell bounds follow from monotonicity in `n` and the cutoff `N`.
-/

namespace Heat_dirichlet.Contract

open Real

noncomputable section

/-- The real value a scale-`S` integer stands for. -/
def real (v : ℤ) : ℝ := (v : ℝ) / (2 : ℝ) ^ K

/-! ## The primitives -/

theorem lnBounds_spec (n : ℕ) (hn : 1 ≤ n) :
    real (lnBounds n).1 ≤ log n ∧ log n ≤ real (lnBounds n).2 := by
  sorry

theorem expUpper_spec (x v : ℤ) (h : expUpper x = some v) : exp (real x) ≤ real v := by
  sorry

theorem expLower_spec (x v : ℤ) (h : expLower x = some v) : real v ≤ exp (real x) := by
  sorry

/-! ## The quantities of the module, as real functions -/

variable (P : Params)

/-- `t`, `sigma`, `C` and the height `y` as reals. -/
def t : ℝ := (P.tNum : ℝ) / P.tDen
def sigma : ℝ := (P.sigmaNum : ℝ) / P.sigmaDen
def C : ℝ := (P.cNum : ℝ) / P.cDen
def y : ℝ := (P.yNum : ℝ) / P.yDen

/-- The heat weight `b_n = exp((t/4) ln² n)`. -/
def b (n : ℕ) : ℝ := exp (t P / 4 * log n ^ 2)

/-- `g(n) = b_n n^(-sigma)`. -/
def g (n : ℕ) : ℝ := exp (t P / 4 * log n ^ 2 - sigma P * log n)

/-- `λ_d = ∏_{p ∣ d} (−b_p)` over the chosen primes. -/
def lam (d : ℕ) : ℝ := ((P.primes.filter fun p => d % p = 0).map fun p => -b P p).prod

/-- `ρ_d(n) = b_{n/d} / b_n`, written for every real `n` as `exp(−(t/4) ln d (2 ln n − ln d))`. -/
def rho (n d : ℕ) : ℝ := exp (-(t P / 4) * log d * (2 * log n - log d))

/-- `π_d(n) = min(1, n / (d N₋))^y`. -/
def pi' (n d : ℕ) : ℝ := (min 1 ((n : ℝ) / (d * P.nMinus))) ^ y P

/-- The divisors of `gcd(n, D)` that survive truncation at the cutoff `N`: `n / d ≤ N`. -/
def survivors (n N : ℕ) : List ℕ := (P.divisorsOf n).filter fun d => n ≤ d * N

/-- `β_n / b_n` for the cutoff `N`. -/
def beta (n N : ℕ) : ℝ := ((survivors P n N).map fun d => lam P d * rho P n d).sum

/-- `N₋^{-y} α_n / b_n` for the cutoff `N`. -/
def alpha (n N : ℕ) : ℝ :=
  C P * ((survivors P n N).map fun d => lam P d * pi' P n d * rho P n d).sum

/-- `r = (1 − C N₋^{-y}) / (1 + C N₋^{-y})`. -/
def r : ℝ := (1 - C P * (P.nMinus : ℝ) ^ (-y P)) / (1 + C P * (P.nMinus : ℝ) ^ (-y P))

/-- The `n`-th summand of the mollified polynomial bound, at cutoff `N`. -/
def term (n N : ℕ) : ℝ :=
  max |beta P n N - alpha P n N| (r P * |beta P n N + alpha P n N|) * g P n

/-- The cell: cutoffs in `[N₋, N₊]`. -/
def InCell (N : ℕ) : Prop := P.nMinus ≤ N ∧ N ≤ P.nPlus

/-! ## The operations -/

theorem gUpper_spec (n : ℕ) (hn : 1 ≤ n) (v : ℤ) (h : P.gUpper (lnBounds n) = some v) :
    g P n ≤ real v := by
  sorry

theorem termUpper_spec (hP : P.valid = true) (n : ℕ) (hn : 1 ≤ n) (v : ℤ) (h : P.termUpper n = some v)
    (N : ℕ) (hc : InCell P N) :
    term P n N ≤ real v := by
  sorry

theorem blockUpper_spec (hP : P.valid = true) (k : ℕ) (v : ℤ) (h : P.blockUpper k = some v)
    (N : ℕ) (hc : InCell P N) :
    ∑ n ∈ Finset.Ico (P.n0 + k * P.width) (P.n0 + (k + 1) * P.width), term P n N ≤ real v := by
  sorry

/-- `sigmaLower` is below `(1 + y)/2 + (t/4) ln(N² − 1) − t / (144 (N² − 1)²)`, which is itself
below `Re s_*` on every `x ≥ x_N = 4πN² − πt/4` at heights `y` and above (that last step is the
model's `Re s_*` bound and belongs to the caller). -/
theorem sigmaLower_spec (n : ℕ) (hn : 2 ≤ n) :
    real (P.sigmaLower n) ≤ (1 + y P) / 2 + t P / 4 * log ((n : ℝ) ^ 2 - 1)
      - t P / (144 * ((n : ℝ) ^ 2 - 1) ^ 2) := by
  sorry

/-! ## The phase-aware bound -/

section Phase

variable (Q : PhaseParams)

/-- The mollifier `μ_d` as a real. -/
def mu (d : ℕ) : ℝ :=
  match Q.mollifier.find? (·.1 = d) with
  | some (_, num, den) => (num : ℝ) / den
  | none => 0

/-- `ε_k = (t/2) ln k`. -/
def eps (k : ℕ) : ℝ := t Q.base / 2 * log k

/-- The 7-smooth divisors `d` of `m` that survive the truncation `m k / d ≤ N`. -/
def smoothSurvivors (m k N : ℕ) : List ℕ :=
  ((Q.smooth.map (·.1)).filter fun d => m % d = 0 ∧ m * k ≤ d * N)

/-- `cS_{m,k}` at cutoff `N`: `∑ μ_d b_{m/d} (m/d)^{ε_k} m^{-sigma}`. -/
def cS (m k N : ℕ) : ℝ :=
  ((smoothSurvivors Q m k N).map fun d =>
    mu Q d * b Q.base (m / d) * ((m / d : ℕ) : ℝ) ^ eps Q k).sum * (m : ℝ) ^ (-sigma Q.base)

/-- `cA_{m,k}` at cutoff `N`: the partner coefficient with the weight `(mk/(d N₋))^y`. -/
def cA (m k N : ℕ) : ℝ :=
  C Q.base * ((smoothSurvivors Q m k N).map fun d =>
    mu Q d * ((m * k : ℕ) / (d * Q.base.nMinus : ℕ) : ℝ) ^ y Q.base * b Q.base (m / d)
      * ((m / d : ℕ) : ℝ) ^ eps Q k).sum * (m : ℝ) ^ (-sigma Q.base)

/-- `θ(m) = ∑_p v_p(m) θ_p` for the smooth `m` with exponent vector `v`. -/
def thetaOf (v : List ℕ) (θ : Fin 4 → ℝ) : ℝ := ∑ i : Fin 4, (v.getD i.val 0 : ℝ) * θ i

/-- `T_k(θ, ψ)` at cutoff `N`. -/
def T (k N : ℕ) (θ : Fin 4 → ℝ) (ψ : ℝ) : ℂ :=
  (Q.smooth.map fun (m, v) =>
    ((cS Q m k N : ℂ) + Complex.exp (ψ * Complex.I) * cA Q m k N) * Complex.exp (thetaOf v θ * Complex.I)).sum

/-- The rough `k` of the cell, `1 < k ≤ N₊`, coprime to `210`. -/
def roughs : List ℕ := ((List.range (Q.base.nPlus + 1)).filter fun k => 1 < k ∧ isRough k)

/-- `F(θ, ψ) = dist(T₁, (−∞, 0]) − ∑_{k > 1} b_k k^{-sigma} |T_k|`, at cutoff `N` and the real
part `s` of the variable (`sigma` bounds it from below across the cell). -/
def F (N : ℕ) (θ : Fin 4 → ℝ) (ψ : ℝ) : ℝ :=
  Metric.infDist (T Q 1 N θ ψ) {z : ℂ | z.im = 0 ∧ z.re ≤ 0}
    - ((roughs Q).map fun k => b Q.base k * (k : ℝ) ^ (-sigma Q.base) * ‖T Q k N θ ψ‖).sum

/-- The `θ` box of the torus a member names: `θ_p ∈ 2π [j_p, j_p + 1] / g_p`. -/
def InBox (js : List ℕ) (θ : Fin 4 → ℝ) : Prop :=
  ∀ i : Fin 4, 2 * π * (js.getD i 0 : ℝ) / (Q.g.getD i 1 : ℝ) ≤ θ i ∧
    θ i ≤ 2 * π * ((js.getD i 0 : ℝ) + 1) / (Q.g.getD i 1 : ℝ)

/-- The value returned for a box, minus the offset, is below `F` at every point of the box, every
`ψ`, every cutoff of the cell, and every real part `sigma'` in `[sigma, sigmaHi]` (the reference
is evaluated with `sigma` enclosed by that interval; the statement fixes the lower end for
readability and the intended proof carries the interval). The clamp at `0` only weakens the
bound. -/
theorem phaseBound_spec (hQ : Q.valid = true) (head : Poly) (tail : List Poly) (loss : ℤ)
    (hp : Q.polys = some (head, tail, loss)) (member js : List ℕ) (hj : Q.boxOf member = some js)
    (o : ℕ) (h : Q.phaseBound head tail loss member = some o) (N : ℕ) (hc : InCell Q.base N)
    (θ : Fin 4 → ℝ) (ψ : ℝ) (hb : InBox Q js θ) :
    (o : ℝ) / (2 : ℝ) ^ Q.base.scale - (Q.offset : ℝ) ≤ F Q N θ ψ := by
  sorry

end Phase

/-- The value returned at the output scale bounds the scale-`S` value from above. -/
theorem out_spec (v : ℤ) (o : ℕ) (h : P.out v = some o) :
    real v ≤ (o : ℝ) / (2 : ℝ) ^ P.scale := by
  sorry

end

end Heat_dirichlet.Contract
