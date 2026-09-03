import Mathlib
import Elliptic_curves_fp.Reference
import Lk.Contract

/-!
# elliptic_curves_fp: contract

What the executable reference means in Mathlib's terms. The reference works with naturals below
`p`; here each member `(a, b)` becomes the Mathlib Weierstrass curve
`Y² = X³ + aX + b` over `ZMod p`, and every operation is stated against Mathlib's discriminant,
j-invariant, group of points and change of variables.

This is not a proof that the backend is correct; it is the exact statement of what a caller
relies on. A theorem with `sorry` is an obligation written down and not yet discharged.
-/

namespace Elliptic_curves_fp.Contract

open Lk WeierstrassCurve

variable {p : ℕ} [Fact p.Prime]

/-- The member `(a, b)` as a Mathlib Weierstrass curve `Y² = X³ + aX + b`. -/
def curve (p a b : ℕ) : WeierstrassCurve (ZMod p) := ⟨0, 0, 0, (a : ZMod p), (b : ZMod p)⟩

/-- The curve of a member is in short normal form, so this really is `Y² = X³ + aX + b`. -/
instance isShortNF (p a b : ℕ) : (curve p a b).IsShortNF := ⟨rfl, rfl, rfl⟩

/-- The affine points of the reference: solutions of `y² = x³ + ax + b` over `ZMod p`. -/
def Solutions (p a b : ℕ) : Type :=
  {q : ZMod p × ZMod p // q.2 ^ 2 = q.1 ^ 3 + (a : ZMod p) * q.1 + (b : ZMod p)}

/-! ## Singularity -/

/-- `4a³ + 27b²` is the discriminant up to the unit `-16`, so for `p > 3` it vanishes exactly on
the singular pairs. This one is small enough to prove. -/
theorem delta_eq (a b : ℕ) :
    (curve p a b).Δ = -16 * (4 * (a : ZMod p) ^ 3 + 27 * (b : ZMod p) ^ 2) := by
  simp only [curve, WeierstrassCurve.Δ, WeierstrassCurve.b₂, WeierstrassCurve.b₄,
    WeierstrassCurve.b₆, WeierstrassCurve.b₈]
  ring

theorem singularForm_spec (a b : ℕ) :
    ((singularForm p a b : ℕ) : ZMod p) = 4 * (a : ZMod p) ^ 3 + 27 * (b : ZMod p) ^ 2 := by
  sorry

/-- The reference calls a pair nonsingular exactly when Mathlib calls the curve elliptic. -/
theorem nonsingular_spec (hp : 3 < p) (a b : ℕ) :
    nonsingular p a b = true ↔ IsUnit (curve p a b).Δ := by
  sorry

/-! ## Counting points

`pointCount` counts the `F_p`-points of the projective cubic `Y²Z = X³ + aXZ² + bZ³`, whose only
point with `Z = 0` is `[0 : 1 : 0]`: the affine solutions, plus one. That is stated first, since
it holds for singular pairs too, and then identified with Mathlib's group of points where the
pair is nonsingular (Mathlib's `Point` omits the singular point of a singular cubic). -/

theorem rootCount_spec (hp : 3 < p) (v : ℕ) :
    rootCount p v = Nat.card {y : ZMod p // y ^ 2 = (v : ZMod p)} := by
  sorry

theorem pointCount_spec (hp : 3 < p) (a b : ℕ) :
    pointCount p a b = Nat.card (Solutions p a b) + 1 := by
  sorry

theorem pointCount_eq_card_point (hp : 3 < p) (a b : ℕ) (h : IsUnit (curve p a b).Δ) :
    pointCount p a b = Nat.card (curve p a b).toAffine.Point := by
  sorry

/-- For `p > 3` the Hasse bound gives `|a_p| ≤ 2√p < p`, so a vanishing trace of Frobenius is the
same as `#E(F_p) = p + 1`, which is what the reference tests. -/
theorem supersingular_spec (hp : 3 < p) (a b : ℕ) :
    supersingular p a b = true ↔
      IsUnit (curve p a b).Δ ∧ Nat.card (curve p a b).toAffine.Point = p + 1 := by
  sorry

/-! ## The j-invariant -/

/-- On a nonsingular pair the reference's representative in `0..p-1` is Mathlib's j-invariant. -/
theorem jInvariant_spec (hp : 3 < p) (a b : ℕ) [(curve p a b).IsElliptic] :
    ((jInvariant p a b : ℕ) : ZMod p) = (curve p a b).j ∧ jInvariant p a b < p := by
  sorry

/-- A singular pair has no j-invariant, and the reference says so with a value outside `F_p`. -/
theorem jInvariant_singular (a b : ℕ) (h : ¬ IsUnit (curve p a b).Δ) :
    jInvariant p a b = p := by
  sorry

/-! ## Isomorphism classes

Over a field of characteristic other than 2 and 3, two short Weierstrass curves are isomorphic
exactly when an admissible change of variables `(X, Y) ↦ (u²X, u³Y)` carries one to the other,
which is `(a, b) ↦ (u⁴a, u⁶b)`. So the reference's orbit is the isomorphism class, and its
lexicographically least element is the canonical representative. -/

theorem isoOrbit_spec (hp : 3 < p) (a b a' b' : ℕ) (ha : a < p) (hb : b < p)
    (ha' : a' < p) (hb' : b' < p) :
    (a', b') ∈ isoOrbit p a b ↔
      ∃ C : VariableChange (ZMod p), C • curve p a b = curve p a' b' := by
  sorry

/-- Exactly one pair of each isomorphism class is canonical. -/
theorem isCanonical_unique (hp : 3 < p) (a b : ℕ) (ha : a < p) (hb : b < p) :
    ∃! q : ℕ × ℕ, q ∈ isoOrbit p a b ∧ isCanonical p q.1 q.2 = true := by
  sorry

/-- The class size counted through the stabiliser is the number of distinct pairs in the class. -/
theorem classSize_spec (hp : 3 < p) (a b : ℕ) :
    classSize p a b = (isoOrbit p a b).eraseDups.length := by
  sorry

/-! ## The group of points -/

/-- `addPoints` is Mathlib's group law: the reference's `none` is the point at infinity and its
affine points are the nonsingular affine points of the curve. -/
theorem addPoints_spec (hp : 3 < p) (a b x₁ y₁ x₂ y₂ x₃ y₃ : ℕ)
    (h₁ : (curve p a b).toAffine.Nonsingular (x₁ : ZMod p) (y₁ : ZMod p))
    (h₂ : (curve p a b).toAffine.Nonsingular (x₂ : ZMod p) (y₂ : ZMod p))
    (h₃ : (curve p a b).toAffine.Nonsingular (x₃ : ZMod p) (y₃ : ZMod p))
    (h : addPoints p a (some (x₁, y₁)) (some (x₂, y₂)) = some (x₃, y₃)) :
    (Affine.Point.some _ _ h₁ + Affine.Point.some _ _ h₂ : (curve p a b).toAffine.Point) =
      Affine.Point.some _ _ h₃ := by
  sorry

/-- And the reference's point at infinity is Mathlib's zero. -/
theorem addPoints_eq_zero (hp : 3 < p) (a b x₁ y₁ x₂ y₂ : ℕ)
    (h₁ : (curve p a b).toAffine.Nonsingular (x₁ : ZMod p) (y₁ : ZMod p))
    (h₂ : (curve p a b).toAffine.Nonsingular (x₂ : ZMod p) (y₂ : ZMod p))
    (h : addPoints p a (some (x₁, y₁)) (some (x₂, y₂)) = none) :
    (Affine.Point.some _ _ h₁ + Affine.Point.some _ _ h₂ : (curve p a b).toAffine.Point) = 0 := by
  sorry

/-- The invariant factors: `E(F_p)` is `Z/n1 × Z/n2` with `n1 ∣ n2`, and the reference returns
that pair. `n2` is the exponent of the group, which in a finite abelian group is the order of one
of its elements, so it is the largest point order. -/
theorem groupStructure_spec (hp : 3 < p) (a b : ℕ) (h : IsUnit (curve p a b).Δ) :
    (groupStructure p a b).1 ∣ (groupStructure p a b).2 ∧
      (groupStructure p a b).1 * (groupStructure p a b).2 = pointCount p a b ∧
      Nonempty ((curve p a b).toAffine.Point ≃+
        (ZMod (groupStructure p a b).1 × ZMod (groupStructure p a b).2)) := by
  sorry

/-- A singular pair has no group of points, and the reference says so with `(0, 0)`. -/
theorem groupStructure_singular (a b : ℕ) (h : ¬ IsUnit (curve p a b).Δ) :
    groupStructure p a b = (0, 0) := by
  sorry

end Elliptic_curves_fp.Contract
