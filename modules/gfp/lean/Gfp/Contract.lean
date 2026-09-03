import Mathlib
import Gfp.Reference
import Lk.Contract

/-!
# gfp: contract

What each output of `Reference.lean` means in Mathlib's terms. These are the statements a caller
relies on when they use the module. They are stated, not proved: the native backends are checked
against `Reference.lean` by `decide`, and `Reference.lean` is related to Mathlib here. A theorem
below with `sorry` is an obligation that has been written down and not yet discharged; there is
no other tooling that would make the native code "verified", and we do not claim it is.

Well-formedness, `toMatrix`, `rowSpace` and `IsRREF` come from `Lk.Contract`.
-/

namespace Gfp.Contract

open Matrix Lk Lk.Contract

variable {p rows cols : ℕ} [Fact p.Prime] {m : Mat}

/-! ## Operations -/

theorem rank_spec (h : WellFormed p rows cols m) :
    rank p m = (toMatrix p rows cols m).rank := by
  sorry

theorem nullity_spec (h : WellFormed p rows cols m) :
    nullity p m = Module.finrank (ZMod p) (LinearMap.ker (toMatrix p rows cols m).mulVecLin) := by
  sorry

theorem rref_spec (h : WellFormed p rows cols m) :
    WellFormed p rows cols (rref p m) ∧ IsRREF p rows cols (rref p m) (pivots p m) ∧
    rowSpace p rows cols (rref p m) = rowSpace p rows cols m := by
  sorry

/-- The rref of a row space is unique, so `rref` is a canonical form for row spaces. -/
theorem rref_canonical {m' : Mat} (h : WellFormed p rows cols m) (h' : WellFormed p rows cols m') :
    rowSpace p rows cols m = rowSpace p rows cols m' ↔ rref p m = rref p m' := by
  sorry

theorem nullspace_spec (h : WellFormed p rows cols m) :
    let vs := (nullspace p m).map (toVec p cols)
    LinearIndependent (ZMod p) (fun i : Fin vs.length => vs[i]) ∧
    Submodule.span (ZMod p) (Set.range fun i : Fin vs.length => vs[i]) =
      LinearMap.ker (toMatrix p rows cols m).mulVecLin := by
  sorry

theorem inSpan_spec {t : Vec} (h : WellFormed p rows cols m) (ht : WellFormedVec p cols t) :
    inSpan p m t = true ↔ toVec p cols t ∈ rowSpace p rows cols m := by
  sorry

theorem solve_spec {b : Vec} (h : WellFormed p rows cols m) (hb : WellFormedVec p rows b) :
    (∀ x, solve p m b = some x →
        (toMatrix p rows cols m).mulVec (toVec p cols x) = toVec p rows b) ∧
    (solve p m b = none → ∀ x : Fin cols → ZMod p, (toMatrix p rows cols m).mulVec x ≠ toVec p rows b) := by
  sorry

theorem inverse_spec {n : ℕ} (h : WellFormed p n n m) :
    (∀ b, inverse p m = some b → toMatrix p n n m * toMatrix p n n b = 1) ∧
    (inverse p m = none → ¬ IsUnit (toMatrix p n n m)) := by
  sorry

theorem witness_spec (h : WellFormed p rows cols m) :
    let (r, t) := witness p m
    r = rref p m ∧ IsUnit (toMatrix p rows rows t) ∧
    toMatrix p rows rows t * toMatrix p rows cols m = toMatrix p rows cols r := by
  sorry

end Gfp.Contract
