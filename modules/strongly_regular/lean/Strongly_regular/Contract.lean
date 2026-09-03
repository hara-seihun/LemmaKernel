import Mathlib
import Strongly_regular.Reference
import Lk.Contract

/-!
# strongly_regular: contract

The executable reference uses lists and integers. This file states its meaning with Mathlib
matrices, characteristic polynomials, and real square roots. The statements are obligations;
`sorry` marks the proofs that have not yet been discharged.
-/

namespace Strongly_regular.Contract

open Matrix Lk Lk.Contract

noncomputable section

def toNatMatrix (v : Nat) (m : Mat) : Matrix (Fin v) (Fin v) Nat :=
  fun i j => (m.getD i []).getD j 0

def toRealMatrix (v : Nat) (m : Mat) : Matrix (Fin v) (Fin v) Real :=
  fun i j => ((m.getD i []).getD j 0 : Real)

def IsStronglyRegular (v k lam mu : Nat) (A : Matrix (Fin v) (Fin v) Nat) : Prop :=
  0 < k ∧ k < v - 1 ∧ A.IsSymm ∧
  (∀ i, A i i = 0) ∧
  (∀ i, (Finset.univ.filter fun j => A i j = 1).card = k) ∧
  (∀ i j, i ≠ j → A i j = 1 →
    (Finset.univ.filter fun x => A i x = 1 ∧ A j x = 1).card = lam) ∧
  (∀ i j, i ≠ j → A i j = 0 →
    (Finset.univ.filter fun x => A i x = 1 ∧ A j x = 1).card = mu)

theorem params_spec {m : Mat} {v k lam mu : Nat} (h : WellFormed 2 v v m) :
    srgParams m = some (v, k, lam, mu) ↔
      IsStronglyRegular v k lam mu (toNatMatrix v m) := by
  sorry

def signedDelta (negative magnitude : Nat) : Int :=
  if negative = 1 then -(magnitude : Int) else magnitude

def rootPlus (delta : Int) (d : Nat) : Real := ((delta : Real) + Real.sqrt d) / 2
def rootMinus (delta : Int) (d : Nat) : Real := ((delta : Real) - Real.sqrt d) / 2

def HasSpectrum {v : Nat} (A : Matrix (Fin v) (Fin v) Real)
    (k : Nat) (delta : Int) (d f g : Nat) : Prop :=
  ∀ x : Real, A.charpoly.eval x =
    (x - k) * (x - rootPlus delta d) ^ f * (x - rootMinus delta d) ^ g

theorem spectrum_spec {m : Mat} {v k lam mu negative magnitude d f g : Nat}
    (h : WellFormed 2 v v m)
    (hp : srgParams m = some (v, k, lam, mu))
    (hs : spectrumOf (v, k, lam, mu) = some (k, negative, magnitude, d, f, g)) :
    HasSpectrum (toRealMatrix v m) k (signedDelta negative magnitude) d f g := by
  sorry

def KreinInequalities (v k : Nat) (r s : Real) : Prop :=
  ((v - k - 1 : Nat) : Real) ^ 2 * ((k : Real) ^ 2 + r ^ 3) ≥
      (r + 1) ^ 3 * (k : Real) ^ 2 ∧
  ((v - k - 1 : Nat) : Real) ^ 2 * ((k : Real) ^ 2 + s ^ 3) ≥
      (s + 1) ^ 3 * (k : Real) ^ 2

theorem krein_spec {v k lam mu : Nat} (hp : spectrumOf (v, k, lam, mu) ≠ none) :
    kreinBound (v, k, lam, mu) = true ↔
      let delta : Int := (lam : Int) - (mu : Int)
      let d := (delta * delta + 4 * ((k : Int) - (mu : Int))).toNat
      KreinInequalities v k (rootPlus delta d) (rootMinus delta d) := by
  sorry

def AbsoluteBound (v f g : Nat) : Prop :=
  v ≤ f * (f + 3) / 2 ∧ v ≤ g * (g + 3) / 2

theorem absolute_bound_spec {v k lam mu negative magnitude d f g : Nat}
    (hs : spectrumOf (v, k, lam, mu) = some (k, negative, magnitude, d, f, g)) :
    absoluteBound (v, k, lam, mu) = true ↔
      mu = 0 ∨ v + lam = 2 * k ∨ AbsoluteBound v f g := by
  sorry

theorem is_srg_spec (m : Mat) :
    (srgParams m).isSome = true ↔ ∃ v k lam mu, srgParams m = some (v, k, lam, mu) := by
  sorry

end
end Strongly_regular.Contract
