import Mathlib
import Polytopes_small.Reference

/-!
# polytopes_small: contract

The executable reference uses determinant signs. This file states its meaning with Mathlib's
convex hull and affine-span definitions. The h-star vector is the canonical representation of
the Ehrhart polynomial.
-/

namespace Polytopes_small.Contract

open scoped BigOperators
open Lk

noncomputable section


def point {d : Nat} (v : Vec) : Fin d → ℚ := fun i => v.getD i 0

def indexedPoint {d : Nat} (m : Mat) (i : Fin m.length) : Fin d → ℚ := point (m.getD i [])

def dot {d : Nat} (a x : Fin d → ℚ) : ℚ := ∑ i, a i * x i

/-- A nonempty exposed face, represented by all selected point indices on its supporting
hyperplane. For a polytope every face is exposed. The zero functional gives the polytope itself. -/
def IsFace {d : Nat} (m : Mat) (s : Finset (Fin m.length)) : Prop :=
  s.Nonempty ∧ ∃ a : Fin d → ℚ, ∃ c : ℚ,
    (∀ i : Fin m.length, dot a (indexedPoint m i) ≤ c) ∧
    s = Finset.univ.filter fun i => dot a (indexedPoint m i) = c

def faceDimension {d : Nat} (m : Mat) (s : Finset (Fin m.length)) : Nat :=
  Module.finrank ℚ
    (affineSpan ℚ ((indexedPoint (d := d) m) '' (s : Set (Fin m.length)))).direction

noncomputable def faceCount {d : Nat} (m : Mat) (k : Nat) : Nat := by
  classical
  exact ((Finset.univ : Finset (Fin m.length)).powerset.filter fun s =>
    IsFace (d := d) m s ∧ faceDimension (d := d) m s = k).card

def affineDimension {d : Nat} (m : Mat) : Nat :=
  Module.finrank ℚ (affineSpan ℚ (Set.range (indexedPoint (d := d) m))).direction

noncomputable def IsSimplicial {d : Nat} (m : Mat) : Prop := by
  classical
  let r := affineDimension (d := d) m
  exact ∀ s : Finset (Fin m.length), IsFace (d := d) m s → faceDimension (d := d) m s + 1 = r →
    (((Finset.univ : Finset (Fin m.length)).powerset.filter fun v =>
      IsFace (d := d) m v ∧ faceDimension (d := d) m v = 0 ∧ v ⊆ s).card = r)


def scaledPoints {d : Nat} (m : Mat) (t : Nat) : Set (Fin d → ℚ) :=
  {x | ∃ i : Fin m.length, x = fun j => (t : ℚ) * indexedPoint m i j}

def integerPoint {d : Nat} (z : Fin d → ℤ) : Fin d → ℚ := fun i => z i

def ehrhartCount {d : Nat} (m : Mat) (t : Nat) : Nat :=
  Set.ncard {z : Fin d → ℤ | integerPoint z ∈ convexHull ℚ (scaledPoints m t)}

def representsEhrhart {d : Nat} (m : Mat) (hs : List Nat) : Prop :=
  let r := affineDimension (d := d) m
  ∀ t : Nat, ehrhartCount (d := d) m t =
    ∑ j ∈ Finset.range (r + 1), hs.getD j 0 * Nat.choose (t + r - j) r


theorem affineDim_spec {m : Mat} (hm : m ≠ []) :
    (analyse m).affineDim = affineDimension (d := (m.headD []).length) m := by
  sorry

theorem vertexCount_spec {m : Mat} (hm : m ≠ []) :
    vertexCount m = faceCount (d := (m.headD []).length) m 0 := by
  sorry

theorem fVector_spec {m : Mat} (hm : m ≠ []) :
    fVector m = (List.range ((m.headD []).length + 1)).map fun k =>
      faceCount (d := (m.headD []).length) m k := by
  sorry

theorem isSimplicial_spec {m : Mat} (hm : m ≠ []) :
    isSimplicial m = true ↔ IsSimplicial (d := (m.headD []).length) m := by
  sorry

theorem ehrhartPolynomial_spec {m : Mat} (hm : m ≠ []) :
    representsEhrhart (d := (m.headD []).length) m (hStar m) := by
  sorry

end

end Polytopes_small.Contract
