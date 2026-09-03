import Mathlib
import Projective_sets.Reference
import Lk.Contract

/-!
# projective_sets: contract

The executable reference represents points by normalised nonzero coordinate rows. This file states
its predicates and counts using Mathlib subspaces, spans and linear independence. The native
backend is checked against the executable reference. The `sorry` declarations below are the
remaining proofs that the reference has these mathematical meanings.
-/

namespace Projective_sets.Contract

open Lk Lk.Contract

variable {p d : ℕ} [Fact p.Prime]

def WellFormedPointSet (p d : ℕ) (points : Mat) : Prop :=
  WellFormed p points.length d points ∧ points.Nodup ∧
    ∀ v ∈ points, v.any (· ≠ 0) = true ∧ normalise p v = v

def spanOf (p d : ℕ) (points : Mat) : Submodule (ZMod p) (Fin d → ZMod p) :=
  Submodule.span (ZMod p) (Set.range fun i : Fin points.length => toVec p d (points.getD i []))

def IsArc (p d : ℕ) (points : Mat) : Prop :=
  ∀ (x : Fin d → Fin points.length), Function.Injective x →
    LinearIndependent (ZMod p) (fun i => toVec p d (points.getD (x i) []))

noncomputable def mathLineIntersection (p d : ℕ) (line points : Mat) : ℕ := by
  classical
  exact (points.filter fun v => toVec p d v ∈ rowSpace p 2 d line).length

def IsCap (p d : ℕ) (points : Mat) : Prop :=
  ∀ line ∈ projectiveLines p d, mathLineIntersection p d line points ≤ 2

def IsBlockingSet (p d : ℕ) (points : Mat) : Prop :=
  ∀ H : Submodule (ZMod p) (Fin d → ZMod p), Module.finrank (ZMod p) H + 1 = d →
    ∃ v ∈ points, toVec p d v ∈ H

theorem projectiveLines_spec :
    ∀ H : Submodule (ZMod p) (Fin d → ZMod p), Module.finrank (ZMod p) H = 2 →
      ∃ line ∈ projectiveLines p d, rowSpace p 2 d line = H := by
  sorry

theorem lineIntersection_spec {line points : Mat}
    (hline : WellFormed p 2 d line) (hpoints : WellFormedPointSet p d points) :
    lineIntersection p line points = mathLineIntersection p d line points := by
  sorry

theorem spannedRank_spec {points : Mat} (hpoints : WellFormedPointSet p d points) :
    Gfp.rank p points = Module.finrank (ZMod p) (spanOf p d points) := by
  sorry

theorem arc_spec {points : Mat} (hpoints : WellFormedPointSet p d points) :
    isArc p points = true ↔ IsArc p d points := by
  sorry

theorem cap_spec {points : Mat} (hpoints : WellFormedPointSet p d points) :
    isCap p points = true ↔ IsCap p d points := by
  sorry

theorem blockingSet_spec {points : Mat} (hpoints : WellFormedPointSet p d points) :
    isBlockingSet p points = true ↔ IsBlockingSet p d points := by
  sorry

theorem hyperoval_spec {points : Mat} (hpoints : WellFormedPointSet p d points) :
    isHyperoval p points = true ↔ p = 2 ∧ d = 3 ∧ points.length = 4 ∧ IsCap p d points := by
  sorry

theorem ovoid_spec {points : Mat} (hpoints : WellFormedPointSet p d points) :
    isOvoid p points = true ↔ d = 4 ∧ points.length = p ^ 2 + 1 ∧ IsCap p d points := by
  sorry

theorem maxCollinear_spec {points : Mat} (hpoints : WellFormedPointSet p d points) :
    maxCollinear p points =
      (projectiveLines p d).foldl (fun best line => max best (mathLineIntersection p d line points)) 0 := by
  sorry

theorem lineCount_spec {points : Mat} (hpoints : WellFormedPointSet p d points) (target : ℕ) :
    lineCount p target points =
      ((projectiveLines p d).filter fun line => mathLineIntersection p d line points = target).length := by
  sorry

end Projective_sets.Contract
