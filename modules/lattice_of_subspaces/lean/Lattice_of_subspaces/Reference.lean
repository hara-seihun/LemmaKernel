import Gfp.Reference

/-!
# lattice_of_subspaces: executable reference

Members of matrix families denote their row spaces. Natural-number range members supply `k` to
`gaussianBinomial`, and word members supply a dimension sequence to `flagCount`. The definitions
use structural recursion so Lean's kernel can evaluate the test oracle directly.
-/

namespace Lattice_of_subspaces

open Lk

/-- The Gaussian binomial `[n choose k]_q`, using its Pascal recurrence. -/
def gaussianBinomial (q : Nat) : (n k : Nat) → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 =>
      gaussianBinomial q n k + q ^ (k + 1) * gaussianBinomial q n (k + 1)

/-- Whether consecutive entries increase strictly. The empty word and a one-entry word qualify. -/
def strictlyIncreasingFrom : Option Nat → List Nat → Bool
  | _, [] => true
  | none, d :: ds => strictlyIncreasingFrom (some d) ds
  | some previous, d :: ds => previous < d && strictlyIncreasingFrom (some d) ds

def strictlyIncreasing (dims : List Nat) : Bool := strictlyIncreasingFrom none dims

/-- Product of quotient-space choices for a valid dimension sequence. -/
def flagProduct (q n : Nat) : Nat → List Nat → Nat
  | _, [] => 1
  | previous, d :: ds =>
      gaussianBinomial q (n - previous) (d - previous) * flagProduct q n d ds

/-- Number of strict flags with the given dimensions in an `n`-space. Invalid dimension
sequences have value zero. -/
def flagCount (q n : Nat) (dims : List Nat) : Nat :=
  if strictlyIncreasing dims && dims.all (fun d => d ≤ n)
  then flagProduct q n 0 dims
  else 0

/-- All rows have the ambient width. -/
def wellShaped (cols : Nat) (m : Mat) : Bool := m.all fun row => row.length = cols

/-- Whether `rowSpace(subspace) ≤ rowSpace(member)`. -/
def contains (p : Nat) (member subspace : Mat) : Bool :=
  Gfp.rank p (member ++ subspace) = Gfp.rank p member

/-- Whether `rowSpace(member) ≤ rowSpace(subspace)`. -/
def isContainedIn (p : Nat) (member subspace : Mat) : Bool :=
  Gfp.rank p (subspace ++ member) = Gfp.rank p subspace

/-- Number of `h`-subspaces below the member's row space. -/
def containedSubspaceCount (p : Nat) (member : Mat) (h : Nat) : Nat :=
  gaussianBinomial p (Gfp.rank p member) h

/-- Number of `h`-subspaces above the member's row space in its ambient coordinate space. -/
def containingSubspaceCount (p : Nat) (member : Mat) (h : Nat) : Nat :=
  let r := Gfp.rank p member
  if r ≤ h then gaussianBinomial p ((member.headD []).length - r) (h - r) else 0

inductive Op
  | gaussianBinomial (n q : Nat)
  | flagCount (n q : Nat)
  | containedSubspaceCount (h : Nat)
  | containingSubspaceCount (h : Nat)
  | contains (subspace : Mat)
  | isContainedIn (subspace : Mat)

/-- This module has only integer and Boolean operations, so it has no materialised value kind. -/
inductive Value
  deriving DecidableEq, Repr

/-- Matrix families have a positive prime. Natural and permutation families do not. -/
def matrixFamily (f : Family) : Bool := 2 ≤ f.p

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let ms := f.members
  match op, f with
  | .gaussianBinomial n q, .range _ _ =>
      if q < 2 then .invalid
      else reduceInt red ms (ms.map fun m => gaussianBinomial q n ((m.headD []).headD 0))
  | .flagCount n q, .words _ _ =>
      if q < 2 then .invalid
      else reduceInt red ms (ms.map fun m => flagCount q n (m.headD []))
  | .containedSubspaceCount h, _ =>
      if !matrixFamily f then .invalid
      else reduceInt red ms (ms.map fun m => containedSubspaceCount f.p m h)
  | .containingSubspaceCount h, _ =>
      if !matrixFamily f then .invalid
      else reduceInt red ms (ms.map fun m => containingSubspaceCount f.p m h)
  | .contains subspace, _ =>
      if !matrixFamily f || !wellShaped ((ms.headD []).headD []).length subspace then .invalid
      else reduceBool red ms (ms.map fun m => contains f.p m subspace)
  | .isContainedIn subspace, _ =>
      if !matrixFamily f || !wellShaped ((ms.headD []).headD []).length subspace then .invalid
      else reduceBool red ms (ms.map fun m => isContainedIn f.p m subspace)
  | _, _ => .invalid

end Lattice_of_subspaces
