import Mathlib
import Circuit_fires.Reference
import Lk.Contract

/-!
# circuit_fires: contract

The definitions below state a concise circuit and its finite-difference fire directly over
`ZMod p`. The transport from row matrices and little-endian point indices to these functions is
left as an explicit proof obligation.
-/

namespace Circuit_fires.Contract

open scoped BigOperators

abbrev F (p : Nat) := ZMod p


def fibreDim (configuration : Lk.Mat) (g : Nat) : Nat :=
  (configuration.headD []).length - g


def z (p : Nat) (configuration : Lk.Mat) (g : Nat)
    (i : Fin configuration.length) (coordinate : Fin g) : F p :=
  (configuration.getD i.val []).getD coordinate.val 0


def u (p : Nat) (configuration : Lk.Mat) (g : Nat)
    (i : Fin configuration.length) (coordinate : Fin (fibreDim configuration g)) : F p :=
  (configuration.getD i.val []).getD (g + coordinate.val) 0


def tensorRelation (p : Nat) (configuration : Lk.Mat) (g : Nat)
    (coefficients : Fin configuration.length → F p) : Prop :=
  ∀ a : Fin (fibreDim configuration g), ∀ b : Fin g,
    ∑ i, coefficients i * u p configuration g i a * z p configuration g i b = 0


def spansBase (p : Nat) (configuration : Lk.Mat) (g : Nat) : Prop :=
  ∀ target : Fin g → F p, ∃ coefficients : Fin configuration.length → F p,
    ∀ coordinate, target coordinate = ∑ i, coefficients i * z p configuration g i coordinate


def spansFibreDual (p : Nat) (configuration : Lk.Mat) (g : Nat) : Prop :=
  ∀ target : Fin (fibreDim configuration g) → F p,
    ∃ coefficients : Fin configuration.length → F p,
      ∀ coordinate, target coordinate = ∑ i, coefficients i * u p configuration g i coordinate


def MathCircuit (p : Nat) (configuration : Lk.Mat) (g : Nat) : Prop :=
  0 < configuration.length ∧ 0 < g ∧ g < (configuration.headD []).length ∧
  (∀ i, z p configuration g i ≠ 0) ∧
  (∀ i, u p configuration g i ≠ 0) ∧
  (∀ i j, i ≠ j → ¬ ∃ scalar : F p,
    ∀ coordinate, z p configuration g i coordinate = scalar * z p configuration g j coordinate) ∧
  spansBase p configuration g ∧ spansFibreDual p configuration g ∧
  tensorRelation p configuration g 1 ∧
  (∀ coefficients, tensorRelation p configuration g coefficients →
    ∃ scalar : F p, coefficients = fun _ => scalar)


def MathFire (p : Nat) (configuration : Lk.Mat) (g : Nat) : Prop :=
  MathCircuit p configuration g ∧
  ∃ potential : (Fin g → F p) → (Fin (fibreDim configuration g) → F p),
    ∃ defects : Fin configuration.length → F p,
      potential 0 = 0 ∧
      (∀ i x,
        ∑ a, u p configuration g i a *
          (potential (x + z p configuration g i) a - potential x a) = defects i) ∧
      ∑ i, defects i ≠ 0


def pointIndex {p g : Nat} (x : Fin g → F p) : Nat :=
  ∑ coordinate, (x coordinate).val * p ^ coordinate.val


def listedPotential (p g h : Nat) (potential : Lk.Mat)
    (x : Fin g → F p) (coordinate : Fin h) : F p :=
  (potential.getD (pointIndex x) []).getD coordinate.val 0


def listedDefect (p : Nat) (defects : Lk.Vec)
    (i : Fin defects.length) : F p := defects.getD i.val 0


def MathVerifies (p : Nat) (configuration : Lk.Mat) (g : Nat)
    (defects : Lk.Vec) (potential : Lk.Mat) : Prop :=
  let h := fibreDim configuration g
  defects.length = configuration.length ∧ potential.length = p ^ g ∧
  (∀ row ∈ potential, row.length = h) ∧ MathCircuit p configuration g ∧
  (∀ coordinate, listedPotential p g h potential 0 coordinate = 0) ∧
  (∀ i : Fin configuration.length, ∀ x : Fin g → F p,
    ∑ a, u p configuration g i a *
      (listedPotential p g h potential (x + z p configuration g i) a -
       listedPotential p g h potential x a) =
      (defects.getD i.val 0 : F p)) ∧
  (∑ i : Fin configuration.length, (defects.getD i.val 0 : F p)) ≠ 0


/-- The executable circuit predicate is exactly the concise minimal tensor relation above. -/
theorem isCircuit_spec (p : Nat) (hp : p.Prime) (configuration : Lk.Mat) (g : Nat) :
    Circuit_fires.isCircuit p configuration g = true ↔ MathCircuit p configuration g := by
  sorry

/-- The rank comparison in the reference decides existence of a nonzero total defect. -/
theorem isFire_spec (p : Nat) (hp : p.Prime) (configuration : Lk.Mat) (g : Nat) :
    Circuit_fires.isFire p configuration g = true ↔ MathFire p configuration g := by
  sorry

def solutionPotential (p : Nat) (configuration : Lk.Mat) (g : Nat)
    (solution : Lk.Vec) : Lk.Mat :=
  let h := fibreDim configuration g
  (List.range (p ^ g)).map fun point =>
    (List.range h).map fun coordinate => solution.getD (point * h + coordinate) 0


def solutionDefects (p : Nat) (configuration : Lk.Mat) (g : Nat)
    (solution : Lk.Vec) : Lk.Vec :=
  solution.drop (p ^ g * fibreDim configuration g)


/-- A returned construction is a listed witness whose total defect has been normalized to one. -/
theorem fireSolution_sound (p : Nat) (hp : p.Prime) (configuration : Lk.Mat)
    (g : Nat) (solution : Lk.Vec)
    (found : Circuit_fires.fireSolution p configuration g = some solution) :
    MathVerifies p configuration g (solutionDefects p configuration g solution)
      (solutionPotential p configuration g solution) ∧
    (solutionDefects p configuration g solution).foldl
      (fun total value => total + (value : F p)) 0 = 1 := by
  sorry

/-- Explicit construction succeeds exactly when a fire exists within its declared finite domain. -/
theorem fireSolution_some_iff (p : Nat) (hp : p.Prime) (configuration : Lk.Mat) (g : Nat) :
    (∃ solution, Circuit_fires.fireSolution p configuration g = some solution) ↔
      MathFire p configuration g := by
  sorry

/-- The listed potential uses row `sum_j x_j p^j`, matching the interchange contract. -/
theorem verifiesPotential_spec (p : Nat) (hp : p.Prime) (configuration : Lk.Mat)
    (g : Nat) (defects : Lk.Vec) (potential : Lk.Mat) :
    Circuit_fires.verifiesPotential p configuration g defects potential = true ↔
      MathVerifies p configuration g defects potential := by
  sorry

end Circuit_fires.Contract
