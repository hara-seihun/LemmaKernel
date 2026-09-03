import Mathlib
import Young.Reference
import Lk.Contract

/-!
# young: contract

Mathlib has Young diagrams and semistandard tableaux, but it does not yet define standard
Young tableaux, RSK, Specht modules, Kostka numbers, or the Murnaghan-Nakayama rule. This file
states those missing bridges instead of pretending that a native implementation is formally
verified. Every `sorry` is an explicit proof obligation.
-/

namespace Young.Contract

open CategoryTheory Lk YoungDiagram


def HasShape (shape : Vec) (tableau : Mat) : Prop :=
  tableau.length = shape.length ∧
  ∀ i, i < shape.length → (tableau.getD i []).length = shape.headD 0 ∧
    ∀ j, shape.getD i 0 ≤ j → j < shape.headD 0 → (tableau.getD i []).getD j 0 = 0

def IsSemistandard (shape : Vec) (tableau : Mat) : Prop :=
  HasShape shape tableau ∧
  (∀ i j, i < shape.length → j + 1 < shape.getD i 0 →
    (tableau.getD i []).getD j 0 ≤ (tableau.getD i []).getD (j + 1) 0) ∧
  (∀ i j, i + 1 < shape.length → j < shape.getD (i + 1) 0 →
    (tableau.getD i []).getD j 0 < (tableau.getD (i + 1) []).getD j 0)

def HasWeight (shape weight : Vec) (tableau : Mat) : Prop :=
  (∀ x, x < weight.length →
    ((cells shape).filter fun cell => (tableau.getD cell.1 []).getD cell.2 0 = x).length = weight.getD x 0) ∧
  ∀ cell ∈ cells shape, (tableau.getD cell.1 []).getD cell.2 0 < weight.length

def IsStandard (shape : Vec) (tableau : Mat) : Prop :=
  HasShape shape tableau ∧
  (∀ i j, i < shape.length → j + 1 < shape.getD i 0 →
    (tableau.getD i []).getD j 0 < (tableau.getD i []).getD (j + 1) 0) ∧
  (∀ i j, i + 1 < shape.length → j < shape.getD (i + 1) 0 →
    (tableau.getD i []).getD j 0 < (tableau.getD (i + 1) []).getD j 0) ∧
  ((cells shape).map fun cell => (tableau.getD cell.1 []).getD cell.2 0).Perm
    ((List.range (sizeOf shape)).map (· + 1))

def HasWordContent (shape word : Vec) (tableau : Mat) : Prop :=
  ∀ x, ((cells shape).filter fun cell => (tableau.getD cell.1 []).getD cell.2 0 = x).length =
    (word.filter (· = x)).length

/-- A positive weakly decreasing list is the row-length list of a Mathlib Young diagram. -/
theorem shape_realized (shape : Vec) (hpos : positive shape = true) (hdec : weakDecreasing shape = true) :
    ∃ μ : YoungDiagram, μ.rowLens = shape := by
  sorry

/-- The matrix predicate above is exactly Mathlib's `SemistandardYoungTableau` predicate. -/
theorem semistandard_mathlib (shape : Vec) (tableau : Mat)
    (hshape : positive shape = true) (hdec : weakDecreasing shape = true) :
    IsSemistandard shape tableau ↔
      ∃ (μ : YoungDiagram) (hμ : μ.rowLens = shape) (T : SemistandardYoungTableau μ),
        ∀ i j, T i j = (tableau.getD i []).getD j 0 := by
  sorry

/-- `kostkaNumber shape weight` counts semistandard tableaux of the stated shape and weight. -/
theorem kostka_spec (shape weight : Vec)
    (hshape : positive shape = true) (hdec : weakDecreasing shape = true)
    (hweight : validComposition weight (sizeOf shape) = true) :
    kostkaNumber shape weight = Nat.card {tableau : Mat //
      IsSemistandard shape tableau ∧ HasWeight shape weight tableau} := by
  sorry

/-- The hook-length result is the number of standard tableaux of the shape. -/
theorem hook_length_spec (shape : Vec)
    (hshape : positive shape = true) (hdec : weakDecreasing shape = true) :
    hookLengthCount shape = Nat.card {tableau : Mat // IsStandard shape tableau} := by
  sorry

/-- The runtime family contains each standard tableau once, in its declared removable-corner
order. -/
theorem standard_tableaux_family_spec (shape : Vec)
    (hshape : positive shape = true) (hdec : weakDecreasing shape = true) :
    let members := (Family.standardTableaux shape).members
    members.Nodup ∧ ∀ tableau, tableau ∈ members ↔ IsStandard shape tableau := by
  sorry

/-- RSK returns same-shape semistandard and standard tableaux, and insertion preserves the
multiset of letters. Its recursive row insertion in `Reference.lean` fixes the correspondence,
including repeated letters. -/
theorem rsk_spec (word shape : Vec) (insertion recording : Mat)
    (h : rskWord word = (shape, insertion, recording)) :
    validPartition (shape.filter (· > 0)) word.length = true ∧
    IsSemistandard (shape.filter (· > 0)) insertion ∧
    IsStandard (shape.filter (· > 0)) recording ∧
    HasWordContent (shape.filter (· > 0)) word insertion := by
  sorry

/-- Mathlib does not yet provide Specht modules. This declaration names the characteristic-zero
irreducible representation indexed by `shape`; constructing it is one of this contract's open
obligations. -/
noncomputable def spechtRepresentation (n : Nat) (shape : Vec) :
    FDRep ℂ (Equiv.Perm (Fin n)) := by
  sorry

theorem specht_simple (n : Nat) (shape : Vec) (hshape : validPartition shape n = true) :
    Simple (spechtRepresentation n shape) := by
  sorry

/-- Murnaghan-Nakayama gives the character of the Specht representation on the conjugacy class
whose Mathlib partition is `cycleType`. -/
theorem murnaghan_nakayama_spec (n : Nat) (shape cycleType : Vec)
    (hshape : validPartition shape n = true) (hcycle : validPartition cycleType n = true)
    (σ : Equiv.Perm (Fin n)) (htype : σ.partition.parts = (cycleType : Multiset Nat)) :
    (spechtRepresentation n shape).character σ = (murnaghanNakayama shape cycleType : ℂ) := by
  sorry

end Young.Contract
