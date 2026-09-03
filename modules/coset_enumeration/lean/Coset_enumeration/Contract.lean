import Mathlib.GroupTheory.Index
import Mathlib.GroupTheory.PresentedGroup
import Coset_enumeration.Reference

/-!
# coset_enumeration: contract

A completed table certifies the subgroup index and gives the transitive coset action. Failure at
the row bound is one-sided: it is not evidence that the index is infinite.
-/

namespace Coset_enumeration.Contract

open Lk


def decodeLetter (generators letter : Nat) : FreeGroup (Fin generators) :=
  if h : letter / 2 < generators then
    let x := FreeGroup.of (⟨letter / 2, h⟩ : Fin generators)
    if letter % 2 = 0 then x else x⁻¹
  else 1

def decodeWord (generators : Nat) (word : List Nat) : FreeGroup (Fin generators) :=
  word.foldl (fun value letter => value * decodeLetter generators letter) 1

/-- The fixed encoded equations and the varying equation `varying = 1` define the presentation. -/
def relationSet (generators : Nat) (relations : List Vec) (varying : List Nat) :
    Set (FreeGroup (Fin generators)) :=
  {r | ∃ equations, decodeEquations (2 * generators) relations = some equations ∧
    ((∃ equation ∈ equations,
        decodeWord generators equation.left * (decodeWord generators equation.right)⁻¹ = r) ∨
      decodeWord generators varying = r)}

abbrev Presented (generators : Nat) (relations : List Vec) (varying : List Nat) :=
  PresentedGroup (relationSet generators relations varying)

def presentedSubgroup (generators : Nat) (relations : List Vec) (varying : List Nat)
    (subgroup : List (List Nat)) : Subgroup (Presented generators relations varying) :=
  Subgroup.closure {x | ∃ word ∈ subgroup,
    PresentedGroup.mk (relationSet generators relations varying) (decodeWord generators word) = x}

def inversePerm (permutation : Perm) : Perm :=
  (List.range permutation.length).map permutation.idxOf

def actLetter (actions : List Perm) (letter point : Nat) : Nat :=
  let permutation := actions.getD (letter / 2) []
  if letter % 2 = 0 then permutation.getD point 0 else (inversePerm permutation).getD point 0

def actWord (actions : List Perm) (point : Nat) (word : List Nat) : Nat :=
  word.foldl (actLetter actions) point

def IsPermutation (degree : Nat) (permutation : Perm) : Prop :=
  permutation.length = degree ∧ permutation.Perm (List.range degree)

/-- Successful bounded enumeration returns the Mathlib subgroup index. -/
theorem index_spec (generators bound : Nat) (relations : List Vec) (equations : List Equation)
    (subgroup : List (List Nat)) (varying : List Nat) (answer : Enumeration)
    (hdecode : decodeEquations (2 * generators) relations = some equations)
    (h : enumerate generators bound (equationRelators equations ++ [varying]) subgroup = some answer) :
    (presentedSubgroup generators relations varying subgroup).index = answer.degree := by
  sorry

/-- The emitted lists are the generator permutations of the transitive coset action. The
`Equiv` identifies table rows with Mathlib's coset type. The remaining clauses are checks that
do not require constructing the quotient. -/
theorem permutationRepresentation_spec (generators bound : Nat) (relations : List Vec)
    (equations : List Equation) (subgroup : List (List Nat)) (varying : List Nat)
    (answer : Enumeration)
    (hdecode : decodeEquations (2 * generators) relations = some equations)
    (h : enumerate generators bound (equationRelators equations ++ [varying]) subgroup = some answer) :
    Nonempty ((Presented generators relations varying ⧸
      presentedSubgroup generators relations varying subgroup) ≃ Fin answer.degree) ∧
    answer.permutations.length = generators ∧
    (∀ permutation ∈ answer.permutations, IsPermutation answer.degree permutation) ∧
    (∀ equation ∈ equations, ∀ point < answer.degree,
      actWord answer.permutations point equation.left =
        actWord answer.permutations point equation.right) ∧
    (∀ point < answer.degree, actWord answer.permutations point varying = point) ∧
    (∀ word ∈ subgroup, actWord answer.permutations 0 word = 0) ∧
    (∀ point < answer.degree, ∃ word, actWord answer.permutations 0 word = point) := by
  sorry

/-- A positive `is_finite` answer proves finiteness. No converse is claimed for a fixed bound. -/
theorem isFinite_spec (generators bound : Nat) (relations : List Vec) (equations : List Equation)
    (varying : List Nat) (answer : Enumeration)
    (hdecode : decodeEquations (2 * generators) relations = some equations)
    (h : enumerate generators bound (equationRelators equations ++ [varying]) [] = some answer) :
    Finite (Presented generators relations varying) := by
  sorry

end Coset_enumeration.Contract
