import Mathlib
import Characters.Reference

/-!
# characters: contract

The executable reference represents a cyclotomic integer by the eigenvalue exponents of a group
element. If a cell contains `[a_1, ..., a_d]`, its value is
`zeta_e^a_1 + ... + zeta_e^a_d`, where `e` is the table conductor. This representation is exact
and canonical because the multiset is the spectrum of the representing matrix, rather than a
non-unique polynomial in a root of unity.

This first backend covers abelian groups. Mathlib proves that an irreducible complex
representation of a finite abelian group has degree one in
`Representation.IsIrreducible.finrank_eq_one_of_isMulCommutative`. The rows below enumerate all
homomorphisms into the roots of unity, hence all of those irreducible characters. The row and
Frobenius-Schur statements are left as explicit obligations where they connect the list model to
Mathlib's complex numbers.
-/

namespace Characters.Contract

open Lk Characters

noncomputable def root (conductor exponent : Nat) : ℂ :=
  Complex.exp ((((2 * Real.pi * exponent / conductor : ℝ) : ℂ)) * Complex.I)

noncomputable def decodedRow (conductor : Nat) (row : List Nat) : List ℂ :=
  row.map (root conductor)

noncomputable def rowInner (conductor : Nat) (left right : List Nat) : ℂ :=
  (List.zipWith (fun a b => root conductor a * starRingEnd ℂ (root conductor b)) left right).sum

/-- The output dimensions agree with the order of the group. Abelian conjugacy classes are
singletons, in the lexicographic order inherited from `group_elements`. -/
theorem table_shape (gens : List Perm) (data : Data) (h : characterData gens = some data) :
    let table := data.table
    table.1.1 = data.elements.length ∧
    table.2.1.1 = List.range data.elements.length ∧
    table.2.1.2 = List.replicate data.elements.length 1 ∧
    table.2.2.1 = List.replicate data.elements.length 1 := by
  sorry

/-- Every encoded row is a one-dimensional character: exponents add modulo the conductor when
group elements multiply. Applying `root` turns this into multiplication in `ℂ`. -/
theorem row_multiplicative (gens : List Perm) (data : Data) (h : characterData gens = some data)
    (row : List Nat) (hrow : row ∈ data.homs) (i j : Nat) :
    root data.conductor (row.getD (multiplicationIndex data.elements i j) 0) =
      root data.conductor (row.getD i 0) * root data.conductor (row.getD j 0) := by
  sorry

/-- Distinct rows are orthogonal and each row has squared norm `|G|`. This is the list form of
`Representation.char_orthonormal`, before division by the group order. -/
theorem row_orthogonality (gens : List Perm) (data : Data) (h : characterData gens = some data)
    (i j : Nat) (hi : i < data.homs.length) (hj : j < data.homs.length) :
    rowInner data.conductor (data.homs.getD i []) (data.homs.getD j []) =
      if i = j then data.elements.length else 0 := by
  sorry

/-- The reference emits exactly `|G|` distinct rows. Together with degree one and orthogonality,
this is completeness for a finite abelian group. -/
theorem rows_complete (gens : List Perm) (data : Data) (h : characterData gens = some data) :
    data.homs.Nodup ∧ data.homs.length = data.elements.length := by
  sorry

/-- Mathlib's general representation-theoretic reason every irreducible in this module's domain
has degree one. -/
theorem mathlib_irreducible_degree_one {G V : Type*} [Group G] [IsMulCommutative G]
    [AddCommGroup V] [Module ℂ V] [FiniteDimensional ℂ V]
    (rho : Representation ℂ G V) [Representation.IsIrreducible rho] :
    Module.finrank ℂ V = 1 := by
  exact Representation.IsIrreducible.finrank_eq_one_of_isMulCommutative rho

noncomputable def frobeniusSchurAverage (data : Data) (row : List Nat) : ℂ :=
  ((List.range data.elements.length).map fun i =>
    root data.conductor (row.getD (multiplicationIndex data.elements i i) 0)).sum /
      data.elements.length

/-- `frobenius_schur` is `|G|^-1 * sum_g chi(g^2)`. For a linear character it is one exactly
when the character is real, and zero otherwise. -/
theorem frobenius_schur_spec (gens : List Perm) (data : Data) (h : characterData gens = some data)
    (i : Nat) (hi : i < data.homs.length) :
    frobeniusSchurAverage data (data.homs.getD i []) =
      ((data.indicators.getD i 0 : Int) : ℂ) := by
  sorry

/-- `sameRestriction` compares cyclotomic values after embedding both root systems into their
least common conductor. -/
theorem sameRestriction_spec (ambient subgroup : Data) (ambientRow subgroupRow : List Nat) :
    sameRestriction ambient subgroup ambientRow subgroupRow = true ↔
      ∀ i < subgroup.elements.length,
        root (lcm ambient.conductor subgroup.conductor)
            (ambientRow.getD (ambient.elements.idxOf (subgroup.elements.getD i [])) 0 *
              (lcm ambient.conductor subgroup.conductor / ambient.conductor)) =
          root (lcm ambient.conductor subgroup.conductor)
            (subgroupRow.getD i 0 * (lcm ambient.conductor subgroup.conductor / subgroup.conductor)) := by
  sorry

/-- Restriction multiplicities are inner products with the subgroup's irreducible rows. Since all
representations in this module's domain are linear, exactly one row has multiplicity one. -/
theorem restriction_spec (ambient subgroup : Data) (character i : Nat)
    (hsub : isSubgroup ambient subgroup = true) (hc : character < ambient.homs.length)
    (hi : i < subgroup.homs.length) :
    (restrictionMultiplicities ambient subgroup character).getD i 0 =
      if sameRestriction ambient subgroup (ambient.homs.getD character [])
          (subgroup.homs.getD i []) then 1 else 0 := by
  sorry

/-- Frobenius reciprocity: an ambient linear character occurs in the induction of a subgroup
character exactly when its restriction is that character. -/
theorem induction_spec (ambient subgroup : Data) (character i : Nat)
    (hsub : isSubgroup ambient subgroup = true) (hc : character < subgroup.homs.length)
    (hi : i < ambient.homs.length) :
    (inductionMultiplicities ambient subgroup character).getD i 0 =
      if sameRestriction ambient subgroup (ambient.homs.getD i [])
          (subgroup.homs.getD character []) then 1 else 0 := by
  sorry

end Characters.Contract
