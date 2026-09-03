import Mathlib
import Gfq.Reference
import Lk.Contract

/-!
# gfq contract

The executable reference uses polynomial-basis integer codes. This file relates that presentation
to Mathlib's `GaloisField p d`, then states what each matrix operation means there. The statements
with `sorry` are proof obligations, not claims that the C++ backend has been formally verified.
-/

namespace Gfq.Contract

open Matrix Lk

/-- An interpretation of the integer codes in the Mathlib field with the same characteristic and
degree. The explicit modulus chooses the polynomial basis and therefore fixes the byte encoding. -/
structure Representation (K : Field) [Fact K.p.Prime] where
  decode : Nat → GaloisField K.p K.d
  decode_zero : decode 0 = 0
  decode_one : decode 1 = 1
  decode_add : ∀ a b, a < K.q → b < K.q → decode (K.add a b) = decode a + decode b
  decode_mul : ∀ a b, a < K.q → b < K.q → decode (K.mul a b) = decode a * decode b
  injective_codes : Set.InjOn decode (Set.Iio K.q)
  covers : ∀ x, ∃ a < K.q, decode a = x

/-- A valid irreducible presentation gives exactly Mathlib's `GaloisField p d`. -/
theorem representation_exists (K : Field) [Fact K.p.Prime]
    (h : field? K.q K.modulus = some K) : Nonempty (Representation K) := by
  sorry

def toMatrix {K : Field} [Fact K.p.Prime] (rep : Representation K) (rows cols : Nat) (m : Mat) :
    Matrix (Fin rows) (Fin cols) (GaloisField K.p K.d) :=
  fun i j => rep.decode ((m.getD i []).getD j 0)

def toVec {K : Field} [Fact K.p.Prime] (rep : Representation K) (cols : Nat) (v : Vec) :
    Fin cols → GaloisField K.p K.d :=
  fun j => rep.decode (v.getD j 0)

noncomputable def rowSpace {K : Field} [Fact K.p.Prime] (rep : Representation K) (rows cols : Nat) (m : Mat) :
    Submodule (GaloisField K.p K.d) (Fin cols → GaloisField K.p K.d) :=
  Submodule.span (GaloisField K.p K.d) (Set.range (toMatrix rep rows cols m))

def IsRREF {K : Field} [Fact K.p.Prime] (rep : Representation K)
    (rows cols : Nat) (m : Mat) (pivotColumns : List Nat) : Prop :=
  let matrix := toMatrix rep rows cols m
  pivotColumns.Pairwise (· < ·) ∧ (∀ c ∈ pivotColumns, c < cols) ∧
  (∀ i : Fin rows, (i : Nat) < pivotColumns.length ↔ matrix i ≠ 0) ∧
  (∀ i : Fin rows, (hi : (i : Nat) < pivotColumns.length) →
      ∀ j : Fin cols, (j : Nat) < pivotColumns[i] → matrix i j = 0) ∧
  (∀ i : Fin rows, (hi : (i : Nat) < pivotColumns.length) →
      ∀ j : Fin cols, (j : Nat) = pivotColumns[i] → matrix i j = 1) ∧
  (∀ i k : Fin rows, (hi : (i : Nat) < pivotColumns.length) → k ≠ i →
      ∀ j : Fin cols, (j : Nat) = pivotColumns[i] → matrix k j = 0)

variable {K : Field} [Fact K.p.Prime] {rows cols : Nat} {m : Mat}

/-! ## Operations -/

theorem rank_spec (rep : Representation K) (h : Lk.Contract.WellFormed K.q rows cols m) :
    rank K m = (toMatrix rep rows cols m).rank := by
  sorry

theorem rref_spec (rep : Representation K) (h : Lk.Contract.WellFormed K.q rows cols m) :
    Lk.Contract.WellFormed K.q rows cols (rref K m) ∧
    IsRREF rep rows cols (rref K m) (pivots K m) ∧
    rowSpace rep rows cols (rref K m) = rowSpace rep rows cols m := by
  sorry

/-- RREF is the unique byte representation of a row space. -/
theorem rref_canonical (rep : Representation K) {m' : Mat}
    (h : Lk.Contract.WellFormed K.q rows cols m)
    (h' : Lk.Contract.WellFormed K.q rows cols m') :
    rowSpace rep rows cols m = rowSpace rep rows cols m' ↔ rref K m = rref K m' := by
  sorry

theorem nullspace_spec (rep : Representation K) (h : Lk.Contract.WellFormed K.q rows cols m) :
    let vectors := (nullspace K m).map (toVec rep cols)
    LinearIndependent (GaloisField K.p K.d) (fun i : Fin vectors.length => vectors[i]) ∧
    Submodule.span (GaloisField K.p K.d) (Set.range fun i : Fin vectors.length => vectors[i]) =
      LinearMap.ker (toMatrix rep rows cols m).mulVecLin := by
  sorry

theorem inSpan_spec (rep : Representation K) {target : Vec}
    (h : Lk.Contract.WellFormed K.q rows cols m)
    (ht : Lk.Contract.WellFormedVec K.q cols target) :
    inSpan K m target = true ↔ toVec rep cols target ∈ rowSpace rep rows cols m := by
  sorry

theorem solve_spec (rep : Representation K) {rhs : Vec}
    (h : Lk.Contract.WellFormed K.q rows cols m)
    (hrhs : Lk.Contract.WellFormedVec K.q rows rhs) :
    (∀ x, solve K m rhs = some x →
      (toMatrix rep rows cols m).mulVec (toVec rep cols x) = toVec rep rows rhs) ∧
    (solve K m rhs = none → ∀ x : Fin cols → GaloisField K.p K.d,
      (toMatrix rep rows cols m).mulVec x ≠ toVec rep rows rhs) := by
  sorry

theorem inverse_spec (rep : Representation K) {n : Nat}
    (h : Lk.Contract.WellFormed K.q n n m) :
    (∀ output, inverse K m = some output →
      toMatrix rep n n m * toMatrix rep n n output = 1) ∧
    (inverse K m = none → ¬IsUnit (toMatrix rep n n m)) := by
  sorry

end Gfq.Contract
