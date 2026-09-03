import Mathlib
import Boolean_functions.Reference
import Lk.Contract

/-!
# Boolean functions: contract

Truth-table column `x` is the binary encoding of an element of `F_2^n`; variable `x_i` is its
`i`-th low bit. The reference definitions below are related to finite sums, Hamming distance,
algebraic normal form, and differential uniformity. The proofs remain explicit obligations.
-/

namespace Boolean_functions.Contract

open scoped BigOperators
open Lk

/-- The Boolean function represented by a truth-table row. -/
def truth (n : Nat) (row : Vec) (x : Fin (2 ^ n)) : Bool := row.getD x.val 0 = 1

/-- Evaluation of the linear form indexed by `u`, using the numeric binary bases on both sides. -/
def linearValue (n : Nat) (u x : Fin (2 ^ n)) : Bool := bitDotParity 64 u.val x.val = 1

def affineValue (n : Nat) (u : Fin (2 ^ n)) (c : Bool) (x : Fin (2 ^ n)) : Bool :=
  xor (linearValue n u x) c

/-- Hamming distance between a table and one affine Boolean function. -/
def affineDistance (n : Nat) (row : Vec) (u : Fin (2 ^ n)) (c : Bool) : Nat :=
  (Finset.univ.filter fun x => truth n row x ≠ affineValue n u c x).card

/-- The character sum defining the Walsh coefficient at `u`. -/
def walshMath (n : Nat) (row : Vec) (u : Fin (2 ^ n)) : Int :=
  ∑ x : Fin (2 ^ n), if truth n row x = linearValue n u x then 1 else -1

/-- The reference nonlinearity is exactly the minimum distance to the first-order Reed-Muller
code. -/
theorem nonlinearity_spec (n : Nat) (row : Vec) (hlen : row.length = 2 ^ n) :
    (∀ u c, nonlinearity row ≤ affineDistance n row u c) ∧
    ∃ u c, nonlinearity row = affineDistance n row u c := by
  sorry

/-- Walsh coefficients use increasing numeric frequency order. -/
theorem walsh_spec (n : Nat) (row : Vec) (hlen : row.length = 2 ^ n) :
    ∀ u : Fin (2 ^ n), (walsh row).getD u.val 0 = walshMath n row u := by
  sorry

/-- Bent means flat Walsh magnitude. The parity condition makes the square root an integer. -/
theorem bent_spec (n : Nat) (row : Vec) (hlen : row.length = 2 ^ n) :
    isBent row = true ↔ n % 2 = 0 ∧ ∀ u : Fin (2 ^ n), (walshMath n row u).natAbs = 2 ^ (n / 2) := by
  sorry

/-- The ANF coefficient of the square-free monomial indexed by `mask`, as a sum in `F_2`. -/
def anfCoeffMath (n : Nat) (row : Vec) (mask : Fin (2 ^ n)) : ZMod 2 :=
  ∑ x : Fin (2 ^ n), if isSubmask 64 x.val mask.val then row.getD x.val 0 else 0

/-- `algebraicDegree` is the largest degree of a nonzero coordinate ANF coefficient. The all-zero
vectorial function uses degree zero. -/
theorem algebraicDegree_spec (n outputs : Nat) (table : Mat)
    (hrows : table.length = outputs) (hcols : ∀ row ∈ table, row.length = 2 ^ n) :
    let d := algebraicDegree table
    (∀ row ∈ table, ∀ mask : Fin (2 ^ n), anfCoeffMath n row mask ≠ 0 → popcount 64 mask.val ≤ d) ∧
    (d = 0 ∨ ∃ row ∈ table, ∃ mask : Fin (2 ^ n), anfCoeffMath n row mask ≠ 0 ∧ popcount 64 mask.val = d) := by
  sorry

/-- The value of a vectorial truth table in `F_2^outputs`. -/
def vectorValue (outputs : Nat) (table : Mat) (x : Nat) : Fin outputs → ZMod 2 :=
  fun i => (table.getD i.val []).getD x 0

/-- APN means every nonzero derivative takes each value at most twice. -/
theorem apn_spec (n outputs : Nat) (table : Mat)
    (hrows : table.length = outputs) (hcols : ∀ row ∈ table, row.length = 2 ^ n) :
    isAPN table = true ↔
      ∀ a : Fin (2 ^ n), a ≠ 0 → ∀ b : Fin outputs → ZMod 2,
        (Finset.univ.filter fun x : Fin (2 ^ n) =>
          vectorValue outputs table x.val + vectorValue outputs table (xorNat 64 x.val a.val) = b).card ≤ 2 := by
  sorry

/-- The numeric input index as its low-bit-first vector over `F_2`. -/
def binaryVector (n : Nat) (x : Fin (2 ^ n)) : Fin n → ZMod 2 :=
  fun i => x.val / 2 ^ i.val

/-- The numeric index represented by a low-bit-first vector. -/
noncomputable def binaryIndex (n : Nat) (x : Fin n → ZMod 2) : Fin (2 ^ n) :=
  ⟨(∑ i, (x i).val * 2 ^ i.val) % 2 ^ n, Nat.mod_lt _ (by positivity)⟩

/-- Domain-affine equivalence: the output is unchanged and the input is precomposed with one
invertible affine map `x ↦ A x + b`. -/
def DomainAffineEquivalent (n : Nat) (row candidate : Vec) : Prop :=
  ∃ A : (Fin n → ZMod 2) ≃ₗ[ZMod 2] (Fin n → ZMod 2),
    ∃ b : Fin n → ZMod 2, ∀ x : Fin (2 ^ n),
      truth n candidate x = truth n row (binaryIndex n (A (binaryVector n x) + b))

/-- `affineClass` is in the input-affine orbit and is lexicographically no larger than every
other truth table in that orbit. -/
theorem affineClass_spec (n : Nat) (row : Vec) (hlen : row.length = 2 ^ n)
    (hbinary : ∀ x ∈ row, x < 2) :
    DomainAffineEquivalent n row (affineClass row) ∧
      ∀ candidate, candidate.length = 2 ^ n → DomainAffineEquivalent n row candidate →
        lexLe (affineClass row) candidate = true := by
  sorry

end Boolean_functions.Contract
