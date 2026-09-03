import Mathlib
import Hadamard.Reference
import Lk.Contract

/-!
# hadamard: contract

The runtime stores a sign matrix as an F_2 matrix. `signMatrix` gives its mathematical value over
the integers, with bit zero equal to +1 and bit one equal to -1. A conference input uses the same
encoding off the diagonal and requires zero bits on the diagonal; `conferenceMatrix` replaces
those diagonal +1 entries by integer zeros.
-/

namespace Hadamard.Contract

open Matrix Lk Lk.Contract


def sign (b : Nat) : Int := if b % 2 = 0 then 1 else -1

def signMatrix (rows cols : Nat) (m : Mat) : Matrix (Fin rows) (Fin cols) Int :=
  fun i j => sign ((m.getD i []).getD j 0)

def conferenceMatrix (n : Nat) (m : Mat) : Matrix (Fin n) (Fin n) Int :=
  fun i j => if i = j then 0 else sign ((m.getD i []).getD j 0)

def oneVec (n : Nat) : Fin n → Int := fun _ => 1

def IsSign (x : Int) : Prop := x = 1 ∨ x = -1

/-- Standard signed row and column equivalence. The permutations may reorder rows and columns,
and `rs` and `cs` independently negate them. -/
def SignEquivalent {rows cols : Nat}
    (a b : Matrix (Fin rows) (Fin cols) Int) : Prop :=
  ∃ pr : Equiv.Perm (Fin rows), ∃ pc : Equiv.Perm (Fin cols),
  ∃ rs : Fin rows → Int, ∃ cs : Fin cols → Int,
    (∀ i, IsSign (rs i)) ∧ (∀ j, IsSign (cs j)) ∧
    ∀ i j, b i j = rs i * a (pr i) (pc j) * cs j

variable {n rows cols : Nat} {m : Mat}

theorem isHadamard_spec (h : WellFormed 2 n n m) :
    isHadamard m = true ↔
      signMatrix n n m * (signMatrix n n m).transpose =
        (n : Int) • (1 : Matrix (Fin n) (Fin n) Int) := by
  sorry

theorem isSkew_spec (h : WellFormed 2 n n m) :
    isSkew m = true ↔
      signMatrix n n m + (signMatrix n n m).transpose =
        (2 : Int) • (1 : Matrix (Fin n) (Fin n) Int) := by
  sorry

theorem isRegular_spec (h : WellFormed 2 n n m) :
    isRegular m = true ↔
      ∃ r : Int,
        (signMatrix n n m).mulVec (oneVec n) = r • oneVec n ∧
        (signMatrix n n m).transpose.mulVec (oneVec n) = r • oneVec n := by
  sorry

theorem isConference_spec (h : WellFormed 2 n n m) :
    isConference m = true ↔
      2 ≤ n ∧ (∀ i : Fin n, (m.getD i []).getD i 0 = 0) ∧
      conferenceMatrix n m * (conferenceMatrix n m).transpose =
        ((n - 1 : Nat) : Int) • (1 : Matrix (Fin n) (Fin n) Int) := by
  sorry

/-- The chosen form remains in the signed row and column equivalence class and is no larger than
any other well-formed binary encoding in that class. -/
theorem canonicalForm_spec (h : WellFormed 2 rows cols m) :
    WellFormed 2 rows cols (canonicalForm m) ∧
    SignEquivalent (signMatrix rows cols m) (signMatrix rows cols (canonicalForm m)) ∧
    ∀ other, WellFormed 2 rows cols other →
      SignEquivalent (signMatrix rows cols m) (signMatrix rows cols other) →
      matrixLexLe (canonicalForm m) other = true := by
  sorry

end Hadamard.Contract
