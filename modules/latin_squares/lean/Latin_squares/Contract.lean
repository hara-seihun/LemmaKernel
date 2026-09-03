import Mathlib
import Latin_squares.Reference
import Lk.Contract

/-!
# Latin squares: contract

The predicates below use Mathlib's finite types, permutations, injectivity, and finite cardinality.
They state what callers may assume about the executable reference. Proofs are left as explicit
obligations.
-/

namespace Latin_squares.Contract

open Lk Latin_squares


def WellFormedSquare (n : Nat) (m : Mat) : Prop :=
  n > 0 ∧ m.length = n ∧ ∀ row ∈ m, row.length = n ∧ ∀ x ∈ row, x < n

def entry (m : Mat) (i j : Nat) : Nat := (m.getD i []).getD j 0

def IsLatinSquare (n : Nat) (m : Mat) : Prop :=
  WellFormedSquare n m ∧
  (∀ i : Fin n, Function.Injective fun j : Fin n => entry m i j) ∧
  (∀ j : Fin n, Function.Injective fun i : Fin n => entry m i j)

def AreOrthogonal (n : Nat) (a b : Mat) : Prop :=
  Function.Injective fun cell : Fin n × Fin n =>
    (entry a cell.1 cell.2, entry b cell.1 cell.2)

def IsTransversal (n : Nat) (m : Mat) (columns : Equiv.Perm (Fin n)) : Prop :=
  Function.Injective fun i : Fin n => entry m i (columns i)

def HasGroupLaw (n : Nat) (m : Mat) : Prop :=
  (∃ e < n, ∀ x < n, entry m e x = x ∧ entry m x e = x) ∧
  ∀ x < n, ∀ y < n, ∀ z < n,
    entry m (entry m x y) z = entry m x (entry m y z)

def Isotopic (n : Nat) (a b : Mat) : Prop :=
  ∃ rows columns symbols : Equiv.Perm (Fin n),
    ∀ i j : Fin n, ∃ h : entry a (rows i) (columns j) < n,
      entry b i j = symbols ⟨entry a (rows i) (columns j), h⟩

theorem latinSquares_members_spec (n : Nat) (hn : n > 0) (m : Mat) :
    m ∈ latinSquaresMembers n ↔ IsLatinSquare n m := by
  sorry

theorem latinSquares_members_nodup (n : Nat) : (latinSquaresMembers n).Nodup := by
  sorry

theorem isLatin_spec (m : Mat) :
    isLatin m = true ↔ IsLatinSquare m.length m := by
  sorry

theorem hasOrthogonalMate_spec (m : Mat) (h : WellFormedSquare m.length m) :
    hasOrthogonalMate m = true ↔
      IsLatinSquare m.length m ∧
      ∃ mate, IsLatinSquare m.length mate ∧ AreOrthogonal m.length m mate := by
  sorry

theorem transversalCount_spec (m : Mat) (h : WellFormedSquare m.length m) :
    transversalCount m = Nat.card {columns : Equiv.Perm (Fin m.length) //
      IsTransversal m.length m columns} := by
  sorry

theorem isGroupTable_spec (m : Mat) (h : WellFormedSquare m.length m) :
    isGroupTable m = true ↔ IsLatinSquare m.length m ∧ HasGroupLaw m.length m := by
  sorry

theorem isotopyCanonicalForm_spec (m : Mat) (h : IsLatinSquare m.length m) :
    let canonical := isotopyCanonicalForm m
    IsLatinSquare m.length canonical ∧ Isotopic m.length m canonical ∧
      ∀ other, WellFormedSquare m.length other → Isotopic m.length m other →
        lexLe canonical.flatten other.flatten = true := by
  sorry

end Latin_squares.Contract
