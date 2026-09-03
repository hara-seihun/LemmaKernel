import Mathlib
import Quantum_codes.Reference
import Lk.Contract

/-!
# quantum_codes: contract

The executable reference uses lists. This file states its meaning in Mathlib's finite-dimensional
vector spaces over `ZMod 2`. A row `(x|z)` is read as a function on `Fin n ⊕ Fin n`; the two
summands are the X and Z coordinates. Under the additive `GF(4)` labelling `(x,z) ↦ x + omega*z`,
the form below is the trace-Hermitian form and `symplecticWeightM` is GF(4) Hamming weight.

The statements are obligations rather than proofs of the native backend. Tests compare every
backend with `Reference.lean`; the `sorry`s here record the bridge from that oracle to Mathlib.
-/

namespace Quantum_codes.Contract

open Lk Lk.Contract
open scoped BigOperators

abbrev SymplecticSpace (n : Nat) := (Fin n ⊕ Fin n) → ZMod 2

def toSymplecticVec (n : Nat) (v : Vec) : SymplecticSpace n
  | Sum.inl i => (v.getD i 0 : ZMod 2)
  | Sum.inr i => (v.getD (n + i) 0 : ZMod 2)

/-- The standard nondegenerate alternating form on `(F_2^n | F_2^n)`. -/
def form {n : Nat} (u v : SymplecticSpace n) : ZMod 2 :=
  ∑ i : Fin n, (u (Sum.inl i) * v (Sum.inr i) + u (Sum.inr i) * v (Sum.inl i))

def codeSpace (n rows : Nat) (m : Mat) : Submodule (ZMod 2) (SymplecticSpace n) :=
  Submodule.span (ZMod 2) (Set.range fun i : Fin rows => toSymplecticVec n (m.getD i []))

def IsLogical {n : Nat} (C : Submodule (ZMod 2) (SymplecticSpace n)) (v : SymplecticSpace n) : Prop :=
  (∀ c ∈ C, form v c = 0) ∧ v ∉ C

def symplecticWeightM {n : Nat} (v : SymplecticSpace n) : Nat :=
  (Finset.univ.filter fun i : Fin n => v (Sum.inl i) ≠ 0 ∨ v (Sum.inr i) ≠ 0).card

def XOnly {n : Nat} (v : SymplecticSpace n) : Prop := ∀ i, v (Sum.inr i) = 0

def ZOnly {n : Nat} (v : SymplecticSpace n) : Prop := ∀ i, v (Sum.inl i) = 0

/-- A CSS space is the sum of its X-only and Z-only subspaces. Their intersection is zero, so the
sum is automatically direct. -/
def IsCSS {n : Nat} (C : Submodule (ZMod 2) (SymplecticSpace n)) : Prop :=
  ∀ v ∈ C, ∃ x ∈ C, ∃ z ∈ C, XOnly x ∧ ZOnly z ∧ v = x + z

variable {n rows : Nat} {m : Mat}

/-- The row generators pair to zero exactly when their whole span is symplectically
self-orthogonal. -/
theorem isSelfOrthogonal_spec (h : WellFormed 2 rows (2 * n) m) :
    isSelfOrthogonal m = true ↔
      ∀ u ∈ codeSpace n rows m, ∀ v ∈ codeSpace n rows m, form u v = 0 := by
  sorry

/-- `distance` is the minimum symplectic weight of a logical Pauli. When no logical Pauli exists,
which includes a Lagrangian `[[n,0]]` stabiliser, its canonical sentinel is `n+1`. -/
theorem distance_spec (h : WellFormed 2 rows (2 * n) m) :
    let C := codeSpace n rows m
    let d := distance m
    ((¬∃ v, IsLogical C v) → d = n + 1) ∧
    ((∃ v, IsLogical C v) →
      (∃ v, IsLogical C v ∧ symplecticWeightM v = d) ∧
      ∀ v, IsLogical C v → d ≤ symplecticWeightM v) := by
  sorry

/-- The rank criterion evaluated by `isCss` is equivalent to generation by separate X and Z
checks and does not depend on the displayed generator rows. -/
theorem isCss_spec (h : WellFormed 2 rows (2 * n) m) :
    isCss m = true ↔ IsCSS (codeSpace n rows m) := by
  sorry

end Quantum_codes.Contract
