import Mathlib
import Cayley.Reference
import Lk.Contract

/-!
# cayley: contract

The executable reference uses lists of permutation images. This file states its graph operations
against Mathlib's `SimpleGraph`. The representation hypothesis says that the indexed permutation
closure and the caller's finite group have the same multiplication table. Proofs are left as
explicit obligations where they require transporting the list algorithms through that encoding.
-/

namespace Cayley.Contract

open Lk Cayley

variable {G : Type*} [_root_.Group G] [Fintype G]

/-- The undirected Cayley graph. `fromRel` removes loops and symmetrises the relation, so this is
total even before the connection set hypotheses are supplied. -/
def cayleyGraph (S : Set G) : SimpleGraph G :=
  SimpleGraph.fromRel fun x y => x⁻¹ * y ∈ S

def IsConnectionSet (S : Set G) : Prop :=
  (1 : G) ∉ S ∧ ∀ x, x ∈ S → x⁻¹ ∈ S

/-- The CI property for one connection set, with no connectedness assumption. -/
def IsCISet (S : Set G) : Prop :=
  IsConnectionSet S ∧ ∀ T : Set G, IsConnectionSet T →
    Nonempty (cayleyGraph S ≃g cayleyGraph T) → ∃ φ : MulAut G, φ '' S = T

/-- Interpret selected indices through an equivalence from the reference closure to `G`. -/
def decodedSet {es : List Perm} (decode : Fin es.length ≃ G) (s : List Nat) : Set G :=
  {x | ∃ i : Fin es.length, (i : Nat) ∈ s ∧ decode i = x}

/-- The indexed multiplication computed by the reference is the multiplication of `G`. -/
def Represents (es : List Perm) (decode : Fin es.length ≃ G) : Prop :=
  ∀ a b : Fin es.length,
    ((decode.symm (decode a * decode b) : Fin es.length) : Nat) = mulIndex es a b

variable {es : List Perm} (decode : Fin es.length ≃ G) (hmodel : Represents es decode)


theorem connected_spec (raw : Mat) (hraw : validMember es raw = true) :
    connected es (memberIndices es raw) = true ↔
      (cayleyGraph (decodedSet decode (memberIndices es raw))).Connected := by
  sorry

theorem regular_spec (raw : Mat) (degree : Nat) (hraw : validMember es raw = true) :
    isRegularOfDegree es (memberIndices es raw) degree = true ↔
      ∀ v, ((cayleyGraph (decodedSet decode (memberIndices es raw))).neighborSet v).ncard = degree := by
  sorry

theorem girth_spec (raw : Mat) (hraw : validMember es raw = true) :
    girth es (memberIndices es raw) =
      (cayleyGraph (decodedSet decode (memberIndices es raw))).girth := by
  sorry

theorem diameter_spec (raw : Mat) (hraw : validMember es raw = true) :
    diameter es (memberIndices es raw) =
      (cayleyGraph (decodedSet decode (memberIndices es raw))).diam := by
  sorry

theorem autOrder_spec (raw : Mat) (hraw : validMember es raw = true) :
    autOrder es (memberIndices es raw) =
      Nat.card ((cayleyGraph (decodedSet decode (memberIndices es raw))) ≃g
        (cayleyGraph (decodedSet decode (memberIndices es raw)))) := by
  sorry

theorem isCISet_spec (raw : Mat) (hraw : validMember es raw = true) :
    isCISet es (memberIndices es raw) = true ↔
      IsCISet (decodedSet decode (memberIndices es raw)) := by
  sorry

end Cayley.Contract
