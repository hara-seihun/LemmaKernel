import Mathlib
import Cayley_iso.Reference
import Automorphisms.Contract
import Cayley.Contract
import Lk.Contract

/-!
# cayley_iso: contract

The executable reference works with Cayley-table labels. This file states the two class counts and
the fixed-size CI predicate for a Mathlib finite group. The transport from labels to group elements
is left as an explicit proof obligation.
-/

namespace Cayley_iso.Contract

variable {G : Type*} [Group G] [Fintype G] [DecidableEq G]

/-- An inverse-closed identity-free connection set of cardinality `k`. -/
def Connection (G : Type*) [Group G] [Fintype G] [DecidableEq G] (k : Nat) :=
  {S : Finset G // (1 : G) ∉ S ∧ S.card = k ∧ ∀ x ∈ S, x⁻¹ ∈ S}

/-- The undirected Cayley graph belonging to a fixed-size connection set. -/
def cayleyGraph {k : Nat} (S : Connection G k) : SimpleGraph G :=
  Cayley.Contract.cayleyGraph (S.1 : Set G)

/-- Equivalence under the action of `Aut(G)` on connection sets. -/
noncomputable def autSetoid (k : Nat) : Setoid (Connection G k) where
  r S T := ∃ automorphism : MulAut G,
    S.1.map automorphism.toEquiv.toEmbedding = T.1
  iseqv := by
    sorry

/-- Equivalence when the unlabelled Cayley graphs are isomorphic. -/
noncomputable def isoSetoid (k : Nat) : Setoid (Connection G k) where
  r S T := Nonempty (cayleyGraph S ≃g cayleyGraph T)
  iseqv := by
    sorry

/-- Number of `Aut(G)`-orbits on the fixed-size connection sets. -/
noncomputable def AutClassCount (G : Type*) [Group G] [Fintype G] [DecidableEq G]
    (k : Nat) : Nat :=
  Nat.card (Quotient (autSetoid (G := G) k))

/-- Number of unlabelled isomorphism classes of the fixed-size Cayley graphs. -/
noncomputable def IsoClassCount (G : Type*) [Group G] [Fintype G] [DecidableEq G]
    (k : Nat) : Nat :=
  Nat.card (Quotient (isoSetoid (G := G) k))

/-- The ordinary undirected CI property restricted to connection sets of cardinality `k`. -/
def IsCIAt (G : Type*) [Group G] [Fintype G] [DecidableEq G] (k : Nat) : Prop :=
  ∀ S T : Connection G k, Nonempty (cayleyGraph S ≃g cayleyGraph T) →
    ∃ automorphism : MulAut G, S.1.map automorphism.toEquiv.toEmbedding = T.1

/-- The ordinary undirected CI property for every simple Cayley graph on `G`. -/
def IsCIGroup (G : Type*) [Group G] [Fintype G] [DecidableEq G] : Prop :=
  ∀ k, IsCIAt G k

open Automorphisms.Contract

/-- The reference Aut-class count is the cardinality of the Mathlib orbit quotient. -/
theorem autClassCount_spec (xs : List G) (hnodup : xs.Nodup)
    (hcomplete : ∀ g : G, g ∈ xs) (k : Nat) :
    Cayley_iso.autClassCount (cayleyTable xs) k = AutClassCount G k := by
  sorry

/-- The reference graph-class count is the cardinality of the Mathlib isomorphism quotient. -/
theorem isoClassCount_spec (xs : List G) (hnodup : xs.Nodup)
    (hcomplete : ∀ g : G, g ∈ xs) (k : Nat) :
    Cayley_iso.isoClassCount (cayleyTable xs) k = IsoClassCount G k := by
  sorry

/-- Equality of the two finite quotients is equivalent to the fixed-size CI property because every
Aut(G)-class lies inside one graph-isomorphism class. -/
theorem isCI_spec (xs : List G) (hnodup : xs.Nodup)
    (hcomplete : ∀ g : G, g ∈ xs) (k : Nat) :
    Cayley_iso.isCI (cayleyTable xs) k = true ↔ IsCIAt G k := by
  sorry

/-- Checking every possible connection-set cardinality decides the CI-group property. -/
theorem isCIGroup_spec (xs : List G) (hnodup : xs.Nodup)
    (hcomplete : ∀ g : G, g ∈ xs) :
    Cayley_iso.isCIGroup (cayleyTable xs) = true ↔ IsCIGroup G := by
  sorry

end Cayley_iso.Contract
