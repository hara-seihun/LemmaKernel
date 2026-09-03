import Mathlib
import Graph_iso.Reference
import Lk.Contract

/-!
# graph_iso: contract

The reference treats a graph as a symmetric matrix over `ZMod 2`. A list `q` records the old
vertex at each new label, so relabelling applies the same permutation to rows and columns.
Canonical forms use the least row-major list in Lean's natural-number lexicographic order.
Automorphism groups contain every element, sorted by its image list, so no generator choice is
left unspecified.
-/

namespace Graph_iso.Contract

open Lk Lk.Contract

/-- The image list of a Mathlib permutation. -/
def permList {n : ℕ} (σ : Equiv.Perm (Fin n)) : Perm :=
  List.ofFn fun i => (σ i : ℕ)

/-- The executable relabelling is simultaneous Mathlib matrix reindexing. -/
theorem relabel_spec {n : ℕ} {m : Mat} (h : WellFormed 2 n n m) (σ : Equiv.Perm (Fin n)) :
    toMatrix 2 n n (relabel m (permList σ)) =
      Matrix.reindex σ.symm σ.symm (toMatrix 2 n n m) := by
  sorry

/-- The canonical label is a permutation, produces the canonical form, and is the least order
that produces that form. -/
theorem canonicalLabel_spec {n : ℕ} {m : Mat} (h : WellFormed 2 n n m) :
    (canonicalLabel m).Perm (List.range n) ∧
    canonicalForm m = relabel m (canonicalLabel m) ∧
    ∀ q, q.Perm (List.range n) → relabel m q = canonicalForm m →
      lexLe (canonicalLabel m) q = true := by
  sorry

/-- The canonical form is an isomorphic Mathlib adjacency matrix and is no larger than any
simultaneous row-column reindexing. -/
theorem canonicalForm_spec {n : ℕ} {m : Mat} (h : WellFormed 2 n n m)
    (hsymm : (toMatrix 2 n n m).IsSymm) :
    (∃ σ : Equiv.Perm (Fin n),
      toMatrix 2 n n (canonicalForm m) =
        Matrix.reindex σ.symm σ.symm (toMatrix 2 n n m)) ∧
    ∀ σ : Equiv.Perm (Fin n),
      lexLe (canonicalForm m).flatten (relabel m (permList σ)).flatten = true := by
  sorry

/-- Two well-formed symmetric matrices have the same canonical form exactly when a Mathlib
permutation reindexes one to the other. -/
theorem canonical_complete {n : ℕ} {a b : Mat}
    (ha : WellFormed 2 n n a) (hb : WellFormed 2 n n b)
    (hsa : (toMatrix 2 n n a).IsSymm) (hsb : (toMatrix 2 n n b).IsSymm) :
    canonicalForm a = canonicalForm b ↔
      ∃ σ : Equiv.Perm (Fin n),
        toMatrix 2 n n b = Matrix.reindex σ.symm σ.symm (toMatrix 2 n n a) := by
  sorry

/-- The returned list is the whole automorphism group, without duplicates and in lexicographic
order. Membership is exactly stabilization of the Mathlib adjacency matrix. -/
theorem automorphisms_spec {n : ℕ} {m : Mat} (h : WellFormed 2 n n m)
    (hsymm : (toMatrix 2 n n m).IsSymm) :
    (automorphisms m).Nodup ∧
    (automorphisms m).Pairwise (fun a b => lexLe a b = true) ∧
    ∀ q, q ∈ automorphisms m ↔
      ∃ σ : Equiv.Perm (Fin n), q = permList σ ∧
        Matrix.reindex σ.symm σ.symm (toMatrix 2 n n m) = toMatrix 2 n n m := by
  sorry

end Graph_iso.Contract
