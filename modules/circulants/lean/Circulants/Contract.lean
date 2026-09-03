import Mathlib
import Circulants.Reference
import Lk.Contract

/-!
# circulants: contract

This file relates the executable formulas to Mathlib matrices, finite cyclic groups, and graph
isomorphism. The corrected Adam classification is the cited mathematical input behind
`isomorphic`, `is_canonical`, and `is_ci`. Open proofs are recorded as `sorry`.
-/

namespace Circulants.Contract

open Matrix Lk
open scoped ComplexConjugate

/-- The effective connection set used by the requested graph type. -/
def connectionSet (n directed : Nat) (s : Vec) : Set (Fin n) :=
  {x | (x : Nat) ∈ effectiveSet n directed s}

/-- Adjacency matrix of the directed Cayley relation `y - x in S`. When `directed = 0`,
`effectiveSet` makes it the adjacency matrix of Mathlib's undirected circulant graph. -/
def adjacencyMatrix (n directed : Nat) (s : Vec) : Matrix (Fin n) (Fin n) ℂ :=
  fun x y => if ((y : Nat) + n - (x : Nat)) % n ∈ effectiveSet n directed s then 1 else 0

/-- Character `j` of `Z_n`, written using a chosen primitive root. -/
def character {n : Nat} (ζ : ℂ) (j : Fin n) : Fin n → ℂ :=
  fun x => ζ ^ ((j : Nat) * (x : Nat))

/-- Interpret one returned exponent row as an exact cyclotomic sum. -/
def evaluateRow (ζ : ℂ) (row : List Nat) : ℂ :=
  row.foldr (fun e total => ζ ^ e + total) 0

/-- The undirected relation agrees with Mathlib's `SimpleGraph.circulantGraph`. -/
theorem undirected_adjacency_spec (n : Nat) [NeZero n] (s : Vec) (x y : Fin n) :
    adjacencyMatrix n 0 s x y = 1 ↔
      (SimpleGraph.circulantGraph (connectionSet n 0 s)).Adj x y := by
  sorry

/-- Every returned row evaluates to the eigenvalue of the corresponding additive character.
This states the closed-form diagonalisation, not a floating-point approximation. -/
theorem spectrum_spec (n directed : Nat) (s : Vec)
    (hmode : validMode directed = true) (hvalid : validConnection n s = true)
    (ζ : ℂ) (hζ : IsPrimitiveRoot ζ n) :
    let exponents := spectrumExponents n directed s
    ∀ j : Fin n,
      adjacencyMatrix n directed s *ᵥ character ζ j =
        fun x => evaluateRow ζ (exponents.getD j []) * character ζ j x := by
  sorry

/-- Graph or digraph isomorphism after identifying both vertex sets with `Fin n`. -/
def Isomorphic (n directed : Nat) (s t : Vec) : Prop :=
  ∃ e : Equiv (Fin n) (Fin n), ∀ x y,
    adjacencyMatrix n directed s x y = adjacencyMatrix n directed t (e x) (e y)

/-- At a corrected-Adam order, the multiplier test is the full isomorphism test. The three
extra undirected orders are the exceptional cyclic CI-groups. -/
theorem isomorphic_spec (n directed : Nat) (s t : Vec)
    (hmode : validMode directed = true) (hs : validConnection n s = true)
    (ht : validConnection n t = true) (horder : ciOrder n directed = true) :
    multiplierEquivalent n (effectiveSet n directed s) (effectiveSet n directed t) = true ↔
      Isomorphic n directed s t := by
  sorry

/-- A cyclic group is CI for this graph type when every isomorphic pair of its Cayley relations
is carried by a group automorphism, which is multiplication by a unit. -/
def IsCI (n directed : Nat) : Prop :=
  ∀ s t : Vec, validConnection n s = true → validConnection n t = true →
    Isomorphic n directed s t →
    multiplierEquivalent n (effectiveSet n directed s) (effectiveSet n directed t) = true

/-- Muzychuk's cyclic classification: DCI orders have no factor 8 or odd prime square. Ordinary
CI has the same orders plus 8, 9, and 18. -/
theorem isCi_spec (n directed : Nat) (hn : 0 < n) (hmode : validMode directed = true) :
    ciOrder n directed = true ↔ IsCI n directed := by
  sorry

/-- A reported canonical connection set is the unique inverse-closed representation in the
undirected case and is lexicographically no larger than every unit multiple. -/
theorem canonical_spec (n directed : Nat) (s : Vec)
    (h : let raw := sortNats s
         let effective := effectiveSet n directed raw
         (directed = 1 || raw = effective) && multiplierCanonical n effective = true) :
    let effective := effectiveSet n directed s
    (directed = 1 ∨ sortNats s = effective) ∧
      ∀ a ∈ units n, lexLe effective (multiplierImage n a effective) = true := by
  sorry

end Circulants.Contract
