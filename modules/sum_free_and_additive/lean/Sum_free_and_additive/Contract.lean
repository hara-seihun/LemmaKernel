import Mathlib
import Sum_free_and_additive.Reference
import Lk.Contract

/-!
# sum_free_and_additive: contract

What the executable reference computes, said in Mathlib's terms. A member is a list of natural
numbers; the ambient group is `ZMod modulus`, which is the integers when `modulus = 0`, so one
statement covers both readings the module offers.

The reference is the test oracle; this file is what a caller relies on when they believe an
answer. The `sorry` declarations are the proofs that the reference has these meanings, written
down and not yet discharged.
-/

namespace Sum_free_and_additive.Contract

open Finset
open scoped Pointwise

/-- The set a member denotes: its elements in the ambient group. -/
def toFinset (modulus : ℕ) (xs : List ℕ) : Finset (ZMod modulus) :=
  (xs.map (Nat.cast : ℕ → ZMod modulus)).toFinset

/-- What the module requires of a member: distinct elements, already inside `Z/modulus`. -/
def WellFormedSet (modulus : ℕ) (xs : List ℕ) : Prop :=
  xs.Nodup ∧ (modulus = 0 ∨ ∀ x ∈ xs, x < modulus)

/-- Ordered pairs of `A` with a given difference. -/
def multiplicity {modulus : ℕ} (A : Finset (ZMod modulus)) (d : ZMod modulus) : ℕ :=
  ((A ×ˢ A).filter fun p => p.1 - p.2 = d).card

/-- `A` contains the progression of `length` terms from `a` with common difference `d`. -/
def HasProgression {modulus : ℕ} (A : Finset (ZMod modulus)) (length : ℕ)
    (a d : ZMod modulus) : Prop :=
  ∀ i < length, a + i • d ∈ A

variable {modulus length num den : ℕ} {xs : List ℕ}

/-- A well-formed member lists its set once, so `|S|` is the number of rows. -/
theorem card_toFinset (h : WellFormedSet modulus xs) :
    (toFinset modulus xs).card = xs.length := by
  sorry

theorem sumsetSize_spec (h : WellFormedSet modulus xs) :
    sumsetSize modulus xs = (toFinset modulus xs + toFinset modulus xs).card := by
  sorry

theorem differenceSetSize_spec (h : WellFormedSet modulus xs) :
    differenceSetSize modulus xs = (toFinset modulus xs - toFinset modulus xs).card := by
  sorry

theorem schurTripleCount_spec (h : WellFormedSet modulus xs) :
    let A := toFinset modulus xs
    schurTripleCount modulus xs = ((A ×ˢ A).filter fun p => p.1 + p.2 ∈ A).card := by
  sorry

theorem isSumFree_spec (h : WellFormedSet modulus xs) :
    let A := toFinset modulus xs
    isSumFree modulus xs = true ↔ ∀ a ∈ A, ∀ b ∈ A, a + b ∉ A := by
  sorry

/-- Sidon, in the `B_2` form: a sum determines its pair of summands. -/
theorem isSidon_spec (h : WellFormedSet modulus xs) :
    let A := toFinset modulus xs
    isSidon modulus xs = true ↔
      ∀ a ∈ A, ∀ b ∈ A, ∀ c ∈ A, ∀ d ∈ A, a + b = c + d → (a = c ∧ b = d) ∨ (a = d ∧ b = c) := by
  sorry

theorem maxDifferenceMultiplicity_spec (h : WellFormedSet modulus xs) :
    let A := toFinset modulus xs
    maxDifferenceMultiplicity modulus xs = (((A - A).erase 0).sup (multiplicity A)) := by
  sorry

/-- Progression-freedom asks only that the common difference is nonzero: in `Z/n` a difference
whose multiples wrap gives a progression with repeated terms, and that counts. -/
theorem isApFree_spec (h : WellFormedSet modulus xs) (hlen : 2 ≤ length) :
    let A := toFinset modulus xs
    isApFree length modulus xs = true ↔
      ∀ a : ZMod modulus, ∀ d : ZMod modulus, d ≠ 0 → ¬ HasProgression A length a d := by
  sorry

/-- The `extends_*` operations are the predicates above on `context ++ xs`; `validContext` makes
that concatenation well-formed whenever the member is, so every statement here applies to it. -/
theorem extends_wellFormed {context : List ℕ} (hc : WellFormedSet modulus context)
    (h : WellFormedSet modulus xs) (hd : ∀ x ∈ xs, x ∉ context) :
    WellFormedSet modulus (context ++ xs) := by
  refine ⟨List.nodup_append.mpr ⟨hc.1, h.1, by intro a ha b hb hab; subst hab; exact hd a hb ha⟩, ?_⟩
  rcases h.2 with h0 | h'
  · exact Or.inl h0
  rcases hc.2 with h0 | hc'
  · exact Or.inl h0
  refine Or.inr fun x hx => ?_
  rcases List.mem_append.mp hx with hx | hx
  · exact hc' x hx
  · exact h' x hx

/-- At three terms this is Mathlib's `ThreeAPFree`. -/
theorem isApFree_three (h : WellFormedSet modulus xs) :
    isApFree 3 modulus xs = true ↔ ThreeAPFree (toFinset modulus xs : Set (ZMod modulus)) := by
  sorry

/-- The doubling constant `σ[A] = |A + A| / |A|`, compared without leaving the integers. -/
theorem isSmallDoubling_spec (h : WellFormedSet modulus xs) (hden : 1 ≤ den) :
    let A := toFinset modulus xs
    isSmallDoubling modulus num den xs = true ↔ Finset.addConst A A ≤ (num : ℚ≥0) / (den : ℚ≥0) := by
  sorry

end Sum_free_and_additive.Contract
