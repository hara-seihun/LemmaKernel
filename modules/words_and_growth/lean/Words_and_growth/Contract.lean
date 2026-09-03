import Mathlib
import Words_and_growth.Reference

/-!
# words_and_growth: contract

The encoded equations define a Mathlib `PresentedGroup` on `Fin generators`. The executable
validator requires the oriented equations and implicit inverse cancellations to form a finite
complete shortlex-decreasing rewriting system. Under that check, normal forms decide equality in
the presented group and have minimum word length.
-/

namespace Words_and_growth.Contract

open Lk Words_and_growth


def evalLetter {g : Nat} (letter : Nat) : FreeGroup (Fin g) :=
  if h : letter / 2 < g then
    let generator := FreeGroup.of (⟨letter / 2, h⟩ : Fin g)
    if letter % 2 = 0 then generator else generator⁻¹
  else 1

def evalWord {g : Nat} (w : Word) : FreeGroup (Fin g) :=
  w.foldl (fun value letter => value * evalLetter letter) 1

/-- Each encoded equation `left = right` contributes the relator `left * right⁻¹`. -/
def relationSet (g : Nat) (encoded : List Vec) : Set (FreeGroup (Fin g)) :=
  { relator | ∃ equations, decodeEquations (2 * g) encoded = some equations ∧
      ∃ equation ∈ equations,
        relator = evalWord equation.left * (evalWord equation.right)⁻¹ }

abbrev Presented (g : Nat) (encoded : List Vec) := PresentedGroup (relationSet g encoded)

def classOf (g : Nat) (encoded : List Vec) (w : Word) : Presented g encoded :=
  PresentedGroup.mk (relationSet g encoded) (evalWord w)

def ValidWord (g : Nat) (w : Word) : Prop := ∀ letter ∈ w, letter < 2 * g

def Geodesic (g : Nat) (encoded : List Vec) (w : Word) : Prop :=
  ValidWord g w ∧ ∀ v, ValidWord g v → classOf g encoded v = classOf g encoded w → w.length ≤ v.length

def sphere (g : Nat) (encoded : List Vec) (n : Nat) : Set (Presented g encoded) :=
  { x | ∃ w, Geodesic g encoded w ∧ w.length = n ∧ classOf g encoded w = x }

def ball (g : Nat) (encoded : List Vec) (n : Nat) : Set (Presented g encoded) :=
  { x | ∃ w, ValidWord g w ∧ w.length ≤ n ∧ classOf g encoded w = x }

def functionWord {q n : Nat} (w : Fin n → Fin q) : Word :=
  List.ofFn fun i => (w i).val

/-- The validated normal form represents the same element and is a shortest representative. -/
theorem normalise_spec {g : Nat} {encoded : List Vec} {rules : List Rule} {w nf : Word}
    (hbuild : buildRules g encoded = some rules) (hw : ValidWord g w)
    (hnf : normalise (2 * g) rules w = some nf) :
    classOf g encoded w = classOf g encoded nf ∧ Geodesic g encoded nf := by
  sorry

theorem isGeodesic_spec {g : Nat} {encoded : List Vec} {rules : List Rule} {w : Word}
    (hbuild : buildRules g encoded = some rules) (hw : ValidWord g w) :
    isGeodesicWord (2 * g) rules w = true ↔ Geodesic g encoded w := by
  sorry

theorem sphereSize_spec {g : Nat} {encoded : List Vec} {rules : List Rule} (n : Nat)
    (hbuild : buildRules g encoded = some rules) :
    sphereSize (2 * g) rules n = Set.ncard (sphere g encoded n) := by
  sorry

theorem ballSize_spec {g : Nat} {encoded : List Vec} {rules : List Rule} (n : Nat)
    (hbuild : buildRules g encoded = some rules) :
    ballSize (2 * g) rules n = Set.ncard (ball g encoded n) := by
  sorry

theorem geodesicCount_spec {g : Nat} {encoded : List Vec} {rules : List Rule} (n : Nat)
    (hbuild : buildRules g encoded = some rules) :
    geodesicCount (2 * g) rules n =
      Set.ncard { w : Fin n → Fin (2 * g) | Geodesic g encoded (functionWord w) } := by
  sorry

end Words_and_growth.Contract
