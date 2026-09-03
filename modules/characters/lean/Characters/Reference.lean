import Orbits.Reference

/-!
# characters: executable reference

For a finite abelian permutation group, every ordinary irreducible character has degree one.
Such a character is a homomorphism from the group to the roots of unity. This reference closes
the group, finds its exponent `e`, and filters all maps `G -> Z/eZ` by the homomorphism law.
That direct definition is intentionally slow but gives the kernel a small, independent oracle.

A table cell is the sorted list of exponents of the eigenvalues of its representing matrix. Thus
`[a_1, ..., a_d]` denotes `zeta_e^a_1 + ... + zeta_e^a_d`. All rows here have degree one. The
encoding also works for higher-degree characters without choosing a basis for a cyclotomic field.
-/

namespace Characters

open Lk

def validPerm (n : Nat) (g : Perm) : Bool :=
  g.length = n ∧ (List.range n).all fun i => (g.filter (· = i)).length = 1

def permPow (g : Perm) : Nat → Perm
  | 0 => identityPerm g.length
  | k + 1 => compose (permPow g k) g

def elementOrder (g : Perm) : Nat :=
  ((List.range g.length).map (· + 1)).find? (fun k => permPow g k = identityPerm g.length) |>.getD 1

def gcdAux : Nat → Nat → Nat → Nat
  | 0, a, _ => a
  | fuel + 1, a, b => if b = 0 then a else gcdAux fuel b (a % b)

def lcm (a b : Nat) : Nat := if a = 0 ∨ b = 0 then 0 else a / gcdAux 128 a b * b

def groupExponent (elements : List Perm) : Nat := elements.foldl (fun e g => lcm e (elementOrder g)) 1

def multiplicationIndex (elements : List Perm) (i j : Nat) : Nat :=
  elements.idxOf (compose (elements.getD i []) (elements.getD j []))

def isAbelian (elements : List Perm) : Bool :=
  (List.range elements.length).all fun i =>
    (List.range elements.length).all fun j => multiplicationIndex elements i j = multiplicationIndex elements j i

def validHom (elements : List Perm) (conductor : Nat) (values : List Nat) : Bool :=
  let identity := elements.idxOf (identityPerm (elements.headD []).length)
  values.length = elements.length ∧ values.getD identity 0 = 0 ∧
    (List.range elements.length).all fun i =>
      (List.range elements.length).all fun j =>
        values.getD (multiplicationIndex elements i j) 0 =
          (values.getD i 0 + values.getD j 0) % conductor

def homomorphisms (elements : List Perm) (conductor : Nat) : List (List Nat) :=
  (tuples conductor elements.length).filter (validHom elements conductor)

structure Data where
  elements : List Perm
  conductor : Nat
  homs : List (List Nat)

def characterData (gens : List Perm) : Option Data :=
  if gens.isEmpty then none
  else
    let degree := (gens.headD []).length
    if !gens.all (validPerm degree) then none
    else
      let elements := permElements gens
      if !isAbelian elements then none
      else
        let conductor := groupExponent elements
        let homs := homomorphisms elements conductor
        if conductor = 0 ∨ homs.length ≠ elements.length then none
        else some { elements, conductor, homs }

abbrev TableData :=
  (Nat × Nat) × (List Nat × List Nat) × (List Nat × List (List (List Nat)))

def Data.table (d : Data) : TableData :=
  let order := d.elements.length
  ((order, d.conductor), (List.range order, List.replicate order 1),
    (List.replicate order 1, d.homs.map fun row => row.map fun exponent => [exponent]))

def Data.indicators (d : Data) : List Int :=
  d.homs.map fun row => if row.all (fun exponent => 2 * exponent % d.conductor = 0) then 1 else 0

def groupData : Orbits.Group → Option Data
  | .perms gens => characterData gens
  | .mats _ _ => none

def isSubgroup (ambient subgroup : Data) : Bool :=
  subgroup.elements.all (· ∈ ambient.elements)

def sameRestriction (ambient subgroup : Data) (ambientRow subgroupRow : List Nat) : Bool :=
  let conductor := lcm ambient.conductor subgroup.conductor
  (List.range subgroup.elements.length).all fun i =>
    let ambientIndex := ambient.elements.idxOf (subgroup.elements.getD i [])
    ambientRow.getD ambientIndex 0 * (conductor / ambient.conductor) % conductor =
      subgroupRow.getD i 0 * (conductor / subgroup.conductor) % conductor

def restrictionMultiplicities (ambient subgroup : Data) (character : Nat) : List Nat :=
  let row := ambient.homs.getD character []
  subgroup.homs.map fun candidate => if sameRestriction ambient subgroup row candidate then 1 else 0

def inductionMultiplicities (ambient subgroup : Data) (character : Nat) : List Nat :=
  let row := subgroup.homs.getD character []
  ambient.homs.map fun candidate => if sameRestriction ambient subgroup candidate row then 1 else 0

inductive Op
  | characterTable
  | frobeniusSchur
  | restrict (character : Nat) (subgroup : Orbits.Group)
  | induce (character : Nat) (subgroup : Orbits.Group)

inductive Value
  | table (data : TableData)
  | indicator (value : Int)
  | multiplicity (value : Nat)
  deriving DecidableEq, Repr

def run (op : Op) (family : Family) (red : Red) : Result Value :=
  match family, red with
  | .groupElements p gens, .all =>
    if p ≠ 0 then .invalid else
    match characterData (gens.map (·.headD [])) with
    | none => .invalid
    | some ambient =>
      match op with
      | .characterTable => .values [.table ambient.table]
      | .frobeniusSchur => .values (ambient.indicators.map .indicator)
      | .restrict character subgroup =>
        match groupData subgroup with
        | some sub =>
          if isSubgroup ambient sub ∧ character < ambient.homs.length then
            .values ((restrictionMultiplicities ambient sub character).map .multiplicity)
          else .invalid
        | none => .invalid
      | .induce character subgroup =>
        match groupData subgroup with
        | some sub =>
          if isSubgroup ambient sub ∧ character < sub.homs.length then
            .values ((inductionMultiplicities ambient sub character).map .multiplicity)
          else .invalid
        | none => .invalid
  | _, _ => .invalid

end Characters
