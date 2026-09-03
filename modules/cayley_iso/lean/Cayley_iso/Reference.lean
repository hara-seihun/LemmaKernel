import Automorphisms.Reference
import Cayley.Reference

/-!
# cayley_iso: executable reference

Each member is a finite Cayley table. For the supplied `k`, the reference enumerates every
inverse-closed identity-free connection set of cardinality `k`. It counts equivalence classes
under table automorphisms and under isomorphism of the corresponding undirected Cayley graphs.
The implementation uses list folds and the structural permutation enumeration from the modules it
builds on. Oracle cases therefore keep the groups small.
-/

namespace Cayley_iso

open Lk

/-- The right regular permutations in Cayley-table label order. Their indexed multiplication is
exactly the table multiplication. -/
def regularElements (table : Mat) : List Perm :=
  (List.range table.length).map fun a =>
    (List.range table.length).map fun x => (table.getD x []).getD a 0

/-- Keep the first member of each class in input order. -/
def classRepresentatives (related : α → α → Bool) (members : List α) : List α :=
  members.foldl (fun representatives member =>
    if representatives.any (related member) then representatives
    else representatives ++ [member]) []

def classCount (related : α → α → Bool) (members : List α) : Nat :=
  (classRepresentatives related members).length

/-- Whether a table automorphism sends one connection set to the other. -/
def tableAutomorphic (n : Nat) (automorphisms : List Perm) (s t : List Nat) : Bool :=
  automorphisms.any fun automorphism =>
    Cayley.sameSet n (s.map fun x => automorphism.getD x 0) t

/-- The two class counts, with the Aut(G) count first. -/
def counts (table : Mat) (k : Nat) : Nat × Nat :=
  let elements := regularElements table
  let sets := Cayley.connectionSets elements k
  let automorphisms := Automorphisms.automorphisms table
  let autCount := classCount (tableAutomorphic table.length automorphisms) sets
  let isoCount := classCount (Cayley.graphIsomorphic elements) sets
  (autCount, isoCount)

def autClassCount (table : Mat) (k : Nat) : Nat := (counts table k).1

def isoClassCount (table : Mat) (k : Nat) : Nat := (counts table k).2

def isCI (table : Mat) (k : Nat) : Bool :=
  let result := counts table k
  result.1 == result.2

inductive Op
  | autClassCount (k : Nat)
  | isoClassCount (k : Nat)
  | isCi (k : Nat)
  | isNonCi (k : Nat)

inductive Value deriving DecidableEq, Repr


def run (op : Op) (family : Family) (red : Red) : Result Value :=
  match family with
  | .groupTables tables =>
      match op with
      | .autClassCount k => reduceInt red tables (tables.map fun table => autClassCount table k)
      | .isoClassCount k => reduceInt red tables (tables.map fun table => isoClassCount table k)
      | .isCi k => reduceBool red tables (tables.map fun table => isCI table k)
      | .isNonCi k => reduceBool red tables (tables.map fun table => !isCI table k)
  | _ => .invalid

end Cayley_iso
