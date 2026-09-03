import Lk.Reference

/-!
# graph_iso: executable reference

An input member is a nonempty symmetric square matrix over `F_2`, interpreted as an undirected
adjacency matrix. Diagonal entries are allowed, so the representation also covers graphs with
loops.

A vertex order `q` lists old vertices in new-label order. Its relabelled matrix has entry
`A[q[i]][q[j]]`. The canonical form is the lexicographically least row-major relabelling, and the
canonical label is the lexicographically least `q` attaining that form. The automorphism group is
returned as its complete lexicographically sorted list, rather than an arbitrary generating set.
The reference enumerates every permutation and uses only structural recursion so `decide +kernel`
can evaluate the small oracle cases.
-/

namespace Graph_iso

open Lk

/-- Insert `x` at every position. -/
def insertions (x : α) : List α → List (List α)
  | [] => [[x]]
  | y :: ys => (x :: y :: ys) :: (insertions x ys).map (y :: ·)

/-- Every permutation of a list. -/
def permutations : List α → List (List α)
  | [] => [[]]
  | x :: xs => (permutations xs).flatMap (insertions x)

/-- Strict lexicographic order on lists of naturals. -/
def lexLt : List Nat → List Nat → Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | x :: xs, y :: ys => if x < y then true else if y < x then false else lexLt xs ys

/-- `q` lists the old vertex at each new label. -/
def relabel (m : Mat) (q : Perm) : Mat :=
  q.map fun i => q.map fun j => (m.getD i []).getD j 0

def code (m : Mat) (q : Perm) : List Nat := (relabel m q).flatten

/-- Nonempty symmetric square adjacency matrix. -/
def validGraph (m : Mat) : Bool :=
  let n := m.length
  n > 0 && m.all (·.length = n) &&
    (List.range n).all fun i =>
      (List.range n).all fun j => (m.getD i []).getD j 0 = (m.getD j []).getD i 0

/-- Prefer the smaller relabelled matrix, then the smaller order when matrices tie. -/
def betterOrder (m : Mat) (q best : Perm) : Bool :=
  let cq := code m q
  let cb := code m best
  lexLt cq cb || (cq = cb && lexLt q best)

def canonicalLabel (m : Mat) : Perm :=
  let ps := permutations (List.range m.length)
  ps.foldl (fun best q => if betterOrder m q best then q else best) (List.range m.length)

def canonicalForm (m : Mat) : Mat := relabel m (canonicalLabel m)

/-- Every automorphism, lexicographically by image list. -/
def automorphisms (m : Mat) : List Perm :=
  sortBy lexLe ((permutations (List.range m.length)).filter fun q => relabel m q = m)

inductive Op
  | canonicalForm
  | canonicalLabel
  | automorphismGroup

inductive Value
  | matrix (m : Mat)
  | perm (q : Perm)
  | group (gs : List Perm)
  deriving DecidableEq, Repr

def acceptedFamily : Family → Bool
  | .explicit p batch => p = 2 && !batch.isEmpty && batch.all validGraph
  | .symmetricMatrices p n => p = 2 && n > 0
  | _ => false

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  if !acceptedFamily f then .invalid else
  let ms := f.members
  match op with
  | .canonicalForm => reduceValues red (ms.map fun m => .matrix (canonicalForm m))
  | .canonicalLabel => reduceValues red (ms.map fun m => .perm (canonicalLabel m))
  | .automorphismGroup => reduceValues red (ms.map fun m => .group (automorphisms m))

end Graph_iso
