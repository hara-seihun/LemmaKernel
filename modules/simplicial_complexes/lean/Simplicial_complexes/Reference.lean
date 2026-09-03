import Gfp.Reference

/-!
# simplicial_complexes: executable reference

A member of a family is a 0/1 matrix over `F_2`. Row `r` stands for the set of columns where it
is 1, and the vertex set is `0..n-1` for `n` columns. `nonfaces = 0` reads those sets as
generators, so the complex is everything below them; `nonfaces = 1` reads them as forbidden sets,
so the complex is everything containing none of them (weight-two rows are the edges of a graph
and the complex is its independence complex).

A face is a *nonempty* set of the complex, so the homology below is unreduced: `beta 0` counts
connected components. Faces are listed by increasing number of vertices, lexicographically
within each size; the boundary matrix of dimension `d` has one row per `d`-face and one column
per `(d-1)`-face, and `Gfp.rank` over `F_p` does the linear algebra.

Everything is plain structural recursion so that `decide +kernel` can evaluate it; a kernel
answer is correct when `example : run op family red = answer := by decide` is accepted.
-/

namespace Simplicial_complexes

open Lk

/-- A face: its vertices in increasing order. -/
abbrev Face := List Nat

/-! ## Building the complex -/

/-- The columns where a row is nonzero, increasing. -/
def support (row : Vec) : Face :=
  (List.range row.length).filter fun j => row.getD j 0 ≠ 0

/-- One set per row of the member. -/
def generators (m : Mat) : List Face := m.map support

/-- Is `s` contained in `t`? Both are sets of vertices. -/
def subsetOf (s t : Face) : Bool := s.all (· ∈ t)

/-- Membership of the complex the member describes, for a nonempty candidate `T`. -/
def isFace (gens : List Face) (nonfaces : Nat) (T : Face) : Bool :=
  if nonfaces = 0 then gens.any (subsetOf T ·) else gens.all fun S => !subsetOf S T

/-- Every nonempty subset of `0..n-1`, by increasing size and lexicographically within a size:
the candidates a complex on `n` vertices is filtered out of. Built once per request. -/
def candidates (n : Nat) : List Face :=
  (List.range n).flatMap fun k => combos (List.range n) (k + 1)

/-- Every nonempty face, in the order of `candidates`. -/
def facesOf (cands gens : List Face) (nonfaces : Nat) : List Face :=
  cands.filter (isFace gens nonfaces)

/-! ## The f-vector and the Euler characteristic -/

/-- `f_d`: the faces with `d + 1` vertices. -/
def fCount (fs : List Face) (d : Nat) : Nat := (fs.filter fun F => F.length = d + 1).length

/-- `sum_i (-1)^i f_i` in `F_p`: faces of even dimension have an odd number of vertices. -/
def eulerMod (p : Nat) (fs : List Face) : Nat :=
  let even := (fs.filter fun F => F.length % 2 = 1).length
  let odd := (fs.filter fun F => F.length % 2 = 0).length
  (even % p + (p - odd % p)) % p

/-! ## Boundary maps and Betti numbers -/

/-- The row of the boundary map for face `F`: `(-1)^j` in the column of `F` without its `j`-th
vertex, zero elsewhere. -/
def boundaryRow (p : Nat) (lower : List Face) (F : Face) : Vec :=
  lower.map fun G =>
    match (List.range F.length).find? fun j => F.eraseIdx j = G with
    | some j => if j % 2 = 0 then 1 % p else p - 1
    | none => 0

/-- The matrix of `boundary_d : C_d -> C_{d-1}`: one row per face with `d + 1` vertices, one
column per face with `d` vertices. `boundary_0` is the zero map. -/
def boundary (p : Nat) (fs : List Face) (d : Nat) : Mat :=
  match d with
  | 0 => []
  | d + 1 =>
    let upper := fs.filter fun F => F.length = d + 2
    let lower := fs.filter fun F => F.length = d + 1
    upper.map (boundaryRow p lower)

def boundaryRank (p : Nat) (fs : List Face) (d : Nat) : Nat := Gfp.rank p (boundary p fs d)

/-- `dim_{F_p} H_d`, unreduced: `f_d - rank(boundary_d) - rank(boundary_{d+1})`. -/
def betti (p : Nat) (fs : List Face) (d : Nat) : Nat :=
  fCount fs d - boundaryRank p fs d - boundaryRank p fs (d + 1)

/-! ## Shellability -/

/-- The maximal faces. -/
def facetsOf (fs : List Face) : List Face :=
  fs.filter fun F => !fs.any fun G => G.length > F.length && subsetOf F G

/-- Bjorner-Wachs: facet `j` may follow the facets already chosen when each of those meets `F_j`
inside a chosen facet that misses exactly one vertex of `F_j`. Which facets `k` witness that for
a pair `(i, j)` does not depend on the order, so they are collected once, as a bit mask of facet
indices: `follow[i][j]` are the `k` with `F_i ∩ F_j ⊆ F_k` and `|F_j \ F_k| = 1`. -/
def followMasks (facets : List Face) : List (List Nat) :=
  facets.map fun Fi => facets.map fun Fj =>
    (List.range facets.length).foldl (fun acc k =>
      let Fk := facets.getD k []
      if subsetOf (Fi.filter (· ∈ Fj)) Fk && (Fj.filter fun v => !(v ∈ Fk)).length = 1
      then acc ||| (1 <<< k) else acc) 0

/-- With the facets of `chosen` placed, may facet `j` come next? -/
def mayFollow (follow : List (List Nat)) (t chosen j : Nat) : Bool :=
  (List.range t).all fun i =>
    !chosen.testBit i || ((follow.getD i []).getD j 0 &&& chosen) ≠ 0

/-- One breadth-first layer: every set of facets reachable from `layer` by placing one more.
Whether a facet may come next depends only on the set already placed, not on its order, so the
search runs over subsets of the facets and not over their orders. -/
def shellStep (follow : List (List Nat)) (t : Nat) (layer : List Nat) : List Nat :=
  layer.foldl (fun acc chosen =>
    (List.range t).foldl (fun acc j =>
      if chosen.testBit j || !mayFollow follow t chosen j then acc
      else
        let next := chosen ||| (1 <<< j)
        if next ∈ acc then acc else acc ++ [next]) acc) []

def shellReach (follow : List (List Nat)) (t : Nat) : Nat → List Nat → List Nat
  | 0, layer => layer
  | fuel + 1, layer => shellReach follow t fuel (shellStep follow t layer)

/-- Shellable when all `t` facets can be placed: the full set is reachable in `t` layers. Note
that `t ≤ 1` is shellable, the empty set of facets included. -/
def isShellable (fs : List Face) : Bool :=
  let facets := facetsOf fs
  let t := facets.length
  2 ^ t - 1 ∈ shellReach (followMasks facets) t t [0]

/-! ## Validity -/

/-- No divisor of `p` in `d, d+1, ...` up to `sqrt p`; `fuel` bounds the loop. -/
def noDivisorFrom (p : Nat) : Nat → Nat → Bool
  | 0, _ => true
  | fuel + 1, d => if p < d * d then true else if p % d = 0 then false else noDivisorFrom p fuel (d + 1)

/-- Trial division. The runtime accepts primes below `2 ^ 32`. -/
def isPrime (p : Nat) : Bool := 2 ≤ p && p < 4294967296 && noDivisorFrom p p 2

/-- Members must be 0/1 matrices over `F_2` and the reading must be one of the two. -/
def accepts (f : Family) (nonfaces : Nat) : Bool := f.p = 2 && nonfaces ≤ 1

/-! ## Operations and values -/

/-- Arguments come in the order the runtime passes them, which is alphabetical by name. -/
inductive Op
  | fCount (dim nonfaces : Nat)
  | faces (nonfaces : Nat)
  | eulerCharacteristic (nonfaces p : Nat)
  | betti (dim nonfaces p : Nat)
  | isShellable (nonfaces : Nat)

/-- No operation of this module materialises a per-member value; every answer is an integer or
a boolean, which the runtime's reductions carry. -/
inductive Value
  | none
  deriving DecidableEq, Repr

/-- The complex of every member, in family order. -/
def complexes (f : Family) (nonfaces : Nat) : List (List Face) :=
  let cands := candidates ((f.members.headD []).headD []).length
  f.members.map fun m => facesOf cands (generators m) nonfaces

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let ms := f.members
  match op with
  | .fCount d nf =>
    if !accepts f nf then .invalid else reduceInt red ms ((complexes f nf).map (fCount · d))
  | .faces nf =>
    if !accepts f nf then .invalid else reduceInt red ms ((complexes f nf).map List.length)
  | .eulerCharacteristic nf p =>
    if !accepts f nf || !isPrime p then .invalid
    else reduceInt red ms ((complexes f nf).map (eulerMod p))
  | .betti d nf p =>
    if !accepts f nf || !isPrime p then .invalid
    else reduceInt red ms ((complexes f nf).map fun fs => betti p fs d)
  | .isShellable nf =>
    if !accepts f nf then .invalid else reduceBool red ms ((complexes f nf).map isShellable)

end Simplicial_complexes
