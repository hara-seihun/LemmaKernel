import Lk.Reference

/-!
# code_equivalence: executable reference

A member is a generator matrix and denotes the code its rows span, so a code here is simply the
set of its codewords. A monomial map of `F_p^n` sends coordinate `j` of a word to `scale[j]`
times its coordinate `perm[j]`; the maps with every scale equal to one are the coordinate
permutations. Two codes are equivalent when some map carries one onto the other.

Every code of dimension `k` in `F_p^n` appears exactly once in `grassmannianMembers p n k`, so
that enumeration orders codes. The canonical representative of an equivalence class is its least
code in that order, and `canonicalForm` is the representative's rref basis.

The reference works with codeword sets rather than with elimination: the image of a code under a
monomial map is the image of each codeword, and the index of a code is where its codeword set
sits in the Grassmannian order. Nothing here needs Gauss-Jordan, which keeps `decide +kernel`
inside its budget on the oracle cases.
-/

namespace Code_equivalence

open Lk

/-- The codeword with coefficients `c` over the generator rows `m`, in `n` coordinates. -/
def combine (p n : Nat) (m : Mat) (c : Vec) : Vec :=
  (List.range n).map fun j =>
    (List.zipWith (fun a row => a * row.getD j 0) c m).foldl (· + ·) 0 % p

/-- Adjacent duplicates removed; on a sorted list this leaves each value once. -/
def dedupSorted [DecidableEq α] : List α → List α
  | [] => []
  | [x] => [x]
  | x :: y :: ys => if x = y then dedupSorted (y :: ys) else x :: dedupSorted (y :: ys)

/-- The code spanned by the rows of `m` in `n` coordinates: every codeword once, in
lexicographic order. Repeated and dependent generator rows change nothing. -/
def code (p n : Nat) (m : Mat) : List Vec :=
  dedupSorted (sortBy lexLe ((tuples p m.length).map (combine p n m)))

/-- Order on codes, so that a set of codewords has one canonical spelling. -/
def codeLe (a b : List Vec) : Bool := lexLe a.flatten b.flatten

/-- The image of one word: coordinate `j` reads coordinate `perm[j]`, scaled by `scale[j]`. -/
def monomialWord (p : Nat) (perm scale : Vec) (v : Vec) : Vec :=
  List.zipWith (fun j d => d * v.getD j 0 % p) perm scale

/-- The image code, spelled the same way as `code`. A monomial map is a bijection of `F_p^n`, so
the image of a set of distinct words is a set of distinct words. -/
def imageCode (p : Nat) (perm scale : Vec) (c : List Vec) : List Vec :=
  sortBy lexLe (c.map (monomialWord p perm scale))

/-- Every monomial map of `F_p^n` as a (permutation, scale) pair. `scalars = 1` gives the full
monomial group, of order `(p-1)^n n!`; `scalars = 0` gives the `n!` coordinate permutations. -/
def monomialMaps (p n scalars : Nat) : List (Vec × Vec) :=
  let scales := if scalars = 1 then (tuples (p - 1) n).map (fun d => d.map (· + 1))
                else [List.replicate n 1]
  (permutations (List.range n)).flatMap fun s => scales.map fun d => (s, d)

/-- The images of a code under every map: the orbit, with repeats. -/
def images (p n scalars : Nat) (c : List Vec) : List (List Vec) :=
  (monomialMaps p n scalars).map fun sd => imageCode p sd.1 sd.2 c

/-- The codes of dimension `h` in `F_p^n`, in Grassmannian order, each with the rref basis that
names it. A member's index is a scan of this table rather than an elimination; the scan stops at
the first hit, so only a prefix of the table is ever built. -/
def codeTable (p n h : Nat) : List (Mat × List Vec) :=
  (grassmannianMembers p n h).map fun b => (b, code p n b)

/-- The dimension of a code of `size` codewords: the `h ≤ n` with `p ^ h = size`. -/
def dimOf (p n size : Nat) : Nat := ((List.range (n + 1)).find? fun h => p ^ h = size).getD 0

/-- What every operation reads off one member: where its code sits in the Grassmannian order,
where its class representative sits, the representative's rref basis, and how many distinct
codes the class contains. -/
structure Class where
  index : Nat
  least : Nat
  form : Mat
  size : Nat

def classify (p n scalars : Nat) (m : Mat) : Class :=
  let c := code p n m
  let table := codeTable p n (dimOf p n c.length)
  let orbit := images p n scalars c
  let least := table.findIdx fun e => e.2 ∈ orbit
  { index := table.findIdx fun e => e.2 = c
    least := least
    form := (table.getD least ([], [])).1
    size := (dedupSorted (sortBy codeLe orbit)).length }

inductive Op
  | isCanonical (scalars : Nat)
  | canonicalIndex (scalars : Nat)
  | canonicalForm (scalars : Nat)
  | orbitSize (scalars : Nat)
  | autOrder (scalars : Nat)

/-- One materialised output per member. -/
inductive Value
  | basis (rows : List Vec)
  deriving DecidableEq, Repr

/-- Generator-matrix families. Words, ranges and the other natural-number kinds are not codes. -/
def supportedFamily : Family → Bool
  | .explicit _ _ | .subsets _ _ _ | .subsetsOf _ _ | .grassmannian _ _ _
  | .allMatrices _ _ _ | .symmetricMatrices _ _ | .transform _ _ | .stack _ _ => true
  | _ => false

def scalarsOf : Op → Nat
  | .isCanonical s | .canonicalIndex s | .canonicalForm s | .orbitSize s | .autOrder s => s

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let scalars := scalarsOf op
  if !supportedFamily f || scalars > 1 then .invalid else
  let p := f.p
  let ms := f.members
  let n := ((ms.headD []).headD []).length
  match op with
  | .isCanonical _ =>
    reduceBool red ms (ms.map fun m => let c := classify p n scalars m; c.index = c.least)
  | .canonicalIndex _ => reduceInt red ms (ms.map fun m => (classify p n scalars m).least)
  | .canonicalForm _ => reduceValues red (ms.map fun m => .basis (classify p n scalars m).form)
  | .orbitSize _ => reduceInt red ms (ms.map fun m => (classify p n scalars m).size)
  | .autOrder _ =>
    let order := (monomialMaps p n scalars).length
    reduceInt red ms (ms.map fun m => order / (classify p n scalars m).size)

end Code_equivalence
