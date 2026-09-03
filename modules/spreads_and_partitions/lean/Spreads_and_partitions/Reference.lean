import Lk.Reference

/-!
# spreads_and_partitions: executable reference

The meaning of every operation of the module, written as plain structural recursion so that
Lean's kernel can evaluate it. Families, reductions and the result shape come from
`Lk.Reference`. This is the oracle: a kernel answer is correct when
`example : run op family red = answer := by decide +kernel` is accepted.

A member is a set of subspaces of `F_p^n`. The ambient dimension `n` comes from the operation;
each row of the member is one component, read row-major as an `h x n` matrix with `h = cols / n`,
and means the row space of that matrix. So a row of rank `d < h` is a `d`-dimensional component,
and two rows spanning the same subspace are two components meeting in that subspace.

Elimination is written here rather than reused from `Gfp.Reference`: this module only ever needs
the rank and the canonical basis of a row space, never the transform matrix `Gfp.gaussJordan`
carries alongside, and the kernel time saved is the budget for the oracle cases.
-/

namespace Spreads_and_partitions

open Lk

/-! ## Echelon bases over `F_p`

A basis here is a list of rows sorted by increasing leading column, each with leading entry 1 and
no two sharing a leading column; its length is the rank of the space it spans. -/

/-- The first nonzero column of `v`, or `v.length` when `v` is zero. -/
def lead : Vec → Nat
  | [] => 0
  | x :: xs => if x = 0 then 1 + lead xs else 0

/-- `v` reduced against a basis. The rows are sorted by increasing leading column, so clearing
them in order never reintroduces a column already cleared: one pass is enough. -/
def reduceVec (p : Nat) (basis : Mat) (v : Vec) : Vec :=
  basis.foldl (fun r b => let c := r.getD (lead b) 0; if c = 0 then r else subMul p r b c) v

/-- `basis` with `v` added, which lengthens it exactly when `v` is outside its span. -/
def insertVec (p : Nat) (basis : Mat) (v : Vec) : Mat :=
  let r := reduceVec p basis v
  if lead r = r.length then basis
  else insertBy (fun a b => decide (lead a ≤ lead b)) (scale p r (inv p (r.getD (lead r) 0))) basis

/-- A basis of the row space of `rows`; `.length` is the rank. -/
def echelon (p : Nat) (rows : Mat) : Mat := rows.foldl (insertVec p) []

/-- The reduced row echelon basis: back-substitution over `echelon`, from the last row up, so
every row is reduced by the rows below it. This is the canonical form of a row space, and the
only thing that decides when two components are the same subspace. -/
def rrefOf (p : Nat) (rows : Mat) : Mat :=
  (echelon p rows).foldr (fun r acc => reduceVec p acc r :: acc) []

/-! ## Components of a member -/

/-- The components a member stands for, as bases: row `i` read as an `h x n` matrix with
`h = cols / n`. -/
def components (p n : Nat) (m : Mat) : List Mat :=
  m.map fun row => echelon p (chunk n (row.length / n) row)

/-- Unordered pairs of a list, `(x_0, x_1), (x_0, x_2), ..., (x_1, x_2), ...`. -/
def pairsOf {α : Type} : List α → List (α × α)
  | [] => []
  | x :: xs => xs.map (fun y => (x, y)) ++ pairsOf xs

/-- Two subspaces given by bases meet only in `0` exactly when their ranks add. -/
def meetsTrivially (p : Nat) (a b : Mat) : Bool :=
  (b.foldl (insertVec p) a).length == a.length + b.length

/-- How many unordered pairs of components meet in more than `0`. -/
def intersecting (p : Nat) (cs : List Mat) : Nat :=
  ((pairsOf cs).filter fun ab => !meetsTrivially p ab.1 ab.2).length

/-- The nonzero vectors of the components, summed: `Σ (p ^ dim - 1)`. When the components meet
pairwise trivially this is the number of nonzero vectors covered, each once. -/
def covered (p : Nat) (cs : List Mat) : Nat := (cs.map fun c => p ^ c.length - 1).foldl (· + ·) 0

/-- Every component is nonzero and they meet pairwise in `0` only. -/
def disjointComponents (p : Nat) (cs : List Mat) : Bool :=
  cs.all (fun c => !c.isEmpty) && intersecting p cs == 0

/-- A partial spread: disjoint components, all of one dimension. -/
def isPartialSpread (p : Nat) (cs : List Mat) : Bool :=
  disjointComponents p cs && cs.all fun c => c.length == (cs.headD []).length

/-- A vector space partition: disjoint components covering every nonzero vector of `F_p^n`. -/
def isPartition (p n : Nat) (cs : List Mat) : Bool :=
  disjointComponents p cs && covered p cs == p ^ n - 1

/-- A spread: a partial spread that covers. -/
def isSpread (p n : Nat) (cs : List Mat) : Bool :=
  isPartialSpread p cs && covered p cs == p ^ n - 1

/-! ## Packings -/

/-- The Gaussian binomial `[n choose h]_p`, the number of `h`-dimensional subspaces of `F_p^n`.
Each factor of the running product is itself a Gaussian binomial, so the divisions are exact. -/
def gaussBinom (p n h : Nat) : Nat :=
  (List.range h).foldl (fun c i => c * (p ^ (n - i) - 1) / (p ^ (i + 1) - 1)) 1

/-- The components of one row of a packing candidate: `m = row.length / (h * n)` blocks of `h * n`
entries, each an `h x n` basis. -/
def blocks (n h : Nat) (row : Vec) : List Mat :=
  (chunk (h * n) (row.length / (h * n)) row).map fun blk => chunk n h blk

/-- A packing of `F_p^n` by `h`-dimensional subspaces: every row a spread by `h`-subspaces, and
the components together every `h`-subspace exactly once (pairwise distinct, and as many as
there are). -/
def isPacking (p n h : Nat) (m : Mat) : Bool :=
  let rows := m.map (blocks n h)
  let cs := rows.map fun row => row.map (echelon p)
  let canon := (rows.flatten).map (rrefOf p)
  cs.all (fun row => isSpread p n row && row.all fun c => c.length == h)
    && canon.length == gaussBinom p n h
    && (pairsOf canon).all fun ab => ab.1 != ab.2

/-! ## Operations and values -/

inductive Op
  | isPartialSpread (n : Nat)
  | isSpread (n : Nat)
  | isVectorSpacePartition (n : Nat)
  | intersectingPairs (n : Nat)
  | isPacking (n h : Nat)

/-- No operation here materialises an object: every value is a Boolean or an integer. -/
inductive Value
  | none
  deriving DecidableEq, Repr

/-- Columns of a member of `f`. -/
def colsOf (ms : List Mat) : Nat := ((ms.headD []).headD []).length

/-- A request the module does not accept: members that are not over a field (permutations,
naturals), no ambient dimension, or a component shape that does not divide the member's rows. -/
def malformed (p unit : Nat) (ms : List Mat) : Bool :=
  p == 0 || unit == 0 || !(colsOf ms % unit == 0)

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let p := f.p
  let ms := f.members
  match op with
  | .isPartialSpread n =>
    if malformed p n ms then .invalid
    else reduceBool red ms (ms.map fun m => isPartialSpread p (components p n m))
  | .isSpread n =>
    if malformed p n ms then .invalid
    else reduceBool red ms (ms.map fun m => isSpread p n (components p n m))
  | .isVectorSpacePartition n =>
    if malformed p n ms then .invalid
    else reduceBool red ms (ms.map fun m => isPartition p n (components p n m))
  | .intersectingPairs n =>
    if malformed p n ms then .invalid
    else reduceInt red ms (ms.map fun m => intersecting p (components p n m))
  | .isPacking n h =>
    if malformed p (n * h) ms || n == 0 then .invalid
    else reduceBool red ms (ms.map (isPacking p n h))

end Spreads_and_partitions
