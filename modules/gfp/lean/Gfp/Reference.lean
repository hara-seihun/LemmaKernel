/-!
# gfp: executable reference

The meaning of every gfp operation, family, and reduction, written as plain structural recursion
so that Lean's kernel can evaluate it. This is the oracle: a kernel answer is correct when
`example : run op family red = answer := by decide` is accepted. The canonical order of every
family and the canonical choice in every output (which nullspace basis, which pivots) are
*defined* here; backends reproduce them bit for bit.

Matrices are lists of rows of naturals, every entry already reduced modulo the prime `p`.
Nothing here imports Mathlib; `Contract.lean` relates these definitions to Mathlib's.
-/

namespace Gfp

abbrev Vec := List Nat
abbrev Mat := List Vec

/-- `a ^ e % p` by repeated squaring; `fuel` bounds the recursion (64 covers any `e < 2^64`). -/
def powMod (p a : Nat) : Nat → Nat → Nat
  | 0, _ => 1 % p
  | fuel + 1, e =>
    if e = 0 then 1 % p
    else
      let h := powMod p (a * a % p) fuel (e / 2)
      if e % 2 = 1 then h * a % p else h

/-- Fermat inverse; only used on nonzero entries below a prime. -/
def inv (p a : Nat) : Nat := powMod p a 64 (p - 2)

/-- `row - c * other` entrywise, modulo `p`. -/
def subMul (p : Nat) (row other : Vec) (c : Nat) : Vec :=
  List.zipWith (fun a b => (a + (p - c) * b) % p) row other

def scale (p : Nat) (row : Vec) (c : Nat) : Vec := row.map (fun a => a * c % p)

def matmul (p : Nat) (a b : Mat) : Mat :=
  let cols := (b.headD []).length
  a.map fun row =>
    (List.range cols).map fun j =>
      (List.zipWith (fun x brow => x * brow.getD j 0) row b).foldl (· + ·) 0 % p

/-! ## Gauss-Jordan elimination

`gaussJordan p rows` returns `(R, T, pivots)` with `T * rows = R`, `R` in reduced row echelon
form, `T` a product of the elementary operations, and `pivots` the pivot columns in increasing
order. Pivoting picks the first row at or below the current one with a nonzero entry. -/

structure GJ where
  rows : Mat
  t : Mat
  r : Nat
  pivots : List Nat

def identity (n : Nat) : Mat := (List.range n).map fun i => (List.range n).map fun j => if i = j then 1 else 0

/-- Index of the first row `i ≥ r` with a nonzero entry in column `c`. -/
def findPivot (rows : Mat) (c r : Nat) : Option Nat :=
  ((List.range rows.length).drop r).find? fun i => (rows.getD i []).getD c 0 ≠ 0

def swapRows (m : Mat) (i j : Nat) : Mat :=
  (List.range m.length).map fun k =>
    if k = i then m.getD j [] else if k = j then m.getD i [] else m.getD k []

def gjColumn (p : Nat) (s : GJ) (c : Nat) : GJ :=
  match findPivot s.rows c s.r with
  | none => s
  | some i =>
    let rows := swapRows s.rows i s.r
    let t := swapRows s.t i s.r
    let f := inv p ((rows.getD s.r []).getD c 0)
    let prow := scale p (rows.getD s.r []) f
    let ptrow := scale p (t.getD s.r []) f
    let factor := fun k => (rows.getD k []).getD c 0
    { rows := (List.range rows.length).map fun k =>
        if k = s.r then prow else subMul p (rows.getD k []) prow (factor k)
      t := (List.range t.length).map fun k =>
        if k = s.r then ptrow else subMul p (t.getD k []) ptrow (factor k)
      r := s.r + 1
      pivots := s.pivots ++ [c] }

def gaussJordan (p : Nat) (m : Mat) : GJ :=
  let cols := (m.headD []).length
  (List.range cols).foldl (gjColumn p) { rows := m, t := identity m.length, r := 0, pivots := [] }

def rref (p : Nat) (m : Mat) : Mat := (gaussJordan p m).rows
def pivots (p : Nat) (m : Mat) : List Nat := (gaussJordan p m).pivots
def rank (p : Nat) (m : Mat) : Nat := (pivots p m).length
def nullity (p : Nat) (m : Mat) : Nat := (m.headD []).length - rank p m

/-- Canonical nullspace basis: one vector per free column `f` in increasing order, with
`x_f = 1`, `x_{pivot_i} = -R[i][f]`, and zeros elsewhere. -/
def nullspace (p : Nat) (m : Mat) : List Vec :=
  let g := gaussJordan p m
  let n := (m.headD []).length
  ((List.range n).filter fun f => f ∉ g.pivots).map fun f =>
    (List.range n).map fun j =>
      if j = f then 1
      else match g.pivots.idxOf? j with
        | some i => (p - (g.rows.getD i []).getD f 0) % p
        | none => 0

def inSpan (p : Nat) (m : Mat) (target : Vec) : Bool := rank p m = rank p (m ++ [target])

/-- `x` with `m xᵀ = rhsᵀ`, free coordinates zero; `none` when inconsistent. -/
def solve (p : Nat) (m : Mat) (rhs : Vec) : Option Vec :=
  let n := (m.headD []).length
  let g := gaussJordan p (List.zipWith (fun row b => row ++ [b]) m rhs)
  if n ∈ g.pivots then none
  else some ((List.range n).map fun j =>
    match g.pivots.idxOf? j with
    | some i => (g.rows.getD i []).getD n 0
    | none => 0)

def inverse (p : Nat) (m : Mat) : Option Mat :=
  let n := m.length
  let g := gaussJordan p (List.zipWith (· ++ ·) m (identity n))
  if g.pivots = List.range n then some (g.rows.map fun row => row.drop n) else none

/-- `(R, T)` with `T * m = R`. -/
def witness (p : Nat) (m : Mat) : Mat × Mat := let g := gaussJordan p m; (g.rows, g.t)

/-! ## Families and their canonical order -/

/-- All `k`-element sublists, in lexicographic order of the chosen positions. -/
def combos : List α → Nat → List (List α)
  | _, 0 => [[]]
  | [], _ + 1 => []
  | x :: xs, k + 1 => (combos xs k).map (x :: ·) ++ combos xs (k + 1)

/-- All lists of length `m` over `0..p-1`, lexicographic (first digit slowest). -/
def tuples (p : Nat) : Nat → List (List Nat)
  | 0 => [[]]
  | m + 1 => (List.range p).flatMap fun d => (tuples p m).map (d :: ·)

/-- Split `rows * cols` entries into `rows` rows. -/
def chunk (cols : Nat) : Nat → List Nat → Mat
  | 0, _ => []
  | r + 1, xs => xs.take cols :: chunk cols r (xs.drop cols)

/-- Free positions of the rref shape with pivot columns `piv` (row-major). -/
def freePositions (piv : List Nat) (n : Nat) : List (Nat × Nat) :=
  (List.range piv.length).flatMap fun i =>
    ((List.range n).filter fun c => piv.getD i 0 < c ∧ c ∉ piv).map fun c => (i, c)

def rrefShape (piv : List Nat) (n : Nat) (free : List (Nat × Nat)) (digits : List Nat) : Mat :=
  (List.range piv.length).map fun i => (List.range n).map fun c =>
    if piv.getD i 0 = c then 1
    else match (List.zip free digits).find? fun ((i', c'), _) => i' = i ∧ c' = c with
      | some (_, d) => d
      | none => 0

/-- Every `h`-dimensional subspace of `F_p^n` as its rref basis: pivot sets in lexicographic
order, then free entries in row-major lexicographic order. -/
def grassmannianMembers (p n h : Nat) : List Mat :=
  (combos (List.range n) h).flatMap fun piv =>
    let free := freePositions piv n
    (tuples p free.length).map (rrefShape piv n free)

inductive Family
  | explicit (p : Nat) (batch : List Mat)
  | subsets (p : Nat) (dictionary : List Vec) (k : Nat)
  | grassmannian (p n h : Nat)
  | allMatrices (p rows cols : Nat)
  | transform (inner : Family) (c : Mat)
  | stack (inner : Family) (rows : Mat)

def Family.p : Family → Nat
  | .explicit p _ | .subsets p _ _ | .grassmannian p _ _ | .allMatrices p _ _ => p
  | .transform f _ | .stack f _ => f.p

def Family.members : Family → List Mat
  | .explicit _ batch => batch
  | .subsets _ d k => combos d k
  | .grassmannian p n h => grassmannianMembers p n h
  | .allMatrices p rows cols => (tuples p (rows * cols)).map (chunk cols rows)
  | .transform f c => f.members.map fun m => matmul f.p m c
  | .stack f rows => f.members.map (· ++ rows)

/-! ## Operations, reductions, results -/

inductive Op
  | rank | nullity | fullRowRank | fullColRank
  | inSpan (target : Vec)
  | rref | nullspace
  | solve (rhs : List Vec) | inverse | rrefWitness

inductive Red
  | all | count | histogram | hits (limit : Nat)

inductive Result
  | integers (xs : List Nat)
  | count (value size : Nat)
  | histogram (size : Nat) (bins : List Nat)
  | hits (size : Nat) (indices : List Nat) (members : List Mat)
  | matrices (ms : List Mat)
  | bases (bs : List (List Vec))
  | solutions (xs : List (Option Vec))
  | inverses (xs : List (Option Mat))
  | witnesses (xs : List (Mat × Mat))
  | invalid
  deriving DecidableEq, Repr

def histogramOf (xs : List Nat) : List Nat :=
  (List.range (xs.foldl max 0 + 1)).map fun v => (xs.filter (· = v)).length

def reduceInt (red : Red) (xs : List Nat) : Result :=
  match red with
  | .all => .integers xs
  | .histogram => .histogram xs.length (histogramOf xs)
  | _ => .invalid

def reduceBool (red : Red) (ms : List Mat) (flags : List Bool) : Result :=
  match red with
  | .all => .integers (flags.map fun b => if b then 1 else 0)
  | .count => .count (flags.filter id).length flags.length
  | .hits limit =>
    let idx := ((List.range flags.length).zip flags).filterMap fun (i, b) => if b then some i else none
    .hits flags.length idx ((idx.take limit).map fun i => ms.getD i [])
  | .histogram => .invalid

def run (op : Op) (f : Family) (red : Red) : Result :=
  let p := f.p
  let ms := f.members
  let rows := (ms.headD []).length
  let cols := ((ms.headD []).headD []).length
  match op, red with
  | .rank, _ => reduceInt red (ms.map (rank p))
  | .nullity, _ => reduceInt red (ms.map (nullity p))
  | .fullRowRank, _ => reduceBool red ms (ms.map fun m => rank p m = rows)
  | .fullColRank, _ => reduceBool red ms (ms.map fun m => rank p m = cols)
  | .inSpan t, _ => reduceBool red ms (ms.map fun m => inSpan p m t)
  | .rref, .all => .matrices (ms.map (rref p))
  | .nullspace, .all => .bases (ms.map (nullspace p))
  | .solve rhs, .all => match f with
    | .explicit _ _ => .solutions (List.zipWith (solve p) ms rhs)
    | _ => .invalid
  | .inverse, .all => match f with
    | .explicit _ _ => .inverses (ms.map (inverse p))
    | _ => .invalid
  | .rrefWitness, .all => match f with
    | .explicit _ _ => .witnesses (ms.map (witness p))
    | _ => .invalid
  | _, _ => .invalid

end Gfp
