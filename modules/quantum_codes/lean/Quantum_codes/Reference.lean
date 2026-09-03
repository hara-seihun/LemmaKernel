import Gfp.Reference

/-!
# quantum_codes: executable reference

A member matrix over `F_2` with `2n` columns generates a binary symplectic subspace. Its columns
are `(x_0, ..., x_{n-1} | z_0, ..., z_{n-1})`. This is also the additive `GF(4)` description
`x + omega*z`; symplectic weight is the Hamming weight in that description.

The distance is the least symplectic weight in `C^perp_s \\ C`. If the difference is empty, the
reference returns `n + 1`, one above every possible symplectic weight. All definitions use list
folds and structural recursion so `decide +kernel` can evaluate the test oracle.
-/

namespace Quantum_codes

open Lk

/-- The number of columns carried by a family description. Empty explicit batches have no shape
in the Lean interchange term and therefore report zero here. -/
def familyCols : Family → Nat
  | .explicit _ batch => ((batch.headD []).headD []).length
  | .subsets _ dictionary _ => (dictionary.headD []).length
  | .grassmannian _ n _ => n
  | .allMatrices _ _ cols => cols
  | .transform _ c => (c.headD []).length
  | .stack inner rows => if rows.isEmpty then familyCols inner else (rows.headD []).length
  | .groupElements _ gens => ((gens.headD []).headD []).length
  | .groupTables tables => ((tables.headD []).headD []).length
  | .subsetsOf inner _ => familyRows inner * familyCols inner
  | .symmetricMatrices _ n => n
  | .range _ _ => 1
  | .words _ length => length
  | .partitions total _ _ _ _ _ => total
  | .compositions total _ _ => total
  | .standardTableaux shape => shape.headD 0
  | .allGraphs _ | .edgeSubgraphs _ _ | .cayleyGraphs _ => 0
  | .sublattices gram _ => (gram.headD []).length
where
  familyRows : Family → Nat
    | .explicit _ batch => (batch.headD []).length
    | .subsets _ _ k => k
    | .grassmannian _ _ h => h
    | .allMatrices _ rows _ => rows
    | .transform inner _ => familyRows inner
    | .stack inner rows => familyRows inner + rows.length
    | .groupElements _ gens => (gens.headD []).length
    | .groupTables tables => (tables.headD []).length
    | .subsetsOf _ k => k
    | .symmetricMatrices _ n => n
    | .range _ _ => 1
    | .words _ _ => 1
    | .partitions _ _ _ _ _ _ => 1
    | .compositions _ _ _ => 1
    | .standardTableaux shape => shape.length
    | .allGraphs _ | .edgeSubgraphs _ _ | .cayleyGraphs _ => 0
    | .sublattices gram _ => gram.length

/-- Families whose members are field matrices rather than permutations or natural-number rows. -/
def matrixFamily : Family → Bool
  | .explicit _ _ | .subsets _ _ _ | .grassmannian _ _ _ | .allMatrices _ _ _ |
    .symmetricMatrices _ _ => true
  | .transform inner _ | .stack inner _ | .subsetsOf inner _ => matrixFamily inner
  | .groupElements _ _ | .groupTables _ | .range _ _ | .words _ _ |
    .partitions _ _ _ _ _ _ | .compositions _ _ _ | .standardTableaux _ |
    .allGraphs _ | .edgeSubgraphs _ _ | .cayleyGraphs _ | .sublattices _ _ => false

def validFamily (f : Family) : Bool := matrixFamily f && f.p = 2 && familyCols f % 2 = 0

/-- Binary dot product. -/
def dot (a b : Vec) : Nat := (List.zipWith (fun x y => x * y) a b).foldl (· + ·) 0 % 2

/-- The standard binary symplectic form on `(x|z)`. -/
def symplecticInner (a b : Vec) : Nat :=
  let n := a.length / 2
  (dot (a.take n) (b.drop n) + dot (a.drop n) (b.take n)) % 2

def isSelfOrthogonal (m : Mat) : Bool :=
  m.all fun a => m.all fun b => symplecticInner a b = 0

/-- Addition in a binary vector space. -/
def add (a b : Vec) : Vec := List.zipWith (fun x y => (x + y) % 2) a b

/-- Every binary linear combination of the rows. Duplicates caused by dependent rows are harmless
for membership, and avoiding a second elimination keeps kernel evaluation cheap on small cases. -/
def spanWords (cols : Nat) (m : Mat) : List Vec :=
  (tuples 2 m.length).map fun coefficients =>
    (List.zip coefficients m).foldl
      (fun v pair => if pair.1 = 0 then v else add v pair.2)
      (List.replicate cols 0)

/-- Digits `0,1,2,3` encode `(0,0),(1,0),(0,1),(1,1)` respectively. -/
def gf4WordToSymplectic (word : List Nat) : Vec :=
  word.map (· % 2) ++ word.map (· / 2)

def symplecticWeight (v : Vec) : Nat :=
  let n := v.length / 2
  (List.zip (v.take n) (v.drop n)).countP fun pair => pair.1 ≠ 0 || pair.2 ≠ 0

/-- Least symplectic weight in `C^perp_s \\ C`, with `n+1` as the empty-set sentinel. -/
def distance (m : Mat) : Nat :=
  let cols := (m.headD []).length
  let n := cols / 2
  let code := spanWords cols m
  let weights := (tuples 4 n).filterMap fun word =>
    let v := gf4WordToSymplectic word
    if m.all (fun row => symplecticInner v row = 0) && !(v ∈ code)
      then some (symplecticWeight v)
      else none
  weights.foldl min (n + 1)

/-- A row space is CSS exactly when its X-only and Z-only parts span it. The rank identity is the
basis-independent dimension criterion for that direct sum. -/
def isCss (m : Mat) : Bool :=
  let cols := (m.headD []).length
  let n := cols / 2
  Gfp.rank 2 (m.map (·.take n)) + Gfp.rank 2 (m.map (·.drop n)) = Gfp.rank 2 m

inductive Op
  | isSelfOrthogonal
  | distance
  | isCss

inductive Value
  | unused
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  if !validFamily f then .invalid
  else
    let ms := f.members
    match op with
    | .isSelfOrthogonal => reduceBool red ms (ms.map isSelfOrthogonal)
    | .distance => reduceInt red ms (ms.map distance)
    | .isCss => reduceBool red ms (ms.map isCss)

end Quantum_codes
