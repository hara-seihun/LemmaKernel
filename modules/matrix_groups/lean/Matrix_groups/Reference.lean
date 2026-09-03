import Gfp.Reference

/-!
# matrix_groups: executable reference

Each member of a `subsets` family is a list of rows. Every row is read row-major as one square
generator. Vectors act on the right, so a subspace image is `W A` and an invariant bilinear form
satisfies `A B Aᵀ = B`.

The reference closes small groups and vector orbits by structural recursion. Irreducibility checks
the orbit span of every projective point. Absolute irreducibility adds the scalar-commutant test.
Imprimitivity searches the Grassmannians for a subspace whose orbit is a direct decomposition.
These definitions favor kernel evaluation on small oracle cases over asymptotic performance.
-/

namespace Matrix_groups

open Lk

/-- The positive square root when `m` is a nonzero perfect square. -/
def squareRoot? (m : Nat) : Option Nat :=
  (List.range (m + 1)).find? fun n => n > 0 && n * n = m

/-- Read one flattened row-major square matrix. -/
def unflatten (n : Nat) (v : Vec) : Mat := chunk n n v

def decodeGenerators (n : Nat) (m : Mat) : List Mat := m.map (unflatten n)

def validFlatGenerator (p n : Nat) (v : Vec) : Bool :=
  v.length = n * n && Gfp.rank p (unflatten n v) = n

/-- The generated matrix group, in breadth-first closure order. -/
def groupElements (p : Nat) (gens : List Mat) : List Mat :=
  saturate (fun a => gens.map (matmul p a)) fuel [identity (gens.headD []).length]


def groupOrder (p : Nat) (gens : List Mat) : Nat := (groupElements p gens).length

/-- The orbit span of a row vector under the generators. -/
def orbitSpan (p : Nat) (gens : List Mat) (v : Vec) : Mat :=
  saturate (fun w => gens.map fun a => (matmul p [w] a).headD []) fuel [v]

/-- One normalized nonzero row from each one-dimensional subspace. -/
def projectiveVectors (p n : Nat) : List Vec :=
  (grassmannianMembers p n 1).map (·.headD [])

def isIrreducible (p n : Nat) (gens : List Mat) : Bool :=
  (projectiveVectors p n).all fun v => Gfp.rank p (orbitSpan p gens v) = n

/-- Linear equations for matrices `X` satisfying `X A = A X` for every generator. Variables
are the row-major entries of `X`. -/
def centralizerRows (p n : Nat) (gens : List Mat) : Mat :=
  gens.flatMap fun a =>
    (List.range n).flatMap fun i =>
      (List.range n).map fun j =>
        (List.range (n * n)).map fun x =>
          let r := x / n
          let c := x % n
          let left := if r = i then (a.getD c []).getD j 0 else 0
          let right := if c = j then (a.getD i []).getD r 0 else 0
          (left + p - right) % p

def centralizerDimension (p n : Nat) (gens : List Mat) : Nat :=
  n * n - Gfp.rank p (centralizerRows p n gens)

def isAbsolutelyIrreducible (p n : Nat) (gens : List Mat) : Bool :=
  isIrreducible p n gens && centralizerDimension p n gens = 1

/-- Linear equations for forms `B` satisfying `A B Aᵀ = B`. -/
def invariantFormRows (p n : Nat) (gens : List Mat) : Mat :=
  gens.flatMap fun a =>
    (List.range n).flatMap fun i =>
      (List.range n).map fun j =>
        (List.range (n * n)).map fun x =>
          let r := x / n
          let c := x % n
          let moved := ((a.getD i []).getD r 0 * (a.getD j []).getD c 0) % p
          let original := if i = r && j = c then 1 else 0
          (moved + p - original) % p

def linearCombination (p cols : Nat) (basis : List Vec) (coeffs : List Nat) : Vec :=
  (List.range cols).map fun j =>
    (List.zipWith (fun c v => c * v.getD j 0) coeffs basis).foldl (· + ·) 0 % p

/-- Exhaustive finite-field search for an invertible member of the invariant-form space. -/
def preservesForm (p n : Nat) (gens : List Mat) : Bool :=
  let basis := Gfp.nullspace p (invariantFormRows p n gens)
  (tuples p basis.length).any fun coeffs =>
    coeffs.any (· ≠ 0) &&
      Gfp.rank p (unflatten n (linearCombination p (n * n) basis coeffs)) = n

/-- Orbit of a subspace under right multiplication, with every image returned in rref. -/
def subspaceOrbit (p : Nat) (gens : List Mat) (w : Mat) : List Mat :=
  saturate (fun u => gens.map fun a => Gfp.rref p (matmul p u a)) fuel [w]

def isBlock (p n d : Nat) (gens : List Mat) (w : Mat) : Bool :=
  let os := subspaceOrbit p gens w
  os.length = n / d && Gfp.rank p os.flatten = n

/-- Linear imprimitivity uses the standard convention that reducible groups are not called
imprimitive. -/
def isImprimitive (p n : Nat) (gens : List Mat) : Bool :=
  isIrreducible p n gens &&
    ((List.range n).filter fun d => d > 0 && d < n && n % d = 0).any fun d =>
      (grassmannianMembers p n d).any (isBlock p n d gens)

inductive Op
  | order
  | isIrreducible
  | isAbsolutelyIrreducible
  | preservesForm
  | isImprimitive

inductive Value
  | unused
  deriving DecidableEq, Repr


def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match f with
  | .subsets p dictionary _ =>
    match squareRoot? (dictionary.headD []).length with
    | none => .invalid
    | some n =>
      if !dictionary.all (validFlatGenerator p n) then .invalid else
      let ms := f.members
      let groups := ms.map (decodeGenerators n)
      match op with
      | .order => reduceInt red ms (groups.map (groupOrder p))
      | .isIrreducible => reduceBool red ms (groups.map (isIrreducible p n))
      | .isAbsolutelyIrreducible => reduceBool red ms (groups.map (isAbsolutelyIrreducible p n))
      | .preservesForm => reduceBool red ms (groups.map (preservesForm p n))
      | .isImprimitive => reduceBool red ms (groups.map (isImprimitive p n))
  | _ => .invalid

end Matrix_groups
