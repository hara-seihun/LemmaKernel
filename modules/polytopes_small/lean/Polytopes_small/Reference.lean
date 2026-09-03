import Lk.Reference

/-!
# polytopes_small: executable reference

A member is a nonempty list of lattice points, one point per matrix row. Supporting hyperplanes
are found with exact integer determinants after projection to the affine span. Intersections of
facets give every nonempty face. Ehrhart data is the h-star vector, which is integral and gives a
canonical encoding of the polynomial.
-/

namespace Polytopes_small

open Lk

abbrev IVec := List Int
abbrev IMat := List IVec


def eraseAt (xs : List α) (i : Nat) : List α := xs.take i ++ xs.drop (i + 1)

def det : IMat → Int
  | [] => 1
  | row :: rows =>
    (List.range row.length).foldl (fun acc j =>
      let term := row.getD j 0 * det (rows.map fun r => eraseAt r j)
      if j % 2 = 0 then acc + term else acc - term) 0
termination_by m => m.length

def sub (a b : IVec) : IVec := List.zipWith (· - ·) a b

def dot (a b : IVec) : Int := (List.zipWith (· * ·) a b).foldl (· + ·) 0

def minor (m : IMat) (rs cs : List Nat) : IMat :=
  rs.map fun i => cs.map fun j => (m.getD i []).getD j 0

def hasMinor (m : IMat) (k : Nat) : Bool :=
  let rows := m.length
  let cols := (m.headD []).length
  (combos (List.range rows) k).any fun rs =>
    (combos (List.range cols) k).any fun cs => det (minor m rs cs) ≠ 0

def matrixRank (m : IMat) : Nat :=
  let bound := min m.length (m.headD []).length
  (List.range (bound + 1)).foldl (fun r k => if hasMinor m k then k else r) 0

def affineRank : List IVec → Nat
  | [] => 0
  | x :: xs => matrixRank (xs.map fun y => sub y x)

def unique [BEq α] (xs : List α) : List α := xs.eraseDups

def project (cols : List Nat) (x : IVec) : IVec := cols.map fun j => x.getD j 0

def projection (pts : List IVec) (r d : Nat) : List Nat :=
  ((combos (List.range d) r).find? fun cs => affineRank (pts.map (project cs)) = r).getD (List.range r)

structure Facet where
  indices : List Nat
  normal : IVec
  constant : Int
  deriving DecidableEq, Repr

def orientedFacet (pts : List IVec) (r : Nat) (choice : List Nat) : Option Facet :=
  let chosen := choice.map fun i => pts.getD i []
  if affineRank chosen ≠ r - 1 then none else
  let base := chosen.headD []
  let differences := chosen.drop 1 |>.map fun x => sub x base
  let normal := (List.range r).map fun j =>
    let cofactor := det (differences.map fun row => eraseAt row j)
    if j % 2 = 0 then cofactor else -cofactor
  let c := dot normal base
  let values := pts.map fun x => dot normal x - c
  let zeroes := ((List.range pts.length).zip values).filterMap fun (i, z) => if z = 0 then some i else none
  if values.all (· ≤ 0) && values.any (· ≠ 0) then some ⟨zeroes, normal, c⟩
  else if values.all (· ≥ 0) && values.any (· ≠ 0) then some ⟨zeroes, normal.map (-·), -c⟩
  else none

def facets (pts : List IVec) (r : Nat) : List Facet :=
  (combos (List.range pts.length) r).foldl (fun out choice =>
    match orientedFacet pts r choice with
    | none => out
    | some f => if out.any fun g => g.indices = f.indices then out else out ++ [f]) []

def intersect (a b : List Nat) : List Nat := a.filter (· ∈ b)

def closeWithFacet (faces : List (List Nat)) (facet : Facet) : List (List Nat) :=
  faces.foldl (fun out face =>
    let x := intersect face facet.indices
    if x.isEmpty || x ∈ out then out else out ++ [x]) faces

def faceSets (pts : List IVec) (fs : List Facet) : List (List Nat) :=
  fs.foldl closeWithFacet [List.range pts.length]

def faceDim (pts : List IVec) (face : List Nat) : Nat := affineRank (face.map fun i => pts.getD i [])

structure Analysis where
  points : List IVec
  ambientDim : Nat
  affineDim : Nat
  projection : List Nat
  projected : List IVec
  facets : List Facet
  faces : List (List Nat)
  deriving Repr

def analyse (m : Mat) : Analysis :=
  let d := (m.headD []).length
  let pts := unique (m.map fun row => row.map Int.ofNat)
  let r := affineRank pts
  let cols := projection pts r d
  let projected := pts.map (project cols)
  let fs := facets projected r
  ⟨pts, d, r, cols, projected, fs, faceSets projected fs⟩

def fVector (m : Mat) : List Nat :=
  let a := analyse m
  (List.range (a.ambientDim + 1)).map fun k => (a.faces.filter fun face => faceDim a.projected face = k).length

def vertexCount (m : Mat) : Nat := (fVector m).headD 0

def vertexIndices (a : Analysis) : List Nat :=
  unique ((a.faces.filter fun face => faceDim a.projected face = 0).flatten)

def isSimplicial (m : Mat) : Bool :=
  let a := analyse m
  if a.affineDim ≤ 1 then true else
  let vertices := vertexIndices a
  a.facets.all fun facet => (facet.indices.filter (· ∈ vertices)).length = a.affineDim


def intervals : List (Nat × Nat) → List Vec
  | [] => [[]]
  | (lo, hi) :: rest =>
    (List.range (hi - lo + 1)).flatMap fun x => (intervals rest).map fun tail => (lo + x) :: tail


def inDilate (a : Analysis) (t : Nat) (x : Vec) : Bool :=
  let xi := x.map Int.ofNat
  let scaled := a.points.map fun p => p.map fun z => Int.ofNat t * z
  affineRank (scaled ++ [xi]) = a.affineDim &&
    a.facets.all fun facet => dot facet.normal (project a.projection xi) ≤ Int.ofNat t * facet.constant


def latticeCount (a : Analysis) (t : Nat) : Nat :=
  if t = 0 then 1 else
  let boxes := (List.range a.ambientDim).map fun j =>
    let coordinates := a.points.map fun p => (p.getD j 0).toNat
    (t * coordinates.foldl min (coordinates.headD 0), t * coordinates.foldl max 0)
  (intervals boxes).filter (inDilate a t) |>.length


def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)


def hStar (m : Mat) : List Nat :=
  let a := analyse m
  let counts := (List.range (a.affineDim + 1)).map (latticeCount a)
  let hs := (List.range (a.affineDim + 1)).map fun j =>
    let z := (List.range (j + 1)).foldl (fun total i =>
      let term := Int.ofNat (choose (a.affineDim + 1) (j - i) * counts.getD i 0)
      if (j - i) % 2 = 0 then total + term else total - term) 0
    z.toNat
  hs ++ List.replicate (a.ambientDim + 1 - hs.length) 0

inductive Op
  | vertexCount | ehrhartPolynomial | fVector | isSimplicial

inductive Value
  | vector (xs : List Nat)
  deriving DecidableEq, Repr


def validFamily : Family → Bool
  | .subsets _ dictionary k => k ≠ 0 && !dictionary.isEmpty && (dictionary.headD []).length ≤ 6
  | _ => false


def run (op : Op) (f : Family) (red : Red) : Result Value :=
  if !validFamily f then .invalid else
  let ms := f.members
  match op with
  | .vertexCount => reduceInt red ms (ms.map vertexCount)
  | .isSimplicial => reduceBool red ms (ms.map isSimplicial)
  | .fVector => reduceValues red (ms.map fun m => .vector (fVector m))
  | .ehrhartPolynomial => reduceValues red (ms.map fun m => .vector (hStar m))

end Polytopes_small
