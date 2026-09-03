import Lk.Reference

/-!
# Latin squares: executable reference

Squares are lists of rows with symbols `0..n-1`. The `latinSquares` family lists every square in
row-major lexicographic order by extending a prefix with permutation rows and rejecting a row as
soon as it repeats a symbol in a column.

The isotopy canonical form is the row-major lexicographic minimum under independent row, column,
and symbol permutations. For fixed row and column permutations, renaming symbols by order of first
appearance gives the least symbol permutation.
-/

namespace Latin_squares

open Lk

def allDistinct [BEq α] : List α → Bool
  | [] => true
  | x :: xs => !xs.contains x && allDistinct xs

def rowValid (n : Nat) (row : Vec) : Bool :=
  row.length == n && row.all (fun x => x < n) && allDistinct row

def column (m : Mat) (j : Nat) : Vec := m.map fun row => row.getD j 0

def wellShaped (m : Mat) : Bool :=
  0 < m.length && m.all (fun row => row.length == m.length)

/-- Every row and column is a permutation of `0..n-1`. -/
def isLatin (m : Mat) : Bool :=
  let n := m.length
  0 < n && m.all (rowValid n) && (List.range n).all (fun j => rowValid n (column m j))

/-- The superimposed ordered pairs are all distinct. -/
def orthogonal (a b : Mat) : Bool :=
  a.length == b.length && a.flatten.length == b.flatten.length &&
    allDistinct (List.zip a.flatten b.flatten)

/-- Whether an orthogonal Latin square exists. Candidates use the family order. -/
def hasOrthogonalMate (m : Mat) : Bool :=
  isLatin m && (latinSquaresMembers m.length).any fun mate => orthogonal m mate

/-- One cell in each row and column, with every symbol represented. -/
def transversalCount (m : Mat) : Nat :=
  ((permutationRows m.length).filter fun columns =>
    allDistinct ((List.range m.length).map fun i =>
      (m.getD i []).getD (columns.getD i 0) 0)).length

def identityAt (m : Mat) (e : Nat) : Bool :=
  (List.range m.length).all fun x =>
    (m.getD e []).getD x 0 == x && (m.getD x []).getD e 0 == x

def associative (m : Mat) : Bool :=
  (List.range m.length).all fun x =>
    (List.range m.length).all fun y =>
      (List.range m.length).all fun z =>
        (m.getD ((m.getD x []).getD y 0) []).getD z 0 ==
          (m.getD x []).getD ((m.getD y []).getD z 0) 0

/-- The labels themselves are the elements: no row, column, or symbol relabelling is implicit. -/
def isGroupTable (m : Mat) : Bool :=
  isLatin m && associative m && (List.range m.length).any (identityAt m)

def symbolOrder (xs : List Nat) : List Nat :=
  xs.foldl (fun seen x => if seen.contains x then seen else seen ++ [x]) []

def normaliseSymbols (m : Mat) : Mat :=
  let symbols := symbolOrder m.flatten
  m.map fun row => row.map symbols.idxOf

def permuteRowsColumns (m : Mat) (rows columns : Vec) : Mat :=
  rows.map fun i => columns.map fun j => (m.getD i []).getD j 0

def isotopes (m : Mat) : List Mat :=
  let orders := permutationRows m.length
  orders.flatMap fun rows => orders.map fun columns =>
    normaliseSymbols (permuteRowsColumns m rows columns)

/-- Row-major lexicographic minimum over the isotopy class. -/
def isotopyCanonicalForm (m : Mat) : Mat :=
  match isotopes m with
  | [] => []
  | first :: rest => rest.foldl (fun best candidate =>
      if lexLe candidate.flatten best.flatten then candidate else best) first

inductive Op
  | isLatin
  | hasOrthogonalMate
  | transversalCount
  | isGroupTable
  | isotopyCanonicalForm

inductive Value
  | naturals (m : Mat)
  deriving DecidableEq, Repr

def supportedFamily : Family → Bool
  | .explicit _ batch => batch.all wellShaped
  | .latinSquares n => 0 < n
  | _ => false

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  if !supportedFamily f then .invalid else
  let ms := f.members
  match op with
  | .isLatin => reduceBool red ms (ms.map isLatin)
  | .hasOrthogonalMate => reduceBool red ms (ms.map hasOrthogonalMate)
  | .transversalCount => reduceInt red ms (ms.map transversalCount)
  | .isGroupTable => reduceBool red ms (ms.map isGroupTable)
  | .isotopyCanonicalForm =>
      reduceValues red (ms.map fun m => .naturals (isotopyCanonicalForm m))

end Latin_squares
