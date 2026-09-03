import Lk.Reference

/-!
# young: executable reference

Partitions are positive row lengths. Runtime partition members pad them with trailing zeros, so
`shapeOf` removes that padding before every operation. Standard tableaux use English notation.
Rows increase weakly from left to right, columns strictly from top to bottom.

RSK uses ordinary row insertion: the first row entry strictly greater than the new letter is
replaced and bumped. Murnaghan-Nakayama enumerates every contained partition and keeps exactly
those skew diagrams that are connected and contain no 2 by 2 block.
-/

namespace Young

open Lk

abbrev Cell := Nat × Nat
abbrev Marking := List (Cell × Nat)

def shapeOf (m : Mat) : Vec := ((m.headD []).reverse.dropWhile (· = 0)).reverse

def weakDecreasing : Vec → Bool
  | [] | [_] => true
  | a :: b :: xs => b ≤ a && weakDecreasing (b :: xs)

def positive (xs : Vec) : Bool := xs.all (· > 0)
def sizeOf (shape : Vec) : Nat := shape.foldl (· + ·) 0

def validComposition (weight : Vec) (n : Nat) : Bool :=
  !weight.isEmpty && sizeOf weight = n

def validPartition (shape : Vec) (n : Nat) : Bool :=
  positive shape && weakDecreasing shape && sizeOf shape = n

def cells (shape : Vec) : List Cell :=
  (List.range shape.length).flatMap fun i =>
    (List.range (shape.getD i 0)).map fun j => (i, j)

def marked (f : Marking) (cell : Cell) : Option Nat :=
  match f.find? fun x => x.1 = cell with
  | some x => some x.2
  | none => none

/-! ## Kostka numbers -/

def kostkaAux : Nat → List Cell → Vec → Marking → Nat
  | 0, pending, weight, _ => if pending.isEmpty && weight.all (· = 0) then 1 else 0
  | _fuel + 1, [], weight, _ => if weight.all (· = 0) then 1 else 0
  | fuel + 1, (i, j) :: pending, weight, filling =>
    let left := if j = 0 then 0 else (marked filling (i, j - 1)).getD 0
    let above := if i = 0 then 0 else (marked filling (i - 1, j)).map (· + 1) |>.getD 0
    let lower := max left above
    (List.range weight.length).foldl (fun total x =>
      if lower ≤ x && weight.getD x 0 > 0 then
        total + kostkaAux fuel pending (weight.set x (weight.getD x 0 - 1)) (((i, j), x) :: filling)
      else total) 0

def kostkaNumber (shape weight : Vec) : Nat :=
  let cs := cells shape
  kostkaAux cs.length cs weight []

/-! ## Hook lengths -/

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def hookLength (shape : Vec) (i j : Nat) : Nat :=
  shape.getD i 0 - j + (((List.range shape.length).drop (i + 1)).filter (shape.getD · 0 > j)).length

def hookProduct (shape : Vec) : Nat :=
  (cells shape).foldl (fun product cell => product * hookLength shape cell.1 cell.2) 1

def hookLengthCount (shape : Vec) : Nat := factorial (sizeOf shape) / hookProduct shape

/-! ## Murnaghan-Nakayama -/

/-- Every weakly decreasing row vector bounded componentwise by `outer`. -/
def subpartitions : Vec → Nat → List Vec
  | [], _ => [[]]
  | cap :: rest, bound =>
    let largest := min cap bound
    (List.range (largest + 1)).reverse.flatMap fun row =>
      (subpartitions rest row).map (row :: ·)

def skewCells (outer inner : Vec) : List Cell :=
  (cells outer).filter fun cell => inner.getD cell.1 0 ≤ cell.2

def adjacent (a b : Cell) : Bool :=
  (a.1 = b.1 && (a.2 + 1 = b.2 || b.2 + 1 = a.2)) ||
  (a.2 = b.2 && (a.1 + 1 = b.1 || b.1 + 1 = a.1))

def connected (xs : List Cell) : Bool :=
  match xs with
  | [] => false
  | x :: _ => (saturate (fun a => xs.filter (adjacent a)) xs.length [x]).length = xs.length

def hasSquare (xs : List Cell) : Bool :=
  xs.any fun cell =>
    (cell.1, cell.2 + 1) ∈ xs && (cell.1 + 1, cell.2) ∈ xs && (cell.1 + 1, cell.2 + 1) ∈ xs

def stripHeight (xs : List Cell) : Nat := (xs.map Prod.fst).eraseDups.length

def isBorderStrip (outer inner : Vec) (k : Nat) : Bool :=
  let skew := skewCells outer inner
  skew.length = k && connected skew && !hasSquare skew

def mnAux : Nat → Vec → Vec → Int
  | 0, shape, cycles => if cycles.isEmpty && sizeOf shape = 0 then 1 else 0
  | _fuel + 1, shape, [] => if sizeOf shape = 0 then 1 else 0
  | fuel + 1, shape, k :: cycles =>
    (subpartitions shape (shape.headD 0)).foldl (fun total inner =>
      if isBorderStrip shape inner k then
        let height := stripHeight (skewCells shape inner)
        let sign : Int := if height % 2 = 1 then 1 else -1
        total + sign * mnAux fuel inner cycles
      else total) 0

def murnaghanNakayama (shape cycleType : Vec) : Int :=
  mnAux cycleType.length shape cycleType

/-! ## RSK row insertion -/

def insertRow (x : Nat) : Vec → Vec × Option Nat
  | [] => ([x], none)
  | y :: ys =>
    if x < y then (x :: ys, some y)
    else
      let next := insertRow x ys
      (y :: next.1, next.2)

def insertTableau : Nat → Nat → Mat → Nat → Mat × Nat
  | 0, _, tableau, row => (tableau, row)
  | fuel + 1, x, tableau, row =>
    let inserted := insertRow x (tableau.getD row [])
    let nextTableau := if row < tableau.length then tableau.set row inserted.1 else tableau ++ [inserted.1]
    match inserted.2 with
    | none => (nextTableau, row)
    | some bumped => insertTableau fuel bumped nextTableau (row + 1)

structure RskState where
  insertion : Mat
  recording : Mat
  nextLabel : Nat


def rskStep (fuel : Nat) (state : RskState) (x : Nat) : RskState :=
  let inserted := insertTableau fuel x state.insertion 0
  let row := inserted.2
  let recording :=
    if row < state.recording.length then
      state.recording.set row (state.recording.getD row [] ++ [state.nextLabel])
    else state.recording ++ [[state.nextLabel]]
  { insertion := inserted.1, recording := recording, nextLabel := state.nextLabel + 1 }

def padShape (n : Nat) (tableau : Mat) : Vec :=
  (tableau.map List.length ++ List.replicate n 0).take n

def padTableau (n : Nat) (tableau : Mat) : Mat :=
  (List.range n).map fun i => (List.range n).map fun j => (tableau.getD i []).getD j 0

def rskWord (word : Vec) : Vec × Mat × Mat :=
  let state := word.foldl (rskStep (word.length + 1)) { insertion := [], recording := [], nextLabel := 1 }
  (padShape word.length state.insertion, padTableau word.length state.insertion,
    padTableau word.length state.recording)

/-! ## Operations and values -/

inductive Op
  | kostka (weight : Vec)
  | rsk
  | murnaghanNakayama (cycleType : Vec)
  | hookLengthCount

inductive Value
  | character (z : Int)
  | rsk (pair : Vec × Mat × Mat)
  deriving DecidableEq, Repr

def run (op : Op) (family : Family) (red : Red) : Result Value :=
  let members := family.members
  match op, family with
  | .kostka weight, .partitions n _ _ _ _ _ =>
    if validComposition weight n then reduceInt red members (members.map fun m => kostkaNumber (shapeOf m) weight)
    else .invalid
  | .rsk, .words _ _ => reduceValues red (members.map fun m => .rsk (rskWord (m.headD [])))
  | .murnaghanNakayama cycleType, .partitions n _ _ _ _ _ =>
    if validPartition cycleType n then
      reduceValues red (members.map fun m => .character (murnaghanNakayama (shapeOf m) cycleType))
    else .invalid
  | .hookLengthCount, .partitions _ _ _ _ _ _ =>
    reduceInt red members (members.map fun m => hookLengthCount (shapeOf m))
  | .hookLengthCount, .standardTableaux shape =>
    reduceInt red members (members.map fun _ => hookLengthCount shape)
  | _, _ => .invalid

end Young
