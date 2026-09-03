import Lk.Reference

/-!
# hadamard: executable reference

An F_2 entry is a sign bit: zero means +1 and one means -1. The predicates compare signs without
introducing integers. Two signs differ exactly when their product is -1, so a dot product of `n`
signs is zero exactly when half of the positions differ.

`canonicalForm` defines the canonical answer for signed row and column equivalence. It considers
every row order and every choice of the first column, negates rows and columns to make the first
row and column zero, sorts the resulting columns, and takes the row-major lexicographic minimum.
Sorting is the least column order for a fixed row order, so this is the minimum over the full
equivalence class.
-/

namespace Hadamard

open Lk

def bit (m : Mat) (i j : Nat) : Nat := (m.getD i []).getD j 0

def square (m : Mat) : Bool :=
  let n := m.length
  m.all fun row => row.length = n

def differences (m : Mat) (i j : Nat) : Nat :=
  let n := (m.headD []).length
  ((List.range n).filter fun k => bit m i k != bit m j k).length

def isHadamard (m : Mat) : Bool :=
  let n := m.length
  square m && (List.range n).all fun i =>
    (List.range n).all fun j => i = j || differences m i j * 2 = n

def isSkew (m : Mat) : Bool :=
  let n := m.length
  square m && (List.range n).all fun i =>
    (List.range n).all fun j =>
      if i = j then bit m i j = 0 else bit m i j != bit m j i

def onesInRow (m : Mat) (i : Nat) : Nat := ((m.getD i []).filter (· = 1)).length

def onesInCol (m : Mat) (j : Nat) : Nat :=
  ((List.range m.length).filter fun i => bit m i j = 1).length

def isRegular (m : Mat) : Bool :=
  let n := m.length
  let target := onesInRow m 0
  square m &&
    (List.range n).all (fun i => onesInRow m i = target) &&
    (List.range n).all (fun j => onesInCol m j = target)

def conferenceDifferences (m : Mat) (i j : Nat) : Nat :=
  ((List.range m.length).filter fun k => k != i && k != j && bit m i k != bit m j k).length

def isConference (m : Mat) : Bool :=
  let n := m.length
  2 ≤ n && square m &&
    (List.range n).all (fun i => bit m i i = 0) &&
    (List.range n).all fun i => (List.range n).all fun j =>
      i = j || conferenceDifferences m i j * 2 = n - 2

/-! ## Signed-equivalence canonical form -/

def insertEverywhere (x : α) : List α → List (List α)
  | [] => [[x]]
  | y :: ys => (x :: y :: ys) :: (insertEverywhere x ys).map (y :: ·)

def permutations : List α → List (List α)
  | [] => [[]]
  | x :: xs => (permutations xs).flatMap (insertEverywhere x)

def normalisedColumns (m : Mat) (rowOrder : List Nat) (firstCol : Nat) : List Vec :=
  let cols := (m.headD []).length
  let firstRow := rowOrder.headD 0
  (List.range cols).map fun c => rowOrder.map fun r =>
    (bit m r c + bit m r firstCol + bit m firstRow c + bit m firstRow firstCol) % 2

def fromColumns (rows : Nat) (cols : List Vec) : Mat :=
  (List.range rows).map fun i => cols.map fun col => col.getD i 0

def normalised (m : Mat) (rowOrder : List Nat) (firstCol : Nat) : Mat :=
  fromColumns m.length (sortBy lexLe (normalisedColumns m rowOrder firstCol))

def matrixLexLe (a b : Mat) : Bool := lexLe a.flatten b.flatten

def canonicalForm (m : Mat) : Mat :=
  let rows := m.length
  let cols := (m.headD []).length
  if rows = 0 || cols = 0 then m else
    let candidates := (permutations (List.range rows)).flatMap fun rp =>
      (List.range cols).map fun c => normalised m rp c
    candidates.foldl (fun best candidate => if matrixLexLe candidate best then candidate else best)
      (candidates.headD m)

/-! ## Operations and values -/

inductive Op
  | isHadamard | isSkew | isRegular | isConference | canonicalForm

inductive Value
  | matrix (m : Mat)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  if f.p != 2 then .invalid else
  let ms := f.members
  match op with
  | .isHadamard => reduceBool red ms (ms.map isHadamard)
  | .isSkew => reduceBool red ms (ms.map isSkew)
  | .isRegular => reduceBool red ms (ms.map isRegular)
  | .isConference => reduceBool red ms (ms.map isConference)
  | .canonicalForm => reduceValues red (ms.map fun m => .matrix (canonicalForm m))

end Hadamard
