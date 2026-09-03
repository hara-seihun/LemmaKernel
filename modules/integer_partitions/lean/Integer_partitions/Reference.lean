import Lk.Reference

/-!
# Constrained integer partitions and compositions: executable reference

The runtime families enumerate each partition or composition as one row of positive parts followed
by zeros. This module computes statistics from that row. Rank and crank may be negative, while the
runtime's integer reductions use natural-number bins, so both statistics are encoded by adding the
partition total. Thus histogram bin `i` means the mathematical statistic `i - total`.
-/

namespace Integer_partitions

open Lk

def positiveParts (m : Mat) : Vec := (m.headD []).filter (· ≠ 0)

def numberOfParts (m : Mat) : Nat := (positiveParts m).length

def largestPart (m : Mat) : Nat := (positiveParts m).headD 0

/-- Dyson rank shifted by the partition total. -/
def rankOffset (total : Nat) (m : Mat) : Nat := total + largestPart m - numberOfParts m

/-- Andrews-Garvan crank shifted by the partition total. -/
def crankOffset (total : Nat) (m : Mat) : Nat :=
  let parts := positiveParts m
  let ones := (parts.filter (· = 1)).length
  if ones = 0 then total + parts.headD 0
  else total + (parts.filter (· > ones)).length - ones

inductive Op
  | numberOfParts
  | largestPart
  | rank
  | crank

def run (op : Op) (f : Family) (red : Red) : Result Empty :=
  let ms := f.members
  match op, f with
  | .numberOfParts, .partitions _ _ _ _ _ _ => reduceInt red ms (ms.map numberOfParts)
  | .numberOfParts, .compositions _ _ _ => reduceInt red ms (ms.map numberOfParts)
  | .largestPart, .partitions _ _ _ _ _ _ => reduceInt red ms (ms.map largestPart)
  | .largestPart, .compositions _ _ _ => reduceInt red ms (ms.map largestPart)
  | .rank, .partitions total _ _ _ _ _ => reduceInt red ms (ms.map (rankOffset total))
  | .crank, .partitions total _ _ _ _ _ => reduceInt red ms (ms.map (crankOffset total))
  | _, _ => .invalid

end Integer_partitions
