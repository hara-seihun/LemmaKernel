import Lk.Reference

/-!
# sum_free_and_additive: executable reference

A member is a finite set of natural numbers, carried as a one-column `lk.naturals` matrix: one
element per row. The ambient group is the integers when `modulus = 0` and `Z/modulus` otherwise,
and every operation takes that modulus as an argument.

Two conventions decide the answers everywhere, and the backend and the naive implementation
follow them: sums and differences run over *all* ordered pairs, `x = y` included, and an
arithmetic progression only needs a nonzero common difference, not distinct terms.

Differences are held as codes rather than as integers, because the elements are naturals: over
`Z/n` the code of `x - y` is `(x + n - y) % n`, and over the integers it is `x + shift - y` with
`shift` the largest element, which is injective in `x - y` and never underflows. The codes are
internal; only the number of distinct ones and how often each occurs leave this file.

Everything is structural list recursion so `decide +kernel` can evaluate the oracle cases.
-/

namespace Sum_free_and_additive

open Lk

/-! ## Members as sets -/

/-- The elements of a member: one per row, each row a single entry. -/
def elems (m : Mat) : List Nat := m.map (·.headD 0)

/-- A member (or a dictionary) is a set of elements of the ambient group: one column per row,
no repeats, and every element below a nonzero modulus. -/
def validRows (modulus : Nat) (rows : Mat) : Bool :=
  rows.all (fun r => r.length = 1) &&
    (elems rows).eraseDups.length == (elems rows).length &&
    (modulus == 0 || (elems rows).all (· < modulus))

/-- The families whose members are sets: the dictionary of a `subsets` or `subsets_of` family
decides every member at once, an `explicit` batch member by member. Members must be naturals. -/
def validFamily (modulus : Nat) (f : Family) : Bool :=
  f.naturals &&
    match f with
    | .explicit _ _ => f.members.all (validRows modulus)
    | .subsets _ _ _ | .subsetsOf _ _ => validRows modulus (f.dictionary.getD [])
    | _ => false

/-! ## The ambient group -/

/-- `x + y`, in the integers when `modulus = 0` and in `Z/modulus` otherwise. -/
def add (modulus x y : Nat) : Nat := if modulus = 0 then x + y else (x + y) % modulus

/-- A code for `x - y`, injective in the difference when every element is at most `shift`. -/
def diff (modulus shift x y : Nat) : Nat :=
  if modulus = 0 then x + shift - y else (x + modulus - y) % modulus

/-- An upper bound for the elements, used as the shift of the difference codes. -/
def shiftOf (xs : List Nat) : Nat := xs.foldl max 0

/-- `a + i * d`, the `i`-th term of the progression from `a` with difference `d`. -/
def apTerm (modulus a d i : Nat) : Nat := if modulus = 0 then a + d * i else (a + d * i) % modulus

/-! ## The operations -/

/-- Every sum of an ordered pair, `x = y` included; `S + S` with multiplicity. -/
def sums (modulus : Nat) (xs : List Nat) : List Nat :=
  xs.flatMap fun x => xs.map fun y => add modulus x y

/-- Every difference of an ordered pair as a code, `x = y` included. -/
def diffs (modulus : Nat) (xs : List Nat) : List Nat :=
  xs.flatMap fun x => xs.map fun y => diff modulus (shiftOf xs) x y

/-- The codes of the differences of ordered pairs of distinct elements. -/
def nonzeroDiffs (modulus : Nat) (xs : List Nat) : List Nat :=
  xs.flatMap fun x => (xs.filter (· != x)).map fun y => diff modulus (shiftOf xs) x y

def sumsetSize (modulus : Nat) (xs : List Nat) : Nat := (sums modulus xs).eraseDups.length

def differenceSetSize (modulus : Nat) (xs : List Nat) : Nat := (diffs modulus xs).eraseDups.length

/-- Ordered pairs `(x, y)` of the set whose sum is again in the set. -/
def schurTripleCount (modulus : Nat) (xs : List Nat) : Nat :=
  ((sums modulus xs).filter (xs.contains ·)).length

def isSumFree (modulus : Nat) (xs : List Nat) : Bool := schurTripleCount modulus xs == 0

/-- How often the most popular nonzero difference occurs. -/
def maxDifferenceMultiplicity (modulus : Nat) (xs : List Nat) : Nat :=
  let ds := nonzeroDiffs modulus xs
  ds.eraseDups.foldl (fun best d => max best (ds.filter (· == d)).length) 0

def isSidon (modulus : Nat) (xs : List Nat) : Bool := maxDifferenceMultiplicity modulus xs ≤ 1

/-- The common differences a progression of length at least two can have: `1..n-1` in `Z/n`, and
`1..max` over the integers, where a longer step leaves the set. -/
def apDiffs (modulus : Nat) (xs : List Nat) : List Nat :=
  if modulus = 0 then (List.range (shiftOf xs)).map (· + 1)
  else (List.range (modulus - 1)).map (· + 1)

/-- Whether some progression of `length` terms with nonzero common difference lies in the set. -/
def hasAp (modulus length : Nat) (xs : List Nat) : Bool :=
  xs.any fun a => (apDiffs modulus xs).any fun d =>
    (List.range length).all fun i => xs.contains (apTerm modulus a d i)

def isApFree (length modulus : Nat) (xs : List Nat) : Bool := !hasAp modulus length xs

/-- `|S + S| / |S| ≤ num / den`, as an exact integer comparison. -/
def isSmallDoubling (modulus num den : Nat) (xs : List Nat) : Bool :=
  decide (sumsetSize modulus xs * den ≤ num * xs.length)

/-! ## Dispatch -/

/-- The harness passes arguments by name, so each binder is the camelCase of the manifest's
argument name. -/
inductive Op
  | isSumFree (modulus : Nat)
  | isSidon (modulus : Nat)
  | isApFree (length modulus : Nat)
  | extendsSumFree (context : List Nat) (modulus : Nat)
  | extendsSidon (context : List Nat) (modulus : Nat)
  | extendsApFree (context : List Nat) (length modulus : Nat)
  | sumsetSize (modulus : Nat)
  | differenceSetSize (modulus : Nat)
  | schurTripleCount (modulus : Nat)
  | maxDifferenceMultiplicity (modulus : Nat)
  | isSmallDoubling (boundDen boundNum modulus : Nat)

def Op.modulus : Op → Nat
  | .isSumFree m | .isSidon m | .isApFree _ m | .sumsetSize m | .differenceSetSize m
  | .schurTripleCount m | .maxDifferenceMultiplicity m => m
  | .extendsSumFree _ m | .extendsSidon _ m | .extendsApFree _ _ m => m
  | .isSmallDoubling _ _ m => m

/-- The fixed set an `extends_*` operation adds to every member; empty otherwise. -/
def Op.context : Op → List Nat
  | .extendsSumFree c _ | .extendsSidon c _ | .extendsApFree c _ _ => c
  | _ => []

/-- A context is a set of the ambient group that meets no member. -/
def validContext (modulus : Nat) (context : List Nat) (f : Family) : Bool :=
  validRows modulus (context.map ([·])) &&
    f.members.all fun m => (elems m).all (!context.contains ·)

/-- Arguments the operations reject: a progression needs at least two terms, and a doubling
bound needs a nonzero denominator. -/
def Op.argsOk : Op → Bool
  | .isApFree length _ => 2 ≤ length
  | .extendsApFree _ length _ => 2 ≤ length
  | .isSmallDoubling boundDen _ _ => 1 ≤ boundDen
  | _ => true

inductive Value
  | unit
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  if !(op.argsOk && validFamily op.modulus f && validContext op.modulus op.context f) then .invalid else
  let ms := f.members
  let sets := ms.map elems
  match op with
  | .isSumFree n => reduceBool red ms (sets.map (isSumFree n))
  | .isSidon n => reduceBool red ms (sets.map (isSidon n))
  | .isApFree length n => reduceBool red ms (sets.map (isApFree length n))
  | .extendsSumFree c n => reduceBool red ms (sets.map fun xs => isSumFree n (c ++ xs))
  | .extendsSidon c n => reduceBool red ms (sets.map fun xs => isSidon n (c ++ xs))
  | .extendsApFree c length n => reduceBool red ms (sets.map fun xs => isApFree length n (c ++ xs))
  | .isSmallDoubling boundDen boundNum n =>
    reduceBool red ms (sets.map (isSmallDoubling n boundNum boundDen))
  | .sumsetSize n => reduceInt red ms (sets.map (sumsetSize n))
  | .differenceSetSize n => reduceInt red ms (sets.map (differenceSetSize n))
  | .schurTripleCount n => reduceInt red ms (sets.map (schurTripleCount n))
  | .maxDifferenceMultiplicity n => reduceInt red ms (sets.map (maxDifferenceMultiplicity n))

end Sum_free_and_additive
