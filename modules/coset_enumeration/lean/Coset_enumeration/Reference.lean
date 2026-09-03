import Lk.Reference

/-!
# coset_enumeration: executable reference

Letters `2*i` and `2*i+1` are generator `i` and its inverse. Fixed relations use the same
encoded equations as `words_and_growth`; each member of a `words` family adds the equation
`word = 1`. The table uses least allocated row numbers as coincidence representatives, so
completed cosets and the resulting permutations have one canonical order. Exhausting
`maxCosets` returns `none`; it makes no claim that the subgroup has infinite index.
-/

namespace Coset_enumeration

open Lk

structure Equation where
  left : List Nat
  right : List Nat
  deriving DecidableEq, Repr

/-- `[left length, right length, left, right, padding]`, with every padding entry equal to `q`. -/
def decodeEquation (q : Nat) (row : Vec) : Option Equation :=
  if row.length < 2 then none else
  let nl := row.getD 0 0
  let nr := row.getD 1 0
  if row.length < 2 + nl + nr then none else
  let left := (row.drop 2).take nl
  let right := (row.drop (2 + nl)).take nr
  let padding := row.drop (2 + nl + nr)
  if (left ++ right).all (· < q) ∧ padding.all (· = q)
  then some { left, right }
  else none

def decodeEquations (q : Nat) : List Vec → Option (List Equation)
  | [] => some []
  | row :: rows =>
    match decodeEquation q row, decodeEquations q rows with
    | some equation, some equations => some (equation :: equations)
    | _, _ => none

def inverseWord (word : List Nat) : List Nat := word.reverse.map (· ^^^ 1)

def equationRelator (equation : Equation) : List Nat :=
  equation.left ++ inverseWord equation.right

def equationRelators (equations : List Equation) : List (List Nat) :=
  equations.map equationRelator

structure State where
  table : List (List (Option Nat))
  parent : List Nat
  failed : Bool
  deriving DecidableEq, Repr

inductive Task
  | link (a letter b : Nat)
  | merge (a b : Nat)
  deriving DecidableEq, Repr

def rootAux (parent : List Nat) : Nat → Nat → Nat
  | 0, x => x
  | fuel + 1, x =>
    let y := parent.getD x x
    if y = x then x else rootAux parent fuel y

def root (s : State) (x : Nat) : Nat := rootAux s.parent s.parent.length x

def cell (s : State) (x letter : Nat) : Option Nat :=
  ((s.table.getD (root s x) []).getD letter none).map (root s)

def setCell (s : State) (x letter : Nat) (value : Option Nat) : State :=
  { s with table := s.table.set x ((s.table.getD x []).set letter value) }

def fail (s : State) : State := { s with failed := true }

def rowTasks (letters lo hi : Nat) (s : State) : List Task :=
  (List.range letters).filterMap fun letter =>
    match (s.table.getD hi []).getD letter none with
    | some value => some (.link lo letter value)
    | none => none

def processTasks (letters : Nat) : Nat → State → List Task → State
  | 0, s, tasks => if tasks.isEmpty then s else fail s
  | _fuel + 1, s, [] => s
  | fuel + 1, s, Task.merge a b :: tasks =>
    let ra := root s a
    let rb := root s b
    if ra = rb then processTasks letters fuel s tasks
    else
      let lo := min ra rb
      let hi := max ra rb
      let next := { s with parent := s.parent.set hi lo }
      processTasks letters fuel next (rowTasks letters lo hi s ++ tasks)
  | fuel + 1, s, Task.link a letter b :: tasks =>
    let ra := root s a
    let rb := root s b
    match cell s ra letter with
    | some forward =>
      if forward ≠ rb then
        processTasks letters fuel s (Task.merge forward rb :: Task.link ra letter rb :: tasks)
      else
        match cell s rb (letter ^^^ 1) with
        | some backward =>
          if backward ≠ ra then
            processTasks letters fuel s (Task.merge backward ra :: Task.link ra letter rb :: tasks)
          else
            let next := setCell (setCell s ra letter (some rb)) rb (letter ^^^ 1) (some ra)
            processTasks letters fuel next tasks
        | none =>
          let next := setCell (setCell s ra letter (some rb)) rb (letter ^^^ 1) (some ra)
          processTasks letters fuel next tasks
    | none =>
      match cell s rb (letter ^^^ 1) with
      | some backward =>
        if backward ≠ ra then
          processTasks letters fuel s (Task.merge backward ra :: Task.link ra letter rb :: tasks)
        else
          let next := setCell (setCell s ra letter (some rb)) rb (letter ^^^ 1) (some ra)
          processTasks letters fuel next tasks
      | none =>
        let next := setCell (setCell s ra letter (some rb)) rb (letter ^^^ 1) (some ra)
        processTasks letters fuel next tasks

def taskBudget (letters bound : Nat) : Nat := 1024 + 16 * (bound + 1) * (bound + 1) * (letters + 1)

def runTasks (letters bound : Nat) (s : State) (tasks : List Task) : State :=
  processTasks letters (taskBudget letters bound) s tasks

def addCoset (letters bound : Nat) (s : State) : State × Option Nat :=
  if s.parent.length ≥ bound then (fail s, none)
  else
    let c := s.parent.length
    ({ s with table := s.table ++ [List.replicate letters none], parent := s.parent ++ [c] }, some c)

def walkForward (s : State) : Nat → List Nat → Nat × List Nat
  | c, [] => (root s c, [])
  | c, letter :: rest =>
    match cell s c letter with
    | some next => walkForward s next rest
    | none => (root s c, letter :: rest)

def walkBackward (s : State) : Nat → List Nat → Nat × List Nat
  | c, [] => (root s c, [])
  | c, letter :: rest =>
    match cell s c (letter ^^^ 1) with
    | some previous => walkBackward s previous rest
    | none => (root s c, letter :: rest)

def enforce (letters bound : Nat) : Nat → State → Nat → List Nat → Nat → State
  | 0, s, _, _, _ => fail s
  | fuel + 1, s, start, word, finish =>
    if s.failed then s
    else
      let (a, remaining) := walkForward s start word
      let (b, reverseGap) := walkBackward s finish remaining.reverse
      match reverseGap.reverse with
      | [] => runTasks letters bound s [.merge a b]
      | [letter] => runTasks letters bound s [.link a letter b]
      | letter :: _ :: _ =>
        let (next, fresh) := addCoset letters bound s
        match fresh with
        | none => next
        | some c =>
          let linked := runTasks letters bound next [.link a letter c]
          enforce letters bound fuel linked start word finish

def enforceOne (letters bound coset : Nat) (s : State) (word : List Nat) : State :=
  enforce letters bound (bound + word.length + 2) s coset word coset

def enforceWords (letters bound coset : Nat) : State → List (List Nat) → State
  | s, [] => s
  | s, word :: rest => enforceWords letters bound coset (enforceOne letters bound coset s word) rest

def fillGenerators (letters bound coset : Nat) : State → List Nat → State
  | s, [] => s
  | s, generator :: rest =>
    if s.failed then s
    else
      let c := root s coset
      match cell s c (2 * generator) with
      | some _ => fillGenerators letters bound coset s rest
      | none =>
        let (next, fresh) := addCoset letters bound s
        match fresh with
        | none => next
        | some d => fillGenerators letters bound coset
            (runTasks letters bound next [.link c (2 * generator) d]) rest

def enumerateRows (letters bound generators : Nat) (relators : List (List Nat)) :
    Nat → Nat → State → State
  | 0, cursor, s => if cursor < s.parent.length then fail s else s
  | fuel + 1, cursor, s =>
    if s.failed then s
    else if cursor < s.parent.length then
      let next := if root s cursor = cursor then
        fillGenerators letters bound cursor
          (enforceWords letters bound cursor s relators) (List.range generators)
        else s
      enumerateRows letters bound generators relators fuel (cursor + 1) next
    else s

structure Enumeration where
  degree : Nat
  permutations : List Perm
  deriving DecidableEq, Repr

def validWords (letters : Nat) (words : List (List Nat)) : Bool :=
  words.all fun word => word.all (· < letters)

/-- Fixed word lists use `letters` as trailing padding, so rows may have different lengths. -/
def unpadWord (letters : Nat) : List Nat → Option (List Nat)
  | [] => some []
  | value :: rest =>
    if value < letters then (unpadWord letters rest).map (value :: ·)
    else if value = letters && rest.all (· = letters) then some []
    else none

def unpadWords (letters : Nat) : List (List Nat) → Option (List (List Nat))
  | [] => some []
  | word :: rest =>
    match unpadWord letters word, unpadWords letters rest with
    | some head, some tail => some (head :: tail)
    | _, _ => none

def enumerate (generators bound : Nat) (relators subgroup : List (List Nat)) : Option Enumeration :=
  let letters := 2 * generators
  if generators = 0 || bound = 0 || !validWords letters (relators ++ subgroup) then none
  else
    let initial : State := { table := [List.replicate letters none], parent := [0], failed := false }
    let withSubgroup := enforceWords letters bound 0 initial subgroup
    let done := enumerateRows letters bound generators relators (bound + 1) 0 withSubgroup
    if done.failed then none
    else
      let roots := (List.range done.parent.length).filter fun i => root done i = i
      let permutations := (List.range generators).map fun generator =>
        roots.map fun c => roots.idxOf ((cell done c (2 * generator)).getD 0)
      some { degree := roots.length, permutations := permutations }

inductive Op
  | index (generators maxCosets : Nat) (relations subgroup : List (List Nat))
  | permutationRepresentation (generators maxCosets : Nat) (relations subgroup : List (List Nat))
  | isFinite (generators maxCosets : Nat) (relations : List (List Nat))

inductive Value
  | representation (action : Option (List Perm))
  deriving DecidableEq, Repr

def answers (generators maxCosets : Nat) (encodedRelations subgroup : List (List Nat)) (f : Family) :
    Option (List (Option Enumeration)) :=
  match f with
  | .words alphabet _ =>
    if generators ≠ 0 && generators < 2 ^ 31 && maxCosets ≠ 0 && maxCosets < 2 ^ 32 &&
        alphabet = 2 * generators then
      match decodeEquations alphabet encodedRelations, unpadWords alphabet subgroup with
      | some equations, some generatorsOfSubgroup =>
        let relators := equationRelators equations
        some (f.members.map fun member =>
          enumerate generators maxCosets (relators ++ [member.headD []]) generatorsOfSubgroup)
      | _, _ => none
    else none
  | _ => none

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  match op with
  | .index generators maxCosets relations subgroup =>
    match answers generators maxCosets relations subgroup f with
    | some xs => reduceInt red f.members (xs.map fun answer => answer.map (·.degree) |>.getD 0)
    | none => .invalid
  | .permutationRepresentation generators maxCosets relations subgroup =>
    match answers generators maxCosets relations subgroup f with
    | some xs => reduceValues red (xs.map fun answer => .representation (answer.map (·.permutations)))
    | none => .invalid
  | .isFinite generators maxCosets relations =>
    match answers generators maxCosets relations [] f with
    | some xs => reduceBool red f.members (xs.map Option.isSome)
    | none => .invalid

end Coset_enumeration
