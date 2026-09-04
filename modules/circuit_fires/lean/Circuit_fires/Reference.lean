import Gfp.Reference

/-!
# circuit_fires: executable reference

A configuration row is `[z_i | u_i]`. The direct reference writes every finite-difference
constraint as a row of one homogeneous linear system. A nonzero total defect exists exactly when
the sum-of-defects functional is not in that system's row space.
-/

namespace Circuit_fires

open Lk

def firstNonzero : Vec → Option Nat
  | [] => none
  | x :: xs => if x = 0 then firstNonzero xs else some x

def normalized (p : Nat) (row : Vec) : Vec :=
  match firstNonzero row with
  | none => []
  | some value => row.map fun entry => entry * inv p value % p

def nonzero (row : Vec) : Bool := row.any fun value => value != 0

def directions (configuration : Mat) (g : Nat) : Mat := configuration.map (·.take g)

def covectors (configuration : Mat) (g : Nat) : Mat := configuration.map (·.drop g)

def tensor (p : Nat) (z u : Vec) : Vec :=
  u.flatMap fun a => z.map fun b => a * b % p

def addRows (p : Nat) (left right : Vec) : Vec :=
  List.zipWith (fun a b => (a + b) % p) left right

def isCircuit (p : Nat) (configuration : Mat) (g : Nat) : Bool :=
  let k := configuration.length
  let columns := (configuration.headD []).length
  let h := columns - g
  let zs := directions configuration g
  let us := covectors configuration g
  let tensors := List.zipWith (tensor p) zs us
  let total := tensors.foldl (addRows p) (List.replicate (g * h) 0)
  0 < k && 0 < g && g < columns &&
    zs.all nonzero && us.all nonzero &&
    (zs.map (normalized p)).eraseDups.length = k &&
    Gfp.rank p zs = g && Gfp.rank p us = h &&
    (total.all fun value => value = 0) &&
    Gfp.rank p tensors + 1 = k

def points (p g : Nat) : List Vec := (tuples p g).map List.reverse

def add (p : Nat) (left right : Vec) : Vec :=
  List.zipWith (fun a b => (a + b) % p) left right

def equationRows (p : Nat) (configuration : Mat) (g : Nat) : Mat :=
  let xs := points p g
  let h := (configuration.headD []).length - g
  let k := configuration.length
  (List.range k).flatMap fun i =>
    let z := (configuration.getD i []).take g
    let u := (configuration.getD i []).drop g
    xs.map fun x =>
      let shifted := add p x z
      let potential := xs.flatMap fun y =>
        (List.range h).map fun coordinate =>
          ((if y = shifted then u.getD coordinate 0 else 0) +
           (if y = x then p - u.getD coordinate 0 else 0)) % p
      let defects := (List.range k).map fun j => if i = j then p - 1 else 0
      potential ++ defects

def pointedRows (p : Nat) (configuration : Mat) (g : Nat) : Mat :=
  let h := (configuration.headD []).length - g
  let width := (points p g).length * h + configuration.length
  (List.range h).map fun coordinate =>
    (List.range width).map fun column => if column = coordinate then 1 else 0

def admitsNonzeroDefect (p : Nat) (configuration : Mat) (g : Nat) : Bool :=
  let equations := equationRows p configuration g ++ pointedRows p configuration g
  let potentialColumns := (points p g).length * ((configuration.headD []).length - g)
  let carry := List.replicate potentialColumns 0 ++ List.replicate configuration.length 1
  !Gfp.inSpan p equations carry

def isFire (p : Nat) (configuration : Mat) (g : Nat) : Bool :=
  isCircuit p configuration g && admitsNonzeroDefect p configuration g

/-- One potential followed by its defect vector, normalized to total defect one. -/
def fireSolution (p : Nat) (configuration : Mat) (g : Nat) : Option Vec :=
  if !isCircuit p configuration g then none else
  let equations := equationRows p configuration g ++ pointedRows p configuration g
  let potentialColumns := (points p g).length * ((configuration.headD []).length - g)
  let carry := List.replicate potentialColumns 0 ++ List.replicate configuration.length 1
  Gfp.solve p (equations ++ [carry]) (List.replicate equations.length 0 ++ [1])

def dot (p : Nat) (left right : Vec) : Nat :=
  (List.zipWith (fun a b => a * b) left right).foldl (· + ·) 0 % p

def verifiesPotential (p : Nat) (configuration : Mat) (g : Nat)
    (defects : Vec) (potential : Mat) : Bool :=
  let xs := points p g
  let h := (configuration.headD []).length - g
  isCircuit p configuration g &&
    defects.length = configuration.length &&
    potential.length = xs.length &&
    (potential.all fun value => value.length = h) &&
    ((potential.headD []).all fun value => value = 0) &&
    defects.foldl (· + ·) 0 % p != 0 &&
    ((List.range configuration.length).all fun i =>
      let row := configuration.getD i []
      let z := row.take g
      let u := row.drop g
      xs.all fun x =>
        let shifted := add p x z
        let currentValue := potential.getD (xs.idxOf x) []
        let shiftedValue := potential.getD (xs.idxOf shifted) []
        let difference := List.zipWith (fun a b => (a + p - b) % p) shiftedValue currentValue
        dot p u difference = defects.getD i 0)

inductive Op
  | isCircuit (baseDim : Nat)
  | isFire (baseDim : Nat)
  | findPotential (baseDim : Nat)
  | verifiesPotential (baseDim : Nat) (defects : Vec) (potential : Mat)

inductive Value
  | solution (value : Option Vec)
  deriving DecidableEq, Repr


def supportedPrime (p : Nat) : Bool := p = 2 || p = 3 || p = 5 || p = 7


def validBaseDimension (configurations : List Mat) (g : Nat) : Bool :=
  0 < g && g < ((configurations.headD []).headD []).length


def run (op : Op) (family : Family) (red : Red) : Result Value :=
  match family with
  | .explicit p configurations =>
      if !supportedPrime p then .invalid else
      match op with
      | .isCircuit g =>
          if !validBaseDimension configurations g then .invalid
          else reduceBool red configurations (configurations.map fun c => isCircuit p c g)
      | .isFire g =>
          if !validBaseDimension configurations g then .invalid
          else reduceBool red configurations (configurations.map fun c => isFire p c g)
      | .findPotential g =>
          let rows := (configurations.headD []).length
          let h := ((configurations.headD []).headD []).length - g
          let variables := p ^ g * h + rows
          let equations := rows * p ^ g + h + 1
          if !validBaseDimension configurations g || 1024 < variables ||
              2 ^ 24 < equations * (variables + 1) then .invalid
          else reduceValues red (configurations.map fun c => .solution (fireSolution p c g))
      | .verifiesPotential g defects potential =>
          let rows := (configurations.headD []).length
          let h := ((configurations.headD []).headD []).length - g
          if !validBaseDimension configurations g || defects.length != rows ||
              potential.length != p ^ g || !(potential.all fun row => row.length = h) then .invalid
          else reduceBool red configurations
            (configurations.map fun c => verifiesPotential p c g defects potential)
  | _ => .invalid

end Circuit_fires
