import Gfp.Reference

/-!
# char_poly: executable reference

Polynomials are coefficient lists in increasing degree. `trim` removes zero coefficients above
one polynomial's leading term; exported polynomial rows are padded on the right to length `n+1`.
The rational form uses row-companion blocks for invariant factors in a divisibility chain.
Everything here uses structural recursion so the kernel can evaluate the oracle cases.
-/

namespace Char_poly

open Lk

abbrev Poly := List Nat

/-! ## Polynomials over F_p -/

def trim (f : Poly) : Poly := ((f.reverse.dropWhile (· = 0)).reverse)

def coeff (f : Poly) (i : Nat) : Nat := f.getD i 0

def polyAdd (p : Nat) (f g : Poly) : Poly :=
  trim ((List.range (max f.length g.length)).map fun i => (coeff f i + coeff g i) % p)

def polyNeg (p : Nat) (f : Poly) : Poly := f.map fun x => (p - x) % p

def polySub (p : Nat) (f g : Poly) : Poly := polyAdd p f (polyNeg p g)

def polyMul (p : Nat) (f g : Poly) : Poly :=
  if f.isEmpty ∨ g.isEmpty then [] else
  trim ((List.range (f.length + g.length - 1)).map fun k =>
    (List.range (k + 1)).foldl (fun s i => (s + coeff f i * coeff g (k - i)) % p) 0)

def polyScale (p c : Nat) (f : Poly) : Poly := trim (f.map fun x => x * c % p)

def polyPow (p : Nat) (f : Poly) : Nat → Poly
  | 0 => [1]
  | n + 1 => polyMul p (polyPow p f n) f

def monic (p : Nat) (f : Poly) : Poly :=
  let f := trim f
  if f.isEmpty then [] else polyScale p (inv p (f.reverse.headD 0)) f

def polyDivModAux (p : Nat) (g : Poly) : Nat → Poly → Poly × Poly
  | 0, f => ([], trim f)
  | fuel + 1, f =>
    let f := trim f
    if f.length < g.length ∨ g.isEmpty then ([], f) else
    let shift := f.length - g.length
    let term := List.replicate shift 0 ++ [f.reverse.headD 0]
    let qr := polyDivModAux p g fuel (polySub p f (polyMul p term g))
    (polyAdd p term qr.1, qr.2)

def polyDivMod (p : Nat) (f g : Poly) : Poly × Poly :=
  let g := monic p g
  polyDivModAux p g (f.length + 1) f

def derivative (p : Nat) (f : Poly) : Poly :=
  trim ((List.range (f.length - 1)).map fun i => ((i + 1) * coeff f (i + 1)) % p)

def polyGcdAux (p : Nat) : Nat → Poly → Poly → Poly
  | 0, f, _ => monic p f
  | fuel + 1, f, g =>
    let g := trim g
    if g.isEmpty then monic p f
    else polyGcdAux p fuel g (polyDivMod p f g).2

def polyGcd (p : Nat) (f g : Poly) : Poly := polyGcdAux p (f.length + g.length + 1) f g

def padPolynomial (n : Nat) (f : Poly) : Poly := (trim f).take n ++ List.replicate (n - (trim f).length) 0

/-! ## Characteristic and minimal polynomials -/

abbrev PolyMat := List (List Poly)

/-- The minor after deleting the first row and column `j`. -/
def minor (m : PolyMat) (j : Nat) : PolyMat :=
  m.drop 1 |>.map fun row =>
    (List.range row.length).filterMap fun i => if i = j then none else some (row.getD i [])

/-- Laplace expansion with an explicit dimension argument. Oracle matrices are deliberately small. -/
def determinantPoly (p : Nat) : Nat → PolyMat → Poly
  | 0, _ => [1]
  | n + 1, m =>
    let row := m.headD []
    (List.range row.length).foldl (fun total j =>
      let term := polyMul p (row.getD j []) (determinantPoly p n (minor m j))
      if j % 2 = 0 then polyAdd p total term else polySub p total term) []

def xIminus (p : Nat) (a : Mat) : PolyMat :=
  (List.range a.length).map fun i =>
    (List.range a.length).map fun j =>
      if i = j then [(p - (a.getD i []).getD j 0) % p, 1]
      else trim [(p - (a.getD i []).getD j 0) % p]

def characteristicPolynomial (p : Nat) (a : Mat) : Poly :=
  trim (determinantPoly p a.length (xIminus p a))

def matrixPowers (p : Nat) (a : Mat) : Nat → List Mat
  | 0 => [identity a.length]
  | n + 1 =>
    let ps := matrixPowers p a n
    ps ++ [matmul p (ps.reverse.headD (identity a.length)) a]

def relationAt (p : Nat) (powers : List Mat) (d : Nat) : Option Poly :=
  let previous := (powers.take d).map List.flatten
  let target := (powers.getD d []).flatten
  let equations := (List.range target.length).map fun i => previous.map fun v => v.getD i 0
  let rhs := target.map fun x => (p - x) % p
  (Gfp.solve p equations rhs).map fun cs => cs ++ [1]

def firstRelation (p : Nat) (powers : List Mat) : List Nat → Poly
  | [] => []
  | d :: ds => match relationAt p powers d with
    | some f => f
    | none => firstRelation p powers ds

def minimalPolynomial (p : Nat) (a : Mat) : Poly :=
  let n := a.length
  firstRelation p (matrixPowers p a n) ((List.range n).map (· + 1))

/-! ## Invariant factors and rational canonical form -/

def monicPolynomials (p d : Nat) : List Poly := (tuples p d).map fun cs => cs ++ [1]

def properMonicDivisors (p : Nat) (f : Poly) : List Poly :=
  (List.range (((trim f).length - 1) / 2)).flatMap fun i => monicPolynomials p (i + 1)

def factorFuel (p : Nat) : Nat → Poly → List Poly
  | 0, f => [monic p f]
  | fuel + 1, f =>
    let f := monic p f
    match (properMonicDivisors p f).find? fun g => (polyDivMod p f g).2.isEmpty with
    | none => [f]
    | some g => factorFuel p fuel g ++ factorFuel p fuel (polyDivMod p f g).1

def irreducibleFactors (p : Nat) (f : Poly) : List Poly :=
  sortBy lexLe (factorFuel p (trim f).length f)

def uniquePolynomials (fs : List Poly) : List Poly :=
  fs.foldl (fun out f => if f ∈ out then out else out ++ [f]) []

def zeroMatrix (n : Nat) : Mat := List.replicate n (List.replicate n 0)

def matAdd (p : Nat) (a b : Mat) : Mat :=
  List.zipWith (fun r s => List.zipWith (fun x y => (x + y) % p) r s) a b

def scalarIdentity (p n c : Nat) : Mat :=
  (List.range n).map fun i => (List.range n).map fun j => if i = j then c % p else 0

def evaluateAtMatrix (p : Nat) (a : Mat) (f : Poly) : Mat :=
  f.reverse.foldl (fun value c => matAdd p (matmul p value a) (scalarIdentity p a.length c))
    (zeroMatrix a.length)

def matrixPower (p : Nat) (a : Mat) : Nat → Mat
  | 0 => identity a.length
  | k + 1 => matmul p (matrixPower p a k) a

def nullityOfPower (p : Nat) (a : Mat) (k : Nat) : Nat := a.length - Gfp.rank p (matrixPower p a k)

def blockExponents (p : Nat) (a : Mat) (f : Poly) (multiplicity : Nat) : List Nat :=
  let b := evaluateAtMatrix p a f
  let degree := (trim f).length - 1
  let atLeast := fun k =>
    if k = 0 ∨ multiplicity < k then 0
    else (nullityOfPower p b k - nullityOfPower p b (k - 1)) / degree
  (List.range multiplicity).flatMap fun i =>
    let k := i + 1
    List.replicate (atLeast k - atLeast (k + 1)) k

def invariantFactors (p : Nat) (a : Mat) : List Poly :=
  let factors := irreducibleFactors p (characteristicPolynomial p a)
  let data := (uniquePolynomials factors).map fun f =>
    (f, blockExponents p a f (factors.filter (· = f)).length)
  let count := data.foldl (fun r item => max r item.2.length) 0
  (List.range count).map fun j =>
    data.foldl (fun q item =>
      let f := item.1
      let exponents := item.2
      if j < count - exponents.length then q
      else polyMul p q (polyPow p f (exponents.getD (j - (count - exponents.length)) 0))) [1]

def companion (p : Nat) (f : Poly) : Mat :=
  let f := trim f
  let n := f.length - 1
  (List.range n).map fun i =>
    (List.range n).map fun j =>
      if i + 1 < n then if j = i + 1 then 1 else 0
      else (p - coeff f j) % p

def blockSum (a b : Mat) : Mat :=
  let ca := (a.headD []).length
  let cb := (b.headD []).length
  (a.map fun row => row ++ List.replicate cb 0) ++
    (b.map fun row => List.replicate ca 0 ++ row)

def rationalCanonicalForm (p : Nat) (a : Mat) : Mat :=
  (invariantFactors p a).foldl (fun out f => blockSum out (companion p f)) []

def conjugacyLabel (p : Nat) (a : Mat) : Mat :=
  let n := a.length
  let factors := invariantFactors p a
  (factors.map (padPolynomial (n + 1))) ++
    List.replicate (n - factors.length) (List.replicate (n + 1) 0)

/-! ## Predicates and order -/

def isRegular (p : Nat) (a : Mat) : Bool := (trim (minimalPolynomial p a)).length = a.length + 1

def isSemisimple (p : Nat) (a : Mat) : Bool :=
  polyGcd p (minimalPolynomial p a) (derivative p (minimalPolynomial p a)) = [1]

def matrixOrderAux (p : Nat) (a : Mat) : Nat → Nat → Mat → Nat
  | 0, _, _ => 0
  | fuel + 1, k, power =>
    let next := matmul p power a
    if next = identity a.length then k else matrixOrderAux p a fuel (k + 1) next

def elementOrder (p : Nat) (a : Mat) : Nat :=
  if Gfp.rank p a ≠ a.length then 0
  else matrixOrderAux p a (p ^ a.length) 1 (identity a.length)

/-! ## Operations -/

inductive Op
  | charpoly | minpoly | rationalCanonicalForm | conjugacyLabel
  | isRegular | isSemisimple | elementOrder

inductive Value
  | matrix (m : Mat)
  deriving DecidableEq, Repr

def run (op : Op) (f : Family) (red : Red) : Result Value :=
  let p := f.p
  let ms := f.members
  let n := (ms.headD []).length
  if p < 2 ∨ n = 0 ∨ (ms.any fun m => m.length ≠ n ∨ m.any (·.length ≠ n)) then .invalid else
  match op with
  | .charpoly => reduceValues red (ms.map fun a => .matrix [padPolynomial (n + 1) (characteristicPolynomial p a)])
  | .minpoly => reduceValues red (ms.map fun a => .matrix [padPolynomial (n + 1) (minimalPolynomial p a)])
  | .rationalCanonicalForm => reduceValues red (ms.map fun a => .matrix (rationalCanonicalForm p a))
  | .conjugacyLabel => reduceValues red (ms.map fun a => .matrix (conjugacyLabel p a))
  | .isRegular => reduceBool red ms (ms.map (isRegular p))
  | .isSemisimple => reduceBool red ms (ms.map (isSemisimple p))
  | .elementOrder => reduceInt red ms (ms.map (elementOrder p))

end Char_poly
