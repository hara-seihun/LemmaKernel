import Mathlib
import Simplicial_complexes.Reference
import Lk.Contract

/-!
# simplicial_complexes: contract

What each output of `Reference.lean` means in Mathlib's terms. These are the statements a caller
relies on when they use the module. They are stated, not proved: the native backends are checked
against `Reference.lean` by `decide`, and `Reference.lean` is related to Mathlib here. A theorem
below with `sorry` is an obligation that has been written down and not yet discharged; there is
no other tooling that would make the native code "verified", and we do not claim it is.

A face of the reference is a list of vertices; here it is the finite set `F.toFinset`. The
complex of a member is `FaceSet`, the nonempty faces, and `boundaryMap` is the simplicial
boundary as a linear map of coordinate spaces over `ZMod p`, in the bases the reference lists.
-/

namespace Simplicial_complexes.Contract

open Lk Lk.Contract

variable {p n d : ℕ} [Fact p.Prime] {gens : List Face} {fs : List Face}

/-- Vertex sets of the rows, as finite sets. -/
def generatorSets (gens : List Face) : List (Finset ℕ) := gens.map List.toFinset

/-- The complex a member describes: nonempty subsets of a generator (`nonfaces = 0`), or
nonempty sets containing no forbidden generator (`nonfaces = 1`). -/
def FaceSet (n : ℕ) (gens : List Face) (nonfaces : ℕ) : Set (Finset ℕ) :=
  {T | T.Nonempty ∧ (∀ v ∈ T, v < n) ∧
    if nonfaces = 0 then ∃ S ∈ generatorSets gens, T ⊆ S
    else ∀ S ∈ generatorSets gens, ¬ S ⊆ T}

/-- Both readings give an abstract simplicial complex: nonempty subsets of a face are faces. -/
theorem faceSet_downward_closed (nonfaces : ℕ) :
    ∀ T ∈ FaceSet n gens nonfaces, ∀ U : Finset ℕ, U.Nonempty → U ⊆ T →
      U ∈ FaceSet n gens nonfaces := by
  sorry

/-- `facesOf` lists exactly that complex, each face once, as a sorted vertex list. -/
theorem facesOf_spec (nonfaces : ℕ) :
    (facesOf (candidates n) gens nonfaces).Nodup ∧
    (∀ F ∈ facesOf (candidates n) gens nonfaces, F.Pairwise (· < ·)) ∧
    ∀ T : Finset ℕ, T ∈ FaceSet n gens nonfaces ↔
      ∃ F ∈ facesOf (candidates n) gens nonfaces, F.toFinset = T := by
  sorry

/-! ## The f-vector and the Euler characteristic -/

/-- `f_d` counts the faces with `d + 1` vertices. -/
theorem fCount_spec (nonfaces : ℕ) :
    fCount (facesOf (candidates n) gens nonfaces) d =
      {T ∈ FaceSet n gens nonfaces | T.card = d + 1}.ncard := by
  sorry

/-- `faces` is the whole complex. -/
theorem faces_spec (nonfaces : ℕ) :
    (facesOf (candidates n) gens nonfaces).length = (FaceSet n gens nonfaces).ncard := by
  sorry

/-- `euler_characteristic` is the alternating sum of the f-vector, read in `ZMod p`. The answer
is that residue as a natural number below `p`. -/
theorem eulerMod_spec :
    (eulerMod p fs : ZMod p) =
      ∑ i ∈ Finset.range (fs.length + 1), (-1 : ZMod p) ^ i * (fCount fs i : ZMod p) ∧
    eulerMod p fs < p := by
  sorry

/-! ## Homology -/

/-- The boundary matrix of dimension `d` over `ZMod p`, rows indexed by the `d`-faces and columns
by the `(d-1)`-faces of `fs`, in the order `facesOf` lists them. -/
noncomputable def boundaryMatrix (p : ℕ) (fs : List Face) (d : ℕ) :
    Matrix (Fin (fCount fs d)) (Fin (fCount fs (d - 1))) (ZMod p) :=
  toMatrix p (fCount fs d) (fCount fs (d - 1)) (boundary p fs d)

/-- `boundary_d` as a linear map from `d`-chains to `(d-1)`-chains. -/
noncomputable def boundaryMap (p : ℕ) (fs : List Face) (d : ℕ) :
    (Fin (fCount fs d) → ZMod p) →ₗ[ZMod p] (Fin (fCount fs (d - 1)) → ZMod p) :=
  Matrix.vecMulLinear (boundaryMatrix p fs d)

/-- The entry of `boundary_d` at a `d`-face `F` and a `(d-1)`-face `G` is `(-1)^j` when `G` is
`F` without its `j`-th vertex, and zero when `G` is not a facet of `F`. -/
theorem boundaryMatrix_spec (i : Fin (fCount fs d)) (j : Fin (fCount fs (d - 1))) :
    let F := (fs.filter fun F => F.length = d + 1).getD i []
    let G := (fs.filter fun F => F.length = d).getD j []
    boundaryMatrix p fs d i j =
      match (List.range F.length).find? fun k => F.eraseIdx k = G with
      | some k => (-1 : ZMod p) ^ k
      | none => 0 := by
  sorry

/-- A chain complex: two boundaries in a row are zero. -/
theorem boundaryMap_comp : (boundaryMap p fs d).comp (boundaryMap p fs (d + 1)) = 0 := by
  sorry

/-- Homology in degree `d`: cycles modulo boundaries. -/
noncomputable def homology (p : ℕ) (fs : List Face) (d : ℕ) : Type :=
  (LinearMap.ker (boundaryMap p fs d)) ⧸
    ((LinearMap.range (boundaryMap p fs (d + 1))).comap
      (LinearMap.ker (boundaryMap p fs d)).subtype)

noncomputable instance : AddCommGroup (homology p fs d) := by
  unfold homology; infer_instance

noncomputable instance : Module (ZMod p) (homology p fs d) := by
  unfold homology; infer_instance

/-- `betti p fs d` is the dimension of the `d`-th homology of the complex over `F_p`: this is
unreduced simplicial homology, so `betti p fs 0` is the number of connected components. -/
theorem betti_spec : betti p fs d = Module.finrank (ZMod p) (homology p fs d) := by
  sorry

/-- Rank-nullity, the identity the reference computes with. -/
theorem betti_eq_nullity_sub_rank :
    betti p fs d =
      Module.finrank (ZMod p) (LinearMap.ker (boundaryMap p fs d)) -
        (boundaryMatrix p fs (d + 1)).rank := by
  sorry

/-! ## Shellability -/

/-- The facets: the maximal faces of the complex. -/
theorem facetsOf_spec :
    ∀ F : Face, F ∈ facetsOf fs ↔ F ∈ fs ∧ ∀ G ∈ fs, F.toFinset ⊆ G.toFinset → G.toFinset = F.toFinset := by
  sorry

/-- Bjorner-Wachs shelling, non-pure allowed: every earlier facet meets `F_j` inside a facet that
misses exactly one of its vertices. -/
def IsShelling (facets : List Face) (order : List Face) : Prop :=
  order.Perm facets ∧
    ∀ i j, i < j → j < order.length →
      ∃ k < j, ((order.getD i []).toFinset ∩ (order.getD j []).toFinset) ⊆ (order.getD k []).toFinset ∧
        ((order.getD j []).toFinset \ (order.getD k []).toFinset).card = 1

theorem isShellable_spec :
    isShellable fs = true ↔ ∃ order, IsShelling (facetsOf fs) order := by
  sorry

/-- A shellable complex of dimension `d` is homotopy equivalent to a wedge of spheres, so its
reduced homology is concentrated in the top dimension: the statement a caller uses shellability
for. -/
theorem shellable_homology (hs : isShellable fs = true)
    (hpure : ∀ F ∈ facetsOf fs, F.length = d + 1) :
    ∀ i, 0 < i → i < d → betti p fs i = 0 := by
  sorry

end Simplicial_complexes.Contract
