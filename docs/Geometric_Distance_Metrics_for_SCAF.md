# Geometric Distance Metrics for SCAF

**Status:** Design note — proposed extension to the SemSimula Causal Auditing Framework.  
**Date:** August 2026  
**Prerequisites:** [Framework_for_Causal_Analysis_SemSimula_Models.md](Framework_for_Causal_Analysis_SemSimula_Models.md), [Exploiting_the_Riemannian_geometry_of_conservative_language_models.md](https://github.com/dimitarpg13/semsimula-paper/blob/main/companion_notes/Exploiting_the_Riemannian_geometry_of_conservative_language_models.md), [Geodesic_Preservation_Experiment.md](https://github.com/dimitarpg13/semsimula-paper/blob/main/companion_notes/Geodesic_Preservation_Experiment.md)

---

## 1. Motivation

SCAF v0.1 detects and sizes causal leaks in logit space:

| Current metric | Space | Purpose |
|---|---|---|
| L∞ (`linf`) | logits | Binary leak detection (exact zero for causal models) |
| AILE (mean \|Δlogit\|) | logits | Effect size |
| τ\_leak (honest − standard NLL) | nats | Task-level impact |
| logit\_l1 | logits per position | LeakFrame outcome for CATE |

These metrics are **necessary and sufficient for binary leak detection**: L∞ > 0 means future information reached past logits. But they are **blind to the model's internal geometry**. A perturbation that moves a hidden state from one attractor basin to another and a perturbation that nudges it slightly within the same basin produce the same kind of logit delta — the qualitative difference is invisible.

The Semantic Simulation framework equips the hidden-state space with a **Riemannian metric** — the Jacobi metric induced by the learned potential $V_\theta$ — whose structure (conformal flatness, layer-dependent conformal factor, intrinsic asymmetry from damping) provides a physically grounded notion of distance that logit-space metrics cannot access. Extending SCAF with this geometric layer enables four new capabilities:

1. **Pre-logit leak detection.** Hidden-state corruption that hasn't yet manifested at the output (e.g., register states carrying future information that haven't been decoded).
2. **Qualitative leak characterisation.** Continuous perturbation (small cosine deviation, same basin) vs. basin-crossing (different attractor well assignment) vs. trajectory derailment (large geodesic distance).
3. **Physically meaningful leak sizing.** The geodesic distance $d_{\text{geo}}(h_{\text{factual}} \to h_{\text{counterfactual}})$ measures "how far the leak moved the trajectory" in the model's own geometry, not in an arbitrary coordinate system.
4. **Leak-direction analysis.** The asymmetry ratio $d_{\text{geo}}(h_f \to h_c) / d_{\text{geo}}(h_c \to h_f)$ distinguishes leaks that propagate through the model's natural dynamics (ratio $\approx 1.35$–$1.40$, matching the architecture's measured asymmetry) from wiring-level bypasses (ratio $\approx 1.0$, short-circuiting the dynamics entirely).

---

## 2. Mathematical foundations

### 2.1 The Jacobi metric and conformal flatness

The SPLM-family Lagrangian $\mathcal{L} = T - V = \frac{1}{2}m\lVert\dot h\rVert^2 - V_\theta(h)$ induces the Jacobi metric:

$$\tilde g_{ij}(h) = 2(E - V_\theta(h)) \cdot m \cdot \delta_{ij}$$

This is **conformally flat**: $\tilde g = \Omega^2 g$ where $g_{ij} = \delta_{ij}$ is the flat Euclidean metric and $\Omega^2(h) = 2(E - V_\theta(h))m$ is a scalar conformal factor that varies with position. In the damped case (the actual operating regime), the layer-dependent version uses kinetic energy:

$$\Omega_\ell^2 = 2 T_\ell \cdot m = m^2 \lVert\dot h_\ell\rVert^2$$

confirmed positive at 100% of positions by Diagnostic Battery Arm 1.

### 2.2 Why cosine similarity is the conformally correct angular metric

For any conformal rescaling $\tilde g = \Omega^2 g$, the inner product and norms rescale as $\langle u,v\rangle_{\tilde g} = \Omega^2 \langle u,v\rangle_g$ and $\lVert u\rVert_{\tilde g} = \Omega\lVert u\rVert_g$. Therefore:

$$\cos_{\tilde g}(u,v) = \frac{\Omega^2 \langle u,v\rangle_g}{\Omega\lVert u\rVert_g \cdot \Omega\lVert v\rVert_g} = \cos_g(u,v)$$

The $\Omega$ factors cancel exactly. This is proven as Proposition `stp-conformal-invariance` in paper v5 §18. **Cosine similarity yields the same value in flat coordinates and in the curved Jacobi metric** — it is the unique angular metric that doesn't depend on where you are in the potential landscape.

Conversely, Euclidean/L2 distance $\lVert u\rVert_{\tilde g} = \Omega(h)\lVert u\rVert_g$ depends on the local conformal factor. Two identical coordinate displacements at different positions in the potential map to different physical distances. **Raw L2 is not a well-defined geometric quantity in this space.**

### 2.3 Geodesic distance and its inherent asymmetry

The true Riemannian distance between two hidden states is the arc length along the connecting geodesic:

$$d_{\text{geo}}(h_A \to h_B) = \int_0^1 \sqrt{m\lVert\dot h_{\ell(t)}\rVert} \lVert\dot\gamma(t)\rVert \mathrm{d}t$$

integrated along the damped geodesic $\gamma$ from $h_A$ to $h_B$ with the layer-dependent conformal factor.

Because the damped geodesic equation includes a friction term $-\gamma \dot h^k$, it is **not time-reversible**:

$$d_{\text{geo}}(h_A \to h_B) \neq d_{\text{geo}}(h_B \to h_A)$$

The Diagnostic Battery (Arm 5) measures asymmetry ratios of 1.35–1.40 across all three SPLM-family models. This is a structural consequence of the damped dynamics: paths "with the flow" (toward broader attractor basins) are shorter than paths against it.

### 2.4 Basin membership via anisotropic Gaussian wells

For models with Gaussian $V_\theta$ (particularly the anisotropic depth-conditioned variant), the learned parameters define a **local Mahalanobis-type metric** around each attractor well $k$:

$$d_{\text{Maha},k}^2(h) = (h - \mu_k)^\top \Sigma_k^{-1} (h - \mu_k)$$

where $\Sigma_k^{-1} = \mathrm{diag}(a_k) + B_k B_k^\top$ is the anisotropic precision matrix (diagonal + low-rank). The **dominant well assignment** is:

$$k^{\ast}(h) = \arg\min_k \bigl[ d_{\text{Maha},k}^2(h) - 2\log w_k \bigr]$$

(the well whose Gaussian bump contributes most to $V_\theta(h)$). This assignment is a discrete, structurally meaningful quantity: two hidden states in the same well are "semantically co-located" regardless of their raw coordinate distance.

<p align="center"><img src="images/scaf_basin_crossing_conformal_landscape.png" alt="A topographic map of a two-well potential landscape showing a factual hidden state and two counterfactual displacements: a small within-basin move (low cosine deviation, no well reassignment) versus a large basin-crossing move into the neighboring well (high cosine deviation and well reassignment). An inset shows that the cosine angle between two directions is the same whether measured deep in a well or near the ridge, while their Euclidean length is not." width="700"></p>

*Figure 1. Basin-crossing leaks vs. continuous perturbation. A future-token perturbation that keeps the past hidden state within its original attractor (top arrow) is a mild, within-basin deviation — nonzero cosine deviation but no change in dominant well. A perturbation that pushes the hidden state across the ridge into a different attractor (bottom arrow) is a basin-crossing leak — a qualitatively more severe corruption that logit-space metrics cannot distinguish from the mild case. The inset illustrates why cosine similarity, unlike raw Euclidean distance, gives the same reading regardless of the local depth of the potential (the conformal factor Omega), which is what makes it the geometrically correct choice for Tier A.*

---

## 3. Three tiers of geometric audit distance for SCAF

### Tier A: Conformally-invariant hidden-state leak detection

**What it measures:** cosine similarity between past hidden states under factual vs. counterfactual inputs — the hidden-state analogue of SCAF's existing $|\Delta\text{logit}|$.

**Metric:**

$$\Delta_{\cos}^{(\ell)}(t) = 1 - \cos\bigl(h_\ell^{(t)}[\text{factual}], h_\ell^{(t)}[\text{counterfactual}]\bigr)$$

for each layer $\ell$ and causal-prefix position $t \le t_p$.

**Why cosine, not L2:** conformal invariance (§2.2). The cosine deviation is the same quantity whether computed in flat $\mathbb{R}^d$ or in the curved Jacobi metric. L2 deviation would conflate the leak magnitude with the local potential energy.

**Why it matters beyond logit L∞:** a hidden-state leak that hasn't yet propagated to logits (e.g., register states carrying future information that the output projection kills, or a leak in an intermediate layer that a subsequent LayerNorm washes out of the magnitude but preserves directionally) would show $\Delta_{\cos}^{(\ell)} > 0$ even when logit L∞ $= 0$.

**Interpretation:**

| $\Delta_{\cos}$ value | Interpretation |
|---|---|
| Exactly 0.0 (all layers, all positions) | Strict causal integrity at the hidden-state level |
| $> 0$ at early layers, decaying to 0 at later layers | Transient perturbation absorbed by the dynamics |
| $> 0$ at late layers but logit L∞ $= 0$ | **Latent leak**: future information in the hidden state, output projection masks it |
| $> 0$ at late layers and logit L∞ $> 0$ | Manifest leak (standard SCAF detection would also catch this) |

**Per-layer profile as a diagnostic:** plotting $\Delta_{\cos}^{(\ell)}$ across layers reveals the leak's propagation path. A spike at the layer where the reverse channel is injected, decaying through subsequent layers, fingerprints the Fock reverse-channel leak mechanism. A spike at the embedding layer, growing through subsequent layers, fingerprints a masking/wiring bug.

### Tier B: Basin-membership audit

**What it measures:** whether perturbing future tokens changes the attractor well assignment of past hidden states.

**Metric:**

$$\beta^{(\ell)}(t) = \mathbb{1}\bigl[k^{\ast}_\ell(h^{(t)}_\ell[\text{factual}]) \neq k^{\ast}_\ell(h^{(t)}_\ell[\text{counterfactual}])\bigr]$$

where $k^{\ast}_\ell(h)$ is the dominant well index at layer $\ell$ (§2.4).

**Why it matters:** a basin-crossing leak is qualitatively more severe than a continuous perturbation. It means the future perturbation has moved the hidden state to a **different semantic attractor** — the model is computing a fundamentally different representation of the past, not just a slightly perturbed one. This is invisible to logit L∞ (which measures magnitude, not attractor structure) and to cosine similarity (which would show $\Delta_{\cos} > 0$ for both, without distinguishing the two cases).

**Aggregation:** the **basin-crossing rate** $\bar\beta = \mathbb{E}[\beta^{(\ell)}(t)]$ over positions and layers gives a single scalar measuring what fraction of past hidden states are knocked into a different well by the future perturbation. This is reported alongside AILE as a separate effect-size axis.

**Applicability:** requires Gaussian $V_\theta$ with accessible well parameters (centres $\mu_k$, precision $\Sigma_k^{-1}$, weights $w_k$). For MLP $V_\theta$ models, an approximate version can be obtained by clustering hidden states into pseudo-basins via k-means on the $V_\theta$ gradient field.

### Tier C: Asymmetric geodesic leak distance

**What it measures:** the damped-geodesic distance from the factual hidden state to the counterfactual one, and vice versa.

**Metrics:**

$$d_{\to}^{(\ell)}(t) = d_{\text{geo}}\bigl(h_\ell^{(t)}[\text{factual}] \to h_\ell^{(t)}[\text{counterfactual}]\bigr)$$

$$d_{\leftarrow}^{(\ell)}(t) = d_{\text{geo}}\bigl(h_\ell^{(t)}[\text{counterfactual}] \to h_\ell^{(t)}[\text{factual}]\bigr)$$

$$r^{(\ell)}(t) = d_{\to}^{(\ell)}(t) / d_{\leftarrow}^{(\ell)}(t)$$

**Computation:** requires shooting-method integration of the damped geodesic equation between the two hidden states, using the model's own $V_\theta$ for Christoffel symbols (closed-form for Gaussian $V_\theta$ via `analytical_grad`). This is expensive — $O(d \cdot n_{\text{steps}})$ per pair — and is intended for Tier-2 re-analysis of already-detected leaks, not for real-time monitoring.

**Interpretation of the asymmetry ratio:**

| $r$ value | Interpretation |
|---|---|
| $r \approx 1.0$ | The leak bypasses the dynamics entirely — a **wiring-level** short circuit (e.g., direct attention to future tokens through a masking bug). The factual↔counterfactual path is equally easy in both directions because it doesn't go through the potential landscape. |
| $r \approx 1.35$–$1.40$ | The leak propagates **through the model's normal dynamical pathway** (e.g., reverse channel). The asymmetry matches the architecture's measured Frobenius asymmetry ratio (Diagnostic Battery Arm 5), indicating the leaked information travels along the same force-field trajectories as legitimate semantic content. |
| $r \gg 1.4$ or $r \ll 1.0$ | Anomalous — the leak follows a pathway with abnormal directional preference. This would indicate a new, previously uncharacterised leak mechanism distinct from both wiring bugs and reverse-channel leaks. |

<p align="center"><img src="images/scaf_asymmetric_geodesic_leak_pathway.png" alt="A potential energy bowl showing a short, direct forward geodesic path from the factual to the counterfactual hidden state going downhill with the damping, versus a long, winding backward path going uphill against the damping. Below, a gauge bar maps the asymmetry ratio r to three diagnoses: wiring bypass near r equals 1.0, dynamical pathway near r equals 1.35 to 1.40, and anomalous outside that range." width="700"></p>

*Figure 2. Asymmetric geodesic leak distance as a pathway diagnostic. Because the damped geodesic equation is not time-reversible, the forward path (factual to counterfactual, "with the flow") is systematically shorter than the backward path (counterfactual to factual, "against the flow"). The ratio between them is not just a number — it identifies the leak's mechanism: a ratio near 1.0 means the leak bypasses the dynamics entirely (a wiring bug), while a ratio matching the architecture's measured asymmetry of 1.35-1.40 means the leak rides the model's own force field (the reverse channel).*

**Connection to Tversky's asymmetry:** the asymmetry ratio $r$ applied to the leak has the same structure as $d_{\text{geo}}(\text{specific} \to \text{general}) < d_{\text{geo}}(\text{general} \to \text{specific})$ from §18 of the paper. A leak with $r > 1$ flows "with the natural generalisation direction" (the future perturbation pushes the past state toward a broader basin), while $r < 1$ means the leak forces the past state toward a narrower, more specific basin — a qualitatively different semantic corruption.

---

## 4. SCAF integration architecture

<p align="center"><img src="images/scaf_geometric_audit_tiers_architecture.png" alt="A pipeline diagram: a factual versus counterfactual input pair flows into forward_with_trajectory, which produces per-layer hidden states. These feed three nested tiers, from largest and cheapest to smallest and most expensive: Tier A cosine deviation, Tier B basin membership, and Tier C geodesic asymmetry, forming a refinement hierarchy where Tier A is a superset of Tier B, which is a superset of Tier C. The output feeds LeakFrame columns and CATE analysis." width="700"></p>

*Figure 3. The three tiers as a SCAF pipeline. Every candidate pair of hidden-state trajectories passes through the cheap, conformally invariant Tier A check first. A subset that shows basin-crossing behaviour is escalated to Tier B, and only the most deviant pairs identified there are escalated further to the expensive Tier C geodesic integration. This mirrors the existing SCAF philosophy of cheap monitoring plus expensive post-hoc diagnosis.*

### 4.1 New adapter capability: hidden-state access

Today `ModelAdapter.forward_logits()` returns `(B, T, V)` logits. The geometric probes need per-layer hidden states. Proposed addition:

```python
class Capabilities:
    ...
    has_hidden_states: bool = False
    has_vtheta_wells: bool = False

class ModelAdapter(ABC):
    ...
    def forward_with_trajectory(
        self, model: nn.Module, x: Tensor
    ) -> tuple[Tensor, list[Tensor]]:
        """Return (logits, [h_0, h_1, ..., h_L])."""
        raise NotImplementedError

    def well_parameters(
        self, model: nn.Module, layer_idx: int
    ) -> dict | None:
        """Return {'mu': (K,d), 'precision': (K,d,d), 'weights': (K,)} or None."""
        return None
```

For the `FockAdapter`, `forward_with_trajectory` calls `model._embed` + `model._stack_forward(..., return_trajectory=True)`, which the Fock-PARFLM model already supports. `well_parameters` extracts from `AnisotropicDepthConditionedGaussianVTheta` (the `_components` method + `set_active_layer`).

### 4.2 New probes

#### `HiddenStateLeakProbe` (Tier A)

Runs the same future-perturbation protocol as `FuturePerturbationProbe` but captures per-layer hidden states and computes $\Delta_{\cos}^{(\ell)}(t)$. Reports:

- `max_delta_cos`: worst cosine deviation across all layers and positions (analogue of `linf`)
- `per_layer_delta_cos`: list of per-layer worst-case $\Delta_{\cos}^{(\ell)}$
- `latent_leak`: `True` if `max_delta_cos > 0` but logit L∞ $= 0$ (hidden-state corruption without logit-level manifestation)

Verdict contribution: `latent_leak = True` raises a `WARNING` (not `LEAK`) since the output is still causal, but it flags a fragile invariant that could break under fine-tuning or quantisation.

#### `BasinMembershipProbe` (Tier B)

Requires `has_vtheta_wells` capability. For each (factual, counterfactual) pair:

1. Extract per-layer hidden states via `forward_with_trajectory`
2. Compute well parameters via `well_parameters(model, ell)` for each layer
3. Assign dominant well $k^{\ast}$ for factual and counterfactual hidden states
4. Record $\beta^{(\ell)}(t)$ for each position $t \le t_p$ and layer $\ell$

Reports:

- `basin_crossing_rate`: $\bar\beta$ (fraction of (layer, position) pairs with changed well assignment)
- `per_layer_crossing_rate`: list of per-layer $\bar\beta^{(\ell)}$
- `worst_layer`: the layer with highest crossing rate (likely the reverse-channel injection point)

#### `GeodesicLeakProbe` (Tier C)

Requires `has_vtheta_wells` capability (for analytical Christoffel symbols). Only runs on the top-K most deviant (layer, position) pairs identified by `HiddenStateLeakProbe` (the shooting-method integration is too expensive for all pairs).

Reports:

- `mean_d_forward`, `mean_d_backward`: average geodesic distance in each direction
- `asymmetry_ratio`: $\bar r = \overline{d_\to / d_\leftarrow}$
- `leak_pathway`: `"wiring"` if $\bar r \approx 1.0$, `"dynamical"` if $\bar r \approx 1.35$–$1.40$, `"anomalous"` otherwise

### 4.3 New LeakFrame columns

The `build_leak_frame()` function gains optional geometric columns when hidden states are available:

| Column | Type | Source |
|---|---|---|
| `hidden_cos_dev` | float | $\Delta_{\cos}^{(\ell)}(t)$ — cosine deviation at (layer, position) |
| `basin_changed` | bool | $\beta^{(\ell)}(t)$ — well assignment changed |
| `layer` | int | Layer index $\ell$ (new stratification axis) |
| `well_id_factual` | int | Dominant well index under factual input |
| `well_id_counterfactual` | int | Dominant well index under counterfactual input |
| `geodesic_d_forward` | float | $d_\to^{(\ell)}(t)$ (if Tier C enabled) |
| `geodesic_d_backward` | float | $d_\leftarrow^{(\ell)}(t)$ (if Tier C enabled) |
| `geodesic_asymmetry` | float | $r^{(\ell)}(t) = d_\to / d_\leftarrow$ |

### 4.4 New CATE axes

These columns enable new causal heterogeneity analyses via `estimate_leak(cate_axes=...)`:

- **By layer:** which layers show the largest hidden-state perturbation? This reveals the leak's entry point and propagation pattern. For the known Fock reverse-channel leak, expect a spike at the layer where `reverse_channel_scale` gates the information.
- **By basin crossing:** do basin-crossing leaks have larger logit-level effects than within-basin leaks? If yes, the attractor structure amplifies the leak; if no, the leak operates independently of the potential landscape.
- **By well identity:** do specific wells (semantic attractors) show higher leak susceptibility? Some attractors may sit near the causal/acausal boundary of the register topology, making them more vulnerable to reverse-channel leaks.

---

## 5. Relationship between tiers

The three tiers form a **refinement hierarchy**:

```
Tier A (cosine)    ⊇    Tier B (basin)    ⊇    Tier C (geodesic)
    cheap,                moderate,               expensive,
    conformally           V_theta-aware,           full Riemannian,
    correct,              discrete leak            asymmetric,
    catches all           characterisation         pathway
    hidden-state                                   diagnosis
    leaks
```

- Every basin-crossing leak ($\beta = 1$) also produces a cosine deviation ($\Delta_{\cos} > 0$), but not vice versa — a small within-basin perturbation has $\Delta_{\cos} > 0$ but $\beta = 0$.
- The geodesic distance refines both by providing a metric that respects the model's potential landscape and reveals the directional structure of the leak.

**Recommended deployment:** Tier A in `LeakMonitor` (cheap enough for every eval step); Tier B in `audit()` for Gaussian $V_\theta$ models (moderate cost, high diagnostic value); Tier C as a post-hoc analysis tool for characterising detected leaks (expensive, not real-time).

---

## 6. Why logit-space metrics remain necessary

The geometric tiers **complement** rather than **replace** the existing logit-space metrics:

| Property | Logit L∞ / AILE | Geometric tiers |
|---|---|---|
| Binary causal integrity | ✅ Necessary and sufficient | ✅ (Tier A catches a superset) |
| Task-level impact (PPL) | ✅ via τ\_leak | ❌ Hidden-state deviation doesn't predict PPL impact |
| Model-agnostic (any `nn.Module`) | ✅ Only needs `forward()` | ❌ Needs `return_trajectory`, `V_theta` access |
| Cheap enough for CI/CD gating | ✅ | ❌ Tier C too expensive |
| Leak pathway diagnosis | ❌ Logits don't reveal which internal pathway | ✅ Tier C asymmetry ratio |
| Attractor-level characterisation | ❌ | ✅ Tier B basin crossing |

The existing logit-level metrics remain the **primary audit gate** — they're model-agnostic, cheap, and have a clean binary interpretation. The geometric tiers are **diagnostic instruments** that characterise the nature and mechanism of detected leaks.

---

## 7. Implementation roadmap

### Phase 1: Tier A (immediate)

1. Add `forward_with_trajectory()` to `FockAdapter` (trivial — the model already has `_stack_forward(..., return_trajectory=True)`)
2. Implement `HiddenStateLeakProbe` reusing the `make_perturbation_pairs` infrastructure
3. Add `hidden_cos_dev` and `layer` columns to `LeakFrame`
4. Add Tier A to `LeakMonitor` (single split, single layer sampled per step for cheapness)
5. Test on the known Fock reverse-channel leak: verify $\Delta_{\cos}^{(\ell)}$ profile spikes at the expected layer

### Phase 2: Tier B (with anisotropic Gaussian training runs)

1. Add `well_parameters()` to `FockAdapter` for `AnisotropicDepthConditionedGaussianVTheta`
2. Implement `BasinMembershipProbe`
3. Add `basin_changed`, `well_id_*` columns to `LeakFrame`
4. Test on the aniso-Gaussian d384 and d768 checkpoints: verify that causal models show $\bar\beta = 0$ and the known leaked checkpoint shows $\bar\beta > 0$
5. Run CATE by `basin_changed` to quantify the attractor-mediated amplification hypothesis

### Phase 3: Tier C (research, not production)

1. Implement shooting-method geodesic integration using `analytical_grad` from the Gaussian $V_\theta$ classes
2. Add `GeodesicLeakProbe` with adaptive step control (the integration must handle the full-rank conformal factor, not just the Euclidean straight line)
3. Compute asymmetry ratios on the known Fock leak and on the `prefix_causal_registers=True` fixed model
4. Compare the measured leak asymmetry ratio to the architecture's Frobenius asymmetry ratio (Arm 5 of the Diagnostic Battery) — if they match, the leak travels through the normal dynamical pathway

### Phase 4: Native retrieval diagnostic (future)

Once Tiers A–C are validated, the basin-membership infrastructure enables a retrieval-like diagnostic: given a query hidden state, find which attractor basin captures it under the model's own dynamics, and check whether this assignment is causally robust (invariant under future perturbation). This connects the SCAF audit to the nearest-neighbor/retrieval discussion: **a causally robust retrieval is one where the basin assignment is invariant under `do(future)` — the same concept that SCAF already tests, but measured in the model's native geometry rather than in logit space.**

---

## 8. Companion documents

- [Framework_for_Causal_Analysis_SemSimula_Models.md](Framework_for_Causal_Analysis_SemSimula_Models.md) — SCAF design and causal formalism
- [Exploiting_the_Riemannian_geometry_of_conservative_language_models.md](https://github.com/dimitarpg13/semsimula-paper/blob/main/companion_notes/Exploiting_the_Riemannian_geometry_of_conservative_language_models.md) — Jacobi metric, Diagnostic Battery, conformal structure
- [Geodesic_Preservation_Experiment.md](https://github.com/dimitarpg13/semsimula-paper/blob/main/companion_notes/Geodesic_Preservation_Experiment.md) — geodesic residual and $\gamma_{\text{geo}}$ measurement
- [Fock_Mechanism_Engagement_MLP_vs_Gaussian_VTheta.md](https://github.com/dimitarpg13/semsimula-paper/blob/main/companion_notes/Fock_Mechanism_Engagement_MLP_vs_Gaussian_VTheta.md) — anisotropic wells, basin structure
- [Determining_optimal_gamma_for_Fock-PARFLM.md](https://github.com/dimitarpg13/semsimula-paper/blob/main/companion_notes/Determining_optimal_gamma_for_Fock-PARFLM.md) — damping regime and asymmetry
- [Position_Dependent_Damping_and_Reinforcement_Field.md](https://github.com/dimitarpg13/semsimula-paper/blob/main/companion_notes/Position_Dependent_Damping_and_Reinforcement_Field.md) — position-dependent $\gamma(h)$ and reinforcement field
- [Fock-PARFLM_Causal_Leak_Audit_Results.md](https://github.com/dimitarpg13/semsimula-paper/blob/main/companion_notes/Fock-PARFLM_Causal_Leak_Audit_Results.md) — the known leak that SCAF generalises
