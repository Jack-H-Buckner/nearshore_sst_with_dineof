# eDINEOF — implementation notes

DINEOF (Data Interpolating Empirical Orthogonal Functions) fills gaps in a
space–time field by repeatedly fitting a truncated SVD and using the low-rank
fit as the estimate for the missing entries. It is an EM algorithm for a
low-rank model, with the rank chosen by cross-validation.

eDINEOF adds one thing: a smoothing filter applied to the temporal covariance
matrix before the eigendecomposition. Everything else is identical.

References: Beckers & Rixen (2003) JAOT 20:1839; Alvera-Azcárate et al. (2005)
Ocean Modelling 9:325; Alvera-Azcárate et al. (2009) Ocean Science 5:475 (the
filter).

---

## 1. Notation

`X` is the data matrix, shape `(m, n)`:

- `m` = number of spatial points (land/permanently-invalid cells dropped)
- `n` = number of time steps (images)
- column `j` = image `j`, flattened
- row `i` = the time series at pixel `i`

The factorization, truncated at `k` modes:

```
X ≈ U @ diag(sigma) @ V.T
```

| object  | shape    | one column is        | name               |
|---------|----------|----------------------|--------------------|
| `U`     | `(m, k)` | a map (length `m`)   | spatial modes/EOFs |
| `sigma` | `(k,)`   | —                    | singular values    |
| `V`     | `(n, k)` | a time series (`n`)  | temporal modes/PCs |

Mode `j` is the outer product `sigma[j] * outer(U[:,j], V[:,j])`: a map times
how strongly that map is present on each day.

`B = X.T @ X`, shape `(n, n)`. `B[i,j]` is the dot product of image `i` with
image `j`, i.e. how similar those two days look. Its eigenvectors are `V` and
its eigenvalues are `sigma**2`, which is why the temporal modes can be obtained
without ever forming an `(m, m)` matrix.

Keep `m >= n`. If you have fewer pixels than images, transpose the problem and
swap the roles of `U` and `V` at the end. The filter assumes the time axis is
the `n` axis.

---

## 2. Preprocessing

**Centering.** Subtract a single scalar: the mean of all valid entries over all
pixels and all times. Not a per-pixel mean, not a per-image mean.

```
mu = mean(X_raw[observed])
X  = X_raw - mu
```

Missing entries are then initialized to `0`, which is the mean — an unbiased
first guess that adds no variance. Add `mu` back at the very end.

If your field has a strong seasonal cycle, consider removing a per-pixel
climatology yourself beforehand and adding it back afterwards, so the leading
modes describe variability rather than the mean state. Fit a smooth harmonic
rather than a raw average of available days, and mask pixels with too few
valid observations. This is preprocessing, outside the algorithm.

**Non-Gaussian variables** (chlorophyll, suspended matter) should be
log-transformed before this step. Temperature needs no transform.

**Representation.** Carry a boolean mask plus a values array rather than NaNs.
You will index by mask constantly.

```
X        : (m, n) float64
observed : (m, n) bool     # True where real data exists
```

Use float64. Forming `B` squares the condition number, so relative error in
`sigma[j]` grows like `eps * (sigma[0]/sigma[j])**2`. In float32 the trailing
modes of a run retaining 40+ modes are computed to only a few digits.

---

## 3. The temporal filter

A forward-Euler diffusion (Laplacian) operator on unevenly spaced samples.
Dividing differences by the actual time increment is the point: two columns
adjacent in the matrix but three weeks apart in reality get coupled weakly.

```
function filter_time(t, M, alpha, p):
    # t     : (n,) sample times, may be irregular
    # M     : (..., n) array; the filter acts along the LAST axis
    # alpha : filter strength, units of time**2
    # p     : number of iterations

    dt = diff(t)                                  # (n-1,)

    # staggered time coordinates, with ghost points at both ends
    tmid          = empty(n+1)
    tmid[1:n]     = (t[:-1] + t[1:]) / 2
    tmid[0]       = t[0]  - dt[0]  / 2
    tmid[n]       = t[-1] + dt[-1] / 2
    dtmid         = diff(tmid)                    # (n,)

    for _ in range(p):
        G            = zeros(M.shape[:-1] + (n+1,))
        G[..., 1:n]  = alpha * diff(M, axis=-1) / dt   # flux between samples
        # G[..., 0] and G[..., n] stay 0: no flux off either end
        M            = M + diff(G, axis=-1) / dtmid

    return M
```

Vectorize over the leading axes so you filter the whole matrix in one call
rather than looping over rows.

**Stability:** `alpha <= min(dt)**2 / 2`. Violate it and the diffusion blows up.

**Cutoff:** variability faster than `2*pi*sqrt(alpha*p)` is suppressed; the
filter reaches `2p+1` samples. Note the diffusive scaling — quadrupling `alpha*p`
only doubles the cutoff period. Fix `alpha` just under the stability ceiling
and tune `p` by cross-validation.

Reference values for daily data: `alpha = 0.01` (days²), `p = 3`, giving a
cutoff around 1.1 days.

### Applying it to B

The paper writes `B~ = F.T @ B @ F`, where `F` is the `(n, n)` matrix form of
the filter. Never build `F`. Apply `filter_time` along both axes:

```
function filter_covariance(t, B, alpha, p):
    B = filter_time(t, B, alpha, p)          # filters rows
    B = filter_time(t, B.T, alpha, p).T      # filters columns
    B = (B + B.T) / 2                        # kill roundoff asymmetry
    return B
```

The symmetrization matters: the eigensolver requires a symmetric matrix and
two floating-point passes will not produce one exactly.

**What this actually does.** Row `i` of `B` is the vector "how much image `i`
resembles image 1, 2, ... n", indexed by time. Smoothing it asserts that image
`i`'s resemblance to image `j` should vary smoothly as `j` walks through the
calendar. An image with almost no valid data has a nearly empty row and column
in `B`; after filtering it inherits structure from its temporal neighbours, so
its amplitude is determined partly by adjacent days rather than entirely by its
handful of surviving pixels. That is the whole mechanism.

---

## 4. Mode extraction

```
function top_k_modes(X, k, t, alpha, p, use_filter):
    B = X.T @ X
    if use_filter:
        B = filter_covariance(t, B, alpha, p)
    else:
        B = (B + B.T) / 2

    lam, V = eigsh(B, k=k, which='LA')       # scipy.sparse.linalg
    lam, V = lam[::-1], V[:, ::-1]           # eigsh returns ascending
    lam    = maximum(lam, 0)                 # guard tiny negatives

    W = X @ V                                # (m, k)

    # --- see note below ---
    sigma = norm(W, axis=0)                  # projection convention
    U     = W / sigma

    return U, sigma, V
```

The temporal modes come from the *filtered* covariance. `U` comes from
projecting the *unfiltered* `X` onto them. The filter constrains when things
happen, never what the spatial patterns look like.

### The sigma convention

Two defensible choices, identical when the filter is off:

- **Projection (above):** `sigma = norm(X @ V, axis=0)`. Then
  `U @ diag(sigma) @ V.T == X @ V @ V.T`, the orthogonal projection of `X` onto
  `span(V)`. Clean and least-squares optimal given the basis.
- **Reference:** `sigma = sqrt(lam)`, with `U` separately normalized to unit
  columns. This is what the Fortran does. Since diffusion reduces variance,
  `sqrt(lam) <= norm(X @ V)`, so mode amplitudes are systematically damped.

Use the projection convention for new code; expose a flag if you need to
reproduce published results. Compare the two by cross-validation on your data.

### A consequence worth knowing

With the filter on, `U.T @ U` is close to but not equal to the identity, and
drifts further as `alpha*p` grows. Orthogonality of the spatial modes follows
from `V` diagonalizing `X.T @ X`, and with the filter `V` diagonalizes
`F.T @ X.T @ X @ F` instead. So the spatial modes are unit-norm but not
orthogonal, mode variances do not add up cleanly, and "EOF" is a slight
misnomer on the spatial side. This does not affect the reconstruction.

---

## 5. The inner iteration

For a fixed `k`, alternate between fitting modes and refilling gaps until the
filled values stop changing.

```
function fill(X, gaps, k, t, alpha, p, tol, max_iter, sd):
    prev = X[gaps].copy()
    for it in range(max_iter):
        U, sigma, V = top_k_modes(X, k, t, alpha, p, use_filter=True)
        X[gaps]     = reconstruct_at(U, sigma, V, gaps)
        delta       = rms(X[gaps] - prev) / sd
        prev        = X[gaps].copy()
        if delta < tol:
            break
    return X
```

`sd` is the standard deviation of the observed data, computed once. `tol = 1e-3`
is the reference default; `max_iter` around 100 is plenty.

Observed entries are held fixed and never updated. There is no
observation-error weighting: a pixel is either present with weight 1 or absent
with weight 0.

`reconstruct_at` should evaluate only at the gap locations rather than forming
the full `(m, n)` product:

```
function reconstruct_at(U, sigma, V, gaps):
    rows, cols = nonzero(gaps)
    return einsum('ik,k,ik->i', U[rows], sigma, V[cols])
```

---

## 6. Cross-validation and mode selection

Set aside a small fraction of valid points, treat them as missing, and score
the reconstruction against their known values.

**How to choose them matters.** Uniformly random pixels are surrounded by data
and therefore much easier to reconstruct than real gaps, which are contiguous
clouds. This makes random-pixel CV optimistic. Prefer taking an actual gap
mask from a different image in the series and pasting it onto the current one,
so the held-out geometry matches the geometry you actually need to fill.

Typical fraction: 1–3% of valid data.

The mode search walks `k` upward, warm-starting each `k` from the previous
one's filled values, and stops once the CV error has risen on three consecutive
values of `k`.

---

## 7. Full algorithm

```
function edineof(X_raw, observed, t, alpha, p,
                 k_max=50, cv_frac=0.02, tol=1e-3, seed=0):

    # --- centering ---
    mu = mean(X_raw[observed])
    X  = where(observed, X_raw - mu, 0.0)

    # --- cross-validation set ---
    CV       = choose_cv_points(observed, cv_frac, seed)   # subset of observed
    cv_truth = X[CV].copy()
    gaps     = (~observed) | CV
    X[gaps]  = 0.0
    sd       = std(X[observed & ~CV])

    # --- mode search ---
    errs      = []
    best      = (inf, None, None)          # (err, k, filled values)
    for k in 1 .. k_max:
        X = fill(X, gaps, k, t, alpha, p, tol, max_iter, sd)   # warm start

        err = rms(X[CV] - cv_truth)
        errs.append(err)
        if err < best.err:
            best = (err, k, X[gaps].copy())

        if len(errs) >= 4 and errs[-1] > errs[-2] > errs[-3] > errs[-4]:
            break

    # --- final pass: give the held-out points back, refit at k* ---
    k_opt   = best.k
    X[gaps] = best.filled
    X[CV]   = cv_truth
    gaps    = ~observed
    X       = fill(X, gaps, k_opt, t, alpha, p, tol, max_iter, sd)

    U, sigma, V = top_k_modes(X, k_opt, t, alpha, p, use_filter=True)

    return X + mu, U, sigma, V, k_opt, best.err
```

Returning `(U, sigma, V)` alongside the field is worth doing: the truncated
basis is what you need for error maps and for residual-based outlier tests.

---

## 8. Parameters

| name      | meaning                              | typical           |
|-----------|--------------------------------------|-------------------|
| `alpha`   | filter strength, units time²         | `0.01` (daily)    |
| `p`       | filter iterations                    | `3`               |
| `k_max`   | max modes to try                     | `50`              |
| `cv_frac` | fraction held out                    | `0.01`–`0.03`     |
| `tol`     | convergence, relative to `sd`        | `1e-3`            |
| `t`       | sample times, length `n`             | actual dates      |

Set `alpha = 0` or `p = 0` to recover plain DINEOF.

---

## 9. Implementation notes

**Eigensolver.** `scipy.sparse.linalg.eigsh(B, k=k, which='LA')`. It returns
ascending eigenvalues, so flip. Pass `v0` from the previous iteration's leading
eigenvector to speed up convergence across the EM loop.

**Cost.** Forming `B` is `O(m n**2)` and happens inside the inner iteration,
inside the mode loop. For daily data over a region (`m` in the tens of
thousands, `n` in the hundreds) this is fine. For high-frequency data where
`n` runs to tens of thousands, `B` alone becomes multi-gigabyte; at that point
go matrix-free by handing `eigsh` a `LinearOperator` implementing
`v -> F.T @ (X.T @ (X @ (F @ v)))`, where `F @ v` is a `filter_time` call.
The adjoint is not identical to the filter on irregular grids — the non-uniform
Laplacian is self-adjoint under the cell-width-weighted inner product, not the
plain Euclidean one — so derive the discrete adjoint or the solver will
complain about asymmetry.

**Sign convention.** Eigenvector signs are arbitrary and flip between runs.
`U` flips with `V` so the reconstruction is invariant, but fix a convention
(e.g. force the largest-magnitude entry of each `V[:,j]` positive) if you plan
to compare modes across runs.

**Near-degenerate eigenvalues.** When `lam[j] ≈ lam[j+1]` only the plane they
span is well determined, not the individual vectors. Common for propagating
features, which appear as a quadrature pair. The reconstruction is fine;
interpreting either mode alone is not.

---

## 10. Correctness checks

Build these as tests before trusting anything:

1. **Filter off, no gaps.** With `alpha=0` and `k = min(m,n)`,
   `U @ diag(sigma) @ V.T` should reproduce `X` to machine precision. Compare
   `sigma` against `numpy.linalg.svd(X, compute_uv=False)`.
2. **Filter off, orthogonality.** `U.T @ U` and `V.T @ V` should both be the
   identity to ~1e-12. With the filter on, `V.T @ V` stays identity and
   `U.T @ U` does not — that is expected, not a bug.
3. **Filter off, eigenvalues.** `sigma**2` should match the eigenvalues of
   `X.T @ X` under both sigma conventions.
4. **Filter symmetry.** `filter_covariance` output must satisfy
   `allclose(B, B.T)`.4
5. **Filter on a constant.** `filter_time` applied to a constant vector must
   return it unchanged (zero-flux boundaries; no mass leaks).
6. **Filter stability.** Assert `alpha <= min(diff(t))**2 / 2` at entry and
   raise otherwise.
7. **Synthetic recovery.** Build `X = U @ diag(s) @ V.T` from known rank-5
   factors, punch out 40% of the entries at random, and confirm the
   reconstruction recovers them and that CV selects `k = 5`.
8. **Monotone convergence.** Log `delta` in the inner loop; it should decrease
   monotonically. If it oscillates, `alpha` is likely over the stability limit.