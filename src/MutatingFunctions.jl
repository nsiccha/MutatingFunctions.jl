module MutatingFunctions

import LinearAlgebra: AbstractVecOrMat, mul!, ldiv!, rdiv!, lu
import Random: rand, randn, rand!, randn!
import Statistics

export apply!!

"""
    apply!!(cache, f, args...; kwargs...)

Evaluate `f(args...; kwargs...)`, optionally reusing `cache` as preallocated
storage for the result.

- `cache === nothing` → the ordinary allocating call `f(args...; kwargs...)`.
- otherwise → the *mutating* counterpart of `f`, writing the result into `cache`
  (resizing / filling in place) and returning the filled `cache`.

`apply!!` is to mutation what `InverseFunctions.inverse` is to inversion: a small
registry mapping an allocating function to its cache-filling form. Register a new
function by adding an `apply!!(cache::AbstractVector, ::typeof(f), args...)` method.
Functions without a registered form fall back to a best-effort copy into `cache`
(the persistent buffer is reused; `f` still allocates its result); scalar results
pass straight through.

A destination of the correct length that has no `resize!` method — a `SubArray`
(`@view`) backing a single-flat buffer pool — is filled in place: every form
grows `cache` only when its length does not already match (see `_resize!`).

# Examples
```julia
apply!!(nothing, zeros, 3)      # zeros(3)                      (allocates)

c = Float64[]
apply!!(c, zeros, 3) === c      # true — resize!(c, 3); fill!(c, 0); return c

flat = zeros(10)
apply!!(@view(flat[1:3]), zeros, 3)   # fills the view in place — no resize! needed
```
"""
apply!!(::Nothing, f, args...; kwargs...) = f(args...; kwargs...)

# Generic fallback for a non-`nothing` cache and an unregistered function: reuse
# `cache` as storage where the shapes line up, otherwise return the result as-is.
apply!!(cache, f, args...; kwargs...) = _store!(cache, f(args...; kwargs...))

# Grow `cache` to length `n` ONLY when it is not already that length. A growable
# `Vector` still grows to fit (and `resize!(v, length(v))` was already a no-op, so
# guarding it changes nothing for the existing grow-to-fit callers), while an
# already-correctly-sized destination with no `resize!` method — a `SubArray`
# (`@view`) into a single flat backing array, the load-bearing buffer-pool case —
# is filled in place instead of throwing `MethodError: no method matching
# resize!(::SubArray, ::Int)`. This is the guard the `broadcast` vector form has
# used since it was added; every resize-to-fit form now shares it.
@inline _resize!(cache, n) = (length(cache) == n || resize!(cache, n); cache)

_store!(cache::AbstractVector, r::AbstractVector) =
    (_resize!(cache, length(r)); copyto!(cache, r); cache)
_store!(_, r) = r   # scalar / shape-mismatched result → passthrough

# --- registered mutating forms (avoid the allocation entirely) ----------------
apply!!(cache::AbstractVector, ::typeof(zeros), n::Integer) =
    (_resize!(cache, n); fill!(cache, zero(eltype(cache))); cache)
apply!!(cache::AbstractVector, ::typeof(ones), n::Integer) =
    (_resize!(cache, n); fill!(cache, one(eltype(cache))); cache)
apply!!(cache::AbstractVector, ::typeof(fill), v, n::Integer) =
    (_resize!(cache, n); fill!(cache, v); cache)
apply!!(cache::AbstractVector, ::typeof(map), f, cs...) =
    (_resize!(cache, length(first(cs))); map!(f, cache, cs...); cache)

# `sort`/`reverse` have no direct 2-buffer mutating form — copy into `cache`,
# then mutate `cache` in place with the `!` counterpart.
apply!!(cache::AbstractVector, ::typeof(sort), A; kwargs...) =
    (_resize!(cache, length(A)); copyto!(cache, A); sort!(cache; kwargs...); cache)
apply!!(cache::AbstractVector, ::typeof(reverse), A) =
    (_resize!(cache, length(A)); copyto!(cache, A); reverse!(cache); cache)

# `cumsum`/`cumprod`/`accumulate` have a genuine 2-buffer mutating form — no copy needed.
apply!!(cache::AbstractVector, ::typeof(cumsum), A; kwargs...) =
    (_resize!(cache, length(A)); cumsum!(cache, A; kwargs...); cache)
apply!!(cache::AbstractVector, ::typeof(cumprod), A; kwargs...) =
    (_resize!(cache, length(A)); cumprod!(cache, A; kwargs...); cache)
apply!!(cache::AbstractVector, ::typeof(accumulate), op, A; kwargs...) =
    (_resize!(cache, length(A)); accumulate!(op, cache, A; kwargs...); cache)

# `collect`/`copy` skip the intermediate allocation the generic fallback would
# otherwise pay for (calling `f` first, then copying its result into `cache`).
apply!!(cache::AbstractVector, ::typeof(collect), itr) =
    (_resize!(cache, length(itr)); copyto!(cache, itr); cache)
apply!!(cache::AbstractVector, ::typeof(copy), A) =
    (_resize!(cache, length(A)); copyto!(cache, A); cache)

# `broadcast` materializes straight into `cache`. `broadcast!` writes in place
# and validates the shape itself, so a correctly-sized `cache` of *any* rank
# (Matrix / N-D, not just Vector) is filled with zero allocation.
apply!!(cache::AbstractArray, ::typeof(broadcast), f, args...) =
    (broadcast!(f, cache, args...); cache)
# Vector keeps the resize-to-fit convenience (an empty/mis-sized buffer grows
# to match), but sizes the buffer from the argument lengths directly rather than
# materializing `Broadcast.combine_axes` — the latter allocates for ≥3-arg and
# scalar-containing broadcasts, so skipping it keeps the pre-sized hot path (a
# no-op `resize!`) allocation-free. `broadcast!` still validates compatibility.
apply!!(cache::AbstractVector, ::typeof(broadcast), f, args...) =
    (_resize!(cache, _broadcast_length(args...)); broadcast!(f, cache, args...); cache)
# 1-D broadcast length = the longest array argument (scalars broadcast to it);
# an isbits fold, so it allocates nothing.
@inline _broadcast_length(args...) = _broadcast_length(1, args...)
@inline _broadcast_length(n::Int) = n
@inline _broadcast_length(n::Int, a, rest...) =
    _broadcast_length(a isa AbstractArray ? max(n, length(a)) : n, rest...)

# `getindex` gather (`A[idx]`, e.g. the `b[group]` gather ubiquitous in
# hierarchical models). Read the gather ELEMENTWISE from `A` rather than
# materializing `@view(A[idx])` + `copyto!`: when `A` is itself a `SubArray`
# (the load-bearing `b = @view θ[lo:hi]; b[group]` PPL case), the composed
# view-of-a-view folds the constant `idx` arithmetic into the differentiable
# store, and plain `AutoEnzyme`'s static activity analysis then throws
# `EnzymeRuntimeActivityError`. The scalar read keeps the constant index
# arithmetic out of the differentiable store, so it differentiates like a
# native elementwise copy — and it allocates nothing (the view form built a
# `SubArray` per call). Integer indices size the cache from `length(idx)`; a
# boolean mask sizes from `count(idx)` and streams the selected elements (its
# output length is the number of set bits, NOT `length(idx)`). Both keep the
# resize-to-fit convenience of the other vector forms.
function apply!!(cache::AbstractVector, ::typeof(getindex), A, idx::AbstractVector{<:Integer})
    _resize!(cache, length(idx))
    for (i, j) in enumerate(idx)
        cache[i] = A[j]
    end
    cache
end
function apply!!(cache::AbstractVector, ::typeof(getindex), A, idx::AbstractVector{Bool})
    length(idx) == length(A) ||
        throw(DimensionMismatch("boolean gather mask has length $(length(idx)), expected $(length(A))"))
    _resize!(cache, count(idx))
    k = 0
    for (i, on) in enumerate(idx)
        on && (cache[k += 1] = A[i])
    end
    cache
end

# Multi-dimensional gather with a leading SCALAR row index into a matrix source
# (`A[i, idx]`, e.g. the `b[k, group]` per-row gather that dominates a grouped
# hierarchical model's gradient — three such gathers, `b[1,group]`/`b[2,group]`/
# `b[3,group]`, are the bulk of its allocation). Without a registered form the
# generic fallback runs `_store!(cache, getindex(A, i, idx))` — it materializes a
# fresh vector, then copies it in — so `apply!!` allocates exactly as the plain
# gather does. Read ELEMENTWISE from `A` instead, exactly as the 1-D form above:
# the scalar read `A[i, idx[k]]` keeps the constant index arithmetic out of the
# differentiable store, so a `ReshapedArray` (reshaped-pool-vector) or `SubArray`
# source differentiates like a native elementwise copy under plain `AutoEnzyme`
# rather than tripping `EnzymeRuntimeActivityError`, and it allocates nothing.
# The `{<:Integer}` / `{Bool}` split mirrors the 1-D forms: a boolean mask's
# output length is the number of set bits, and it selects along `A`'s columns, so
# it sizes from `count(idx)` and validates against `size(A, 2)`. Both grow via
# `_resize!`, so an already-sized `@view` destination (the buffer-pool case)
# fills in place instead of hitting `resize!(::SubArray, ::Int)`.
function apply!!(cache::AbstractVector, ::typeof(getindex), A::AbstractMatrix,
                 i::Integer, idx::AbstractVector{<:Integer})
    _resize!(cache, length(idx))
    for (k, j) in enumerate(idx)
        cache[k] = A[i, j]
    end
    cache
end
function apply!!(cache::AbstractVector, ::typeof(getindex), A::AbstractMatrix,
                 i::Integer, idx::AbstractVector{Bool})
    length(idx) == size(A, 2) ||
        throw(DimensionMismatch("boolean gather mask has length $(length(idx)), expected $(size(A, 2))"))
    _resize!(cache, count(idx))
    k = 0
    for (j, on) in enumerate(idx)
        on && (cache[k += 1] = A[i, j])
    end
    cache
end

# --- LinearAlgebra -----------------------------------------------------------
# A * B → mul!(cache, A, B); `cache` must already be sized to the result shape
# (mul! never resizes — the same pre-sized convention `mul!` itself expects).
apply!!(cache::AbstractVecOrMat, ::typeof(*), A, B) = (mul!(cache, A, B); cache)

# A \ B → ldiv!(cache, lu(A), B). Plain dense `A` has no 3-arg `ldiv!` of its
# own (Base's generic 3-arg form needs a `Factorization`); `lu(A)` still
# allocates the factors, but the *solution* goes straight into `cache`.
apply!!(cache::AbstractVecOrMat, ::typeof(\), A, B) =
    (ldiv!(cache, lu(A), B); cache)

# A / B → same story, rotated: copy `A` into `cache`, then `rdiv!` against `lu(B)`.
apply!!(cache::AbstractVecOrMat, ::typeof(/), A, B) =
    (copyto!(cache, A); rdiv!(cache, lu(B)); cache)

# --- Random ------------------------------------------------------------------
# rand(n) / randn(n) → resize `cache` to `n`, then fill it in place.
apply!!(cache::AbstractVector, ::typeof(rand), n::Integer) =
    (resize!(cache, n); rand!(cache); cache)
apply!!(cache::AbstractVector, ::typeof(randn), n::Integer) =
    (resize!(cache, n); randn!(cache); cache)

# --- Statistics --------------------------------------------------------------
# quantile(v, p::AbstractVector) → a vector of quantiles, written into `cache`.
# Note: `quantile!` SORTS `v` in place (the mutating contract), exactly like Base.
apply!!(cache::AbstractVector, ::typeof(Statistics.quantile), v, p::AbstractVector) =
    (resize!(cache, length(p)); Statistics.quantile!(cache, v, p); cache)

end # module
