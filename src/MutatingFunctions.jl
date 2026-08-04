module MutatingFunctions

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

# Examples
```julia
apply!!(nothing, zeros, 3)      # zeros(3)                      (allocates)

c = Float64[]
apply!!(c, zeros, 3) === c      # true — resize!(c, 3); fill!(c, 0); return c
```
"""
apply!!(::Nothing, f, args...; kwargs...) = f(args...; kwargs...)

# Generic fallback for a non-`nothing` cache and an unregistered function: reuse
# `cache` as storage where the shapes line up, otherwise return the result as-is.
apply!!(cache, f, args...; kwargs...) = _store!(cache, f(args...; kwargs...))

_store!(cache::AbstractVector, r::AbstractVector) =
    (resize!(cache, length(r)); copyto!(cache, r); cache)
_store!(_, r) = r   # scalar / shape-mismatched result → passthrough

# --- registered mutating forms (avoid the allocation entirely) ----------------
apply!!(cache::AbstractVector, ::typeof(zeros), n::Integer) =
    (resize!(cache, n); fill!(cache, zero(eltype(cache))); cache)
apply!!(cache::AbstractVector, ::typeof(ones), n::Integer) =
    (resize!(cache, n); fill!(cache, one(eltype(cache))); cache)
apply!!(cache::AbstractVector, ::typeof(fill), v, n::Integer) =
    (resize!(cache, n); fill!(cache, v); cache)
apply!!(cache::AbstractVector, ::typeof(map), f, cs...) =
    (resize!(cache, length(first(cs))); map!(f, cache, cs...); cache)

# `sort`/`reverse` have no direct 2-buffer mutating form — copy into `cache`,
# then mutate `cache` in place with the `!` counterpart.
apply!!(cache::AbstractVector, ::typeof(sort), A; kwargs...) =
    (resize!(cache, length(A)); copyto!(cache, A); sort!(cache; kwargs...); cache)
apply!!(cache::AbstractVector, ::typeof(reverse), A) =
    (resize!(cache, length(A)); copyto!(cache, A); reverse!(cache); cache)

# `cumsum`/`cumprod`/`accumulate` have a genuine 2-buffer mutating form — no copy needed.
apply!!(cache::AbstractVector, ::typeof(cumsum), A; kwargs...) =
    (resize!(cache, length(A)); cumsum!(cache, A; kwargs...); cache)
apply!!(cache::AbstractVector, ::typeof(cumprod), A; kwargs...) =
    (resize!(cache, length(A)); cumprod!(cache, A; kwargs...); cache)
apply!!(cache::AbstractVector, ::typeof(accumulate), op, A; kwargs...) =
    (resize!(cache, length(A)); accumulate!(op, cache, A; kwargs...); cache)

# `collect`/`copy` skip the intermediate allocation the generic fallback would
# otherwise pay for (calling `f` first, then copying its result into `cache`).
apply!!(cache::AbstractVector, ::typeof(collect), itr) =
    (resize!(cache, length(itr)); copyto!(cache, itr); cache)
apply!!(cache::AbstractVector, ::typeof(copy), A) =
    (resize!(cache, length(A)); copyto!(cache, A); cache)

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
    (n = _broadcast_length(args...); length(cache) == n || resize!(cache, n);
     broadcast!(f, cache, args...); cache)
# 1-D broadcast length = the longest array argument (scalars broadcast to it);
# an isbits fold, so it allocates nothing.
@inline _broadcast_length(args...) = _broadcast_length(1, args...)
@inline _broadcast_length(n::Int) = n
@inline _broadcast_length(n::Int, a, rest...) =
    _broadcast_length(a isa AbstractArray ? max(n, length(a)) : n, rest...)

# `getindex` gather (`A[idx]`, e.g. the `b[group]` gather ubiquitous in
# hierarchical models). `@view A[idx]` is a non-copying gathered view, so
# `copyto!` into the pre-sized `cache` skips the intermediate the generic
# fallback would allocate. Size from the view (correct for integer *and* mask
# indices) and keep the resize-to-fit convenience of the other vector forms.
apply!!(cache::AbstractVector, ::typeof(getindex), A, idx::AbstractVector) =
    (v = @view(A[idx]); resize!(cache, length(v)); copyto!(cache, v); cache)

end # module
