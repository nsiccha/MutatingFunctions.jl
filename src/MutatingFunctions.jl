module MutatingFunctions

export forward

"""
    forward(cache, f, args...; kwargs...)

Evaluate `f(args...; kwargs...)`, optionally reusing `cache` as preallocated
storage for the result.

- `cache === nothing` → the ordinary allocating call `f(args...; kwargs...)`.
- otherwise → the *mutating* counterpart of `f`, writing the result into `cache`
  (resizing / filling in place) and returning the filled `cache`.

`forward` is to mutation what `InverseFunctions.inverse` is to inversion: a small
registry mapping an allocating function to its cache-filling form. Register a new
function by adding a `forward(cache::AbstractVector, ::typeof(f), args...)` method.
Functions without a registered form fall back to a best-effort copy into `cache`
(the persistent buffer is reused; `f` still allocates its result); scalar results
pass straight through.

# Examples
```julia
forward(nothing, zeros, 3)      # zeros(3)                      (allocates)

c = Float64[]
forward(c, zeros, 3) === c      # true — resize!(c, 3); fill!(c, 0); return c
```
"""
forward(::Nothing, f, args...; kwargs...) = f(args...; kwargs...)

# Generic fallback for a non-`nothing` cache and an unregistered function: reuse
# `cache` as storage where the shapes line up, otherwise return the result as-is.
forward(cache, f, args...; kwargs...) = _store!(cache, f(args...; kwargs...))

_store!(cache::AbstractVector, r::AbstractVector) =
    (resize!(cache, length(r)); copyto!(cache, r); cache)
_store!(_, r) = r   # scalar / shape-mismatched result → passthrough

# --- registered mutating forms (avoid the allocation entirely) ----------------
forward(cache::AbstractVector, ::typeof(zeros), n::Integer) =
    (resize!(cache, n); fill!(cache, zero(eltype(cache))); cache)
forward(cache::AbstractVector, ::typeof(ones), n::Integer) =
    (resize!(cache, n); fill!(cache, one(eltype(cache))); cache)
forward(cache::AbstractVector, ::typeof(fill), v, n::Integer) =
    (resize!(cache, n); fill!(cache, v); cache)
forward(cache::AbstractVector, ::typeof(map), f, cs...) =
    (resize!(cache, length(first(cs))); map!(f, cache, cs...); cache)

end # module
