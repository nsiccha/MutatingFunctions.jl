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

# `broadcast` materializes straight into `cache`, sized from the broadcast shape.
apply!!(cache::AbstractVector, ::typeof(broadcast), f, args...) =
    (resize!(cache, length(Broadcast.combine_axes(args...)[1]));
     broadcast!(f, cache, args...); cache)

end # module
