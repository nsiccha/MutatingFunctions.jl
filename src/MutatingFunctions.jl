module MutatingFunctions

export forward

"""
    forward(cache, f, args...; kwargs...)

Evaluate `f(args...; kwargs...)`, optionally reusing `cache` as preallocated
storage for the result.

- `cache === nothing` → the ordinary allocating call `f(args...; kwargs...)`.
- otherwise → the *mutating* counterpart of `f`, writing the result into
  `cache` (resizing / filling as needed) and returning it.

This is to mutation what `InverseFunctions.inverse` is to inversion: a small
registry mapping an allocating function to its cache-filling form. A function
opts in by adding a `forward(cache, ::typeof(f), args...)` method; unregistered
functions fall back to the allocating path (or a generic copy-into-cache).
"""
forward(::Nothing, f, args...; kwargs...) = f(args...; kwargs...)

# TODO: registered mutating forms (zeros, map, sum, mean, quantile, …) plus a
# generic fallback for a non-nothing cache with an unregistered function.

end # module
