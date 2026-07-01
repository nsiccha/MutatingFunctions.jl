# MutatingFunctions.jl

Map allocating functions to their mutating, cache-filling counterparts —
[InverseFunctions.jl](https://github.com/JuliaMath/InverseFunctions.jl), but for
mutation.

```julia
forward(cache, f, args...; kwargs...)
```

- `cache === nothing` → the ordinary allocating call `f(args...)`.
- otherwise → reuse `cache` as preallocated storage, returning the filled `cache`.

```julia
forward(nothing, zeros, 3)   # zeros(3)                      (allocates)
c = Float64[]
forward(c, zeros, 3)         # resize!(c, 3); fill!(c, 0); c  (reuses c)
```

Functions opt in via `forward(cache, ::typeof(f), args...)` methods; a starter
set covers `zeros` / `map` / `sum` / `mean` / `quantile` / …. Status: **early WIP.**
