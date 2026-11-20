# MutatingFunctions.jl

Map allocating functions to their mutating, cache-filling counterparts —
[InverseFunctions.jl](https://github.com/JuliaMath/InverseFunctions.jl), but for
mutation.

```julia
apply!!(cache, f, args...; kwargs...)
```

- `cache === nothing` → the ordinary allocating call `f(args...)`.
- otherwise → reuse `cache` as preallocated storage, returning the filled `cache`.

```julia
apply!!(nothing, zeros, 3)   # zeros(3)                      (allocates)
c = Float64[]
apply!!(c, zeros, 3)         # resize!(c, 3); fill!(c, 0); c  (reuses c)
```

## Registered functions

| function         | mutating form                   | notes                                          |
| ---------------- | ------------------------------- | ---------------------------------------------- |
| `zeros` / `ones` | `fill!` cache with `zero`/`one` | eltype taken from `cache`                      |
| `fill`           | `fill!` cache                   |                                                |
| `map`            | `map!`                          | any number of collections                      |
| `sort`           | copy + `sort!`                  | no 2-buffer form; copies into `cache` first    |
| `reverse`        | copy + `reverse!`               | no 2-buffer form; copies into `cache` first    |
| `cumsum`         | `cumsum!`                       |                                                |
| `cumprod`        | `cumprod!`                      |                                                |
| `accumulate`     | `accumulate!`                   |                                                |
| `collect`        | `copyto!`                       | skips the intermediate `collect` allocation    |
| `copy`           | `copyto!`                       |                                                |
| `broadcast`      | `broadcast!`                    | sized from the broadcast shape                 |
| `getindex`       | elementwise gather              | `A[idx]` and `A[i, idx]`; integer or `Bool` mask; scalar reads, so a `SubArray`/`ReshapedArray` source is Enzyme-safe |
| `*`              | `mul!`                          | LinearAlgebra; `cache` pre-sized               |
| `\`              | `ldiv!` against `lu(A)`         | LinearAlgebra; `cache` pre-sized               |
| `/`              | copy + `rdiv!` against `lu(B)`  | LinearAlgebra; `cache` pre-sized               |
| `rand`           | `rand!`                         | Random                                         |
| `randn`          | `randn!`                        | Random                                         |
| `quantile`       | `quantile!`                     | Statistics; **sorts** `v`, like Base           |

Unregistered functions fall back to a best-effort copy into `cache` (the
persistent buffer is reused; `f` still allocates its result); scalar results pass
straight through.

`apply!!` is as type stable as the call it makes: every method compiles for its
call's concrete types, so `apply!!(nothing, f, args...)` dispatches `f(args...)`
statically whenever `f` and the arguments are concretely typed. (Past 8
arguments, an argument that is itself a function or a type is compiled as
`::Function` or `::Type`; everything else stays concrete.)

`LinearAlgebra`, `Random` and `Statistics` are ordinary (strong) dependencies of
this package — there are **no package extensions**, so every form above is defined
as soon as `using MutatingFunctions` runs. (They were three extensions until
`771b58b`; the split was dropped because Pkg 1.10's parallel precompile
self-deadlocks on sibling extensions whose triggers are sysimage stdlibs, and on
Julia 1.10 those stdlibs are always loaded anyway.)

## Opt in

```julia
MutatingFunctions.apply!!(cache::AbstractVector, ::typeof(myfun), args::Vararg{Any,N}) where {N} =
    (resize!(cache, n); myfun!(cache, args...); cache)
```

Give the arguments a form passes on type parameters, as this package's own forms
do: `f::F` for a function, `args::Vararg{Any,N}` for a variable number, `a::A`
for an argument that may itself be a function or a type (all `where {…}`).
Written `f`, `args...` or an untyped `a`, Julia compiles the method for
`::Function`, `::Type` and, once it has more than a few arguments,
`Vararg{Any}`, and the forwarded call is then dispatched at run time.

Status: **early WIP** — no TreeArrays (or any other) dependency; standalone.

## License

MIT — see [LICENSE](LICENSE).
