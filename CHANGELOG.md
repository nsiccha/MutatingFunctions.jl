# Changelog

## 0.1.1

- `apply!!` is now as type stable as the call it makes. Every method that passes
  a function or a variable number of arguments on (`apply!!(nothing, f, args...)`,
  the generic `apply!!(cache, f, args...)` fallback, and the registered `map`,
  `broadcast` and `accumulate` forms) gives them type parameters
  (`f::F, args::Vararg{Any,N}` with `where {F,N}`), and the `nothing` and generic
  methods also have one method per arity up to 8 arguments with a type parameter
  per argument, so Julia compiles each of them, and the keyword body it forwards
  to, for the call's concrete types, including arguments that are themselves
  functions or types. Before, these were compiled for `::Function`, `::Type` and,
  past a couple of arguments, `Vararg{Any}`, so the forwarded call was dispatched
  at run time (`Core._apply_iterate`) and ReactiveKernels refused the prepared
  program under `type_stability = :error`. Results and dispatch are unchanged.

## 0.1.0

- First version: the `apply!!` registry described in the README.
