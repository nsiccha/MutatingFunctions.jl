# Changelog

## 0.1.2

- Every call no registered form covers resolves to one generic `apply!!` method
  again, as in 0.1.0, so `which(apply!!, Tuple{C, typeof(f), argtypes...})` is
  `which(apply!!, Tuple{Any,Any,Vararg{Any}})` for any unregistered `f`. 0.1.1
  added one method per arity up to 8 beside it, so a call with 1-8 arguments
  resolved to another method. ReactiveKernels tells a registered form from the
  fallback by that identity, so with 0.1.1 it kept a cache buffer for every
  unregistered step: its non-allocating AD preparation threw (`Cannot convert`
  a lazy `Broadcasted` to the type of the first result), and a step whose first
  result was a view failed later with `resize!(::SubArray, ::Int)`.
- The two forwarding methods (`apply!!(nothing, f, args...)` and the generic
  fallback) and their keyword forms are each one `@generated` method, which
  Julia compiles for the exact types of the call, so they stay as type stable
  as in 0.1.1 and now also past 8 arguments.

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
