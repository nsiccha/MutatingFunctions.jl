module MutatingFunctionsLinearAlgebraExt

import MutatingFunctions: apply!!
import LinearAlgebra: mul!, ldiv!, rdiv!, lu

# A * B → mul!(cache, A, B); `cache` must already be sized to the result shape
# (mul! never resizes — the same pre-sized convention `mul!` itself expects).
apply!!(cache::AbstractVecOrMat, ::typeof(*), A, B) = (mul!(cache, A, B); cache)

# A \ B → ldiv!(cache, lu(A), B). Plain dense `A` has no 3-arg `ldiv!` of its
# own (Base's generic 3-arg form needs a `Factorization`); `lu(A)` still
# allocates the factors, but the *solution* goes straight into `cache`.
apply!!(cache::AbstractVecOrMat, ::typeof(\), A, B) = (ldiv!(cache, lu(A), B); cache)

# A / B → same story, rotated: copy `A` into `cache`, then `rdiv!` against `lu(B)`.
apply!!(cache::AbstractVecOrMat, ::typeof(/), A, B) =
    (copyto!(cache, A); rdiv!(cache, lu(B)); cache)

end # module
