module MutatingFunctionsRandomExt

import MutatingFunctions: apply!!
import Random: rand!, randn!

# rand(n) / randn(n) → resize `cache` to `n`, then fill it in place.
apply!!(cache::AbstractVector, ::typeof(rand), n::Integer) =
    (resize!(cache, n); rand!(cache); cache)
apply!!(cache::AbstractVector, ::typeof(randn), n::Integer) =
    (resize!(cache, n); randn!(cache); cache)

end # module
