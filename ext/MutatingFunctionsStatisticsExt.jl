module MutatingFunctionsStatisticsExt

import MutatingFunctions: apply!!
import Statistics

# quantile(v, p::AbstractVector) → a vector of quantiles, written into `cache`.
# Note: `quantile!` SORTS `v` in place (the mutating contract), exactly like Base.
apply!!(cache::AbstractVector, ::typeof(Statistics.quantile), v, p::AbstractVector) =
    (resize!(cache, length(p)); Statistics.quantile!(cache, v, p); cache)

end # module
