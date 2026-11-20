@testitem "MutatingFunctions" begin
    using MutatingFunctions, Test
    using LinearAlgebra
    using Random
    using Statistics: quantile

    @testset "nothing cache = allocating" begin
        @test apply!!(nothing, zeros, 3) == zeros(3)
        @test apply!!(nothing, map, x -> x^2, 1:3) == [1, 4, 9]
    end

    @testset "cache reuse (Base forms)" begin
        c = Float64[]
        @test apply!!(c, zeros, 3) === c
        @test c == zeros(3)
        @test apply!!(c, ones, 2) === c && c == ones(2)
        @test apply!!(c, fill, 7.0, 4) === c && c == fill(7.0, 4)
        d = Int[]
        @test apply!!(d, map, x -> 2x, 1:3) === d && d == [2, 4, 6]
    end

    @testset "generic fallback" begin
        myalloc(n) = collect(1:n) .+ 0.0        # unregistered, returns a Vector
        c = Float64[]
        @test apply!!(c, myalloc, 3) === c && c == [1.0, 2.0, 3.0]
        @test apply!!(c, sum, 1:5) == 15         # scalar result → passthrough
    end

    @testset "Statistics extension: quantile" begin
        c = Float64[]
        p = [0.25, 0.5, 0.75]
        @test apply!!(c, quantile, collect(1.0:4.0), p) === c
        @test c == quantile(1.0:4.0, p)
    end

    @testset "obvious Base functions" begin
        c = Float64[]
        @test apply!!(c, sort, [3.0, 1.0, 2.0]) === c && c == [1.0, 2.0, 3.0]
        @test apply!!(c, sort, [3.0, 1.0, 2.0]; rev=true) === c && c == [3.0, 2.0, 1.0]
        @test apply!!(c, reverse, [1.0, 2.0, 3.0]) === c && c == [3.0, 2.0, 1.0]
        @test apply!!(c, cumsum, [1.0, 2.0, 3.0]) === c && c == [1.0, 3.0, 6.0]
        @test apply!!(c, cumprod, [1.0, 2.0, 3.0]) === c && c == [1.0, 2.0, 6.0]
        @test apply!!(c, accumulate, +, [1.0, 2.0, 3.0]) === c && c == [1.0, 3.0, 6.0]
        @test apply!!(c, collect, 1.0:3.0) === c && c == [1.0, 2.0, 3.0]
        @test apply!!(c, copy, [4.0, 5.0]) === c && c == [4.0, 5.0]
        @test apply!!(c, broadcast, +, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) === c &&
              c == [11.0, 22.0, 33.0]
    end

    @testset "LinearAlgebra extension" begin
        A = [2.0 0.0; 0.0 3.0]
        b = [1.0, 2.0]
        c = zeros(2)
        @test apply!!(c, *, A, b) === c && c == A * b

        c2 = zeros(2)
        @test apply!!(c2, \, A, b) === c2 && c2 ≈ A \ b

        C = [4.0 0.0; 0.0 6.0]
        D = [2.0 0.0; 0.0 3.0]
        c3 = zeros(2, 2)
        @test apply!!(c3, /, C, D) === c3 && c3 ≈ C / D
    end

    @testset "Random extension" begin
        c = Float64[]
        @test apply!!(c, rand, 5) === c && length(c) == 5 && all(0 .<= c .<= 1)
        @test apply!!(c, randn, 3) === c && length(c) == 3
    end

    @testset "N-D broadcast fills any-rank cache in place" begin
        # Matrix cache: the generic fallback used to ignore it and allocate;
        # now it fills in place.
        tau = [2.0, 3.0]
        M = reshape(collect(1.0:10.0), 2, 5)
        cm = zeros(2, 5)
        @test apply!!(cm, broadcast, *, tau, M) === cm && cm == tau .* M
        # 3-arg fused vector broadcast fills the same buffer
        cv = zeros(4)
        @test apply!!(cv, broadcast, (x, y, z) -> x*y + z,
                      [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0],
                      [1.0, 1.0, 1.0, 1.0]) === cv && cv == [11.0, 41.0, 91.0, 161.0]
        # vector resize-to-fit convenience is preserved
        c = Float64[]
        @test apply!!(c, broadcast, +, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) === c &&
              c == [11.0, 22.0, 33.0]
    end

    @testset "gather (getindex) fills cache in place" begin
        A = collect(10.0:2.0:30.0)          # 11 elements
        c = Float64[]
        @test apply!!(c, getindex, A, [3, 1, 5, 2]) === c && c == A[[3, 1, 5, 2]]
        # range and boolean-mask indices size the cache correctly
        @test apply!!(c, getindex, A, 2:4) === c && c == A[2:4]
        mask = falses(length(A)); mask[[1, 6, 11]] .= true
        @test apply!!(c, getindex, A, mask) === c && c == A[mask]

        # SubArray source — the load-bearing PPL case `b = @view θ[lo:hi]; b[group]`.
        # A composed view-of-a-view used to make plain `AutoEnzyme` throw
        # `EnzymeRuntimeActivityError`; the elementwise gather reads scalars and
        # differentiates cleanly. Cover integer, range and boolean-mask indices.
        v = @view A[2:8]                    # 7-element SubArray into A
        @test apply!!(c, getindex, v, [3, 1, 5, 2]) === c && c == v[[3, 1, 5, 2]]
        @test apply!!(c, getindex, v, 2:4) === c && c == v[2:4]
        vmask = falses(length(v)); vmask[[1, 4, 7]] .= true
        @test apply!!(c, getindex, v, vmask) === c && c == v[vmask]
        # BitVector and Vector{Bool} masks both dispatch to the mask form
        @test apply!!(c, getindex, A, Bool[i in (1, 6, 11) for i in 1:length(A)]) === c &&
              c == A[mask]
        # a wrong-length mask is a DimensionMismatch, not a silent bad gather
        @test_throws DimensionMismatch apply!!(c, getindex, A, falses(length(A) - 1))
    end

    end
end

@testitem "MutatingFunctions multi-dim gather" begin
    using MutatingFunctions, Test
    using LinearAlgebra
    using Random
    using Statistics: quantile

        @testset "multi-dim gather (getindex A[i, idx]) fills cache in place" begin
            # scalar row index + vector column index — the `b[k, group]` per-row
            # gather that dominates a grouped hierarchical model. Cover the three
            # PPL source types: dense Matrix, ReshapedArray (reshaped pool vector),
            # SubArray view.
            Mat  = reshape(collect(1.0:30.0), 3, 10)
            Resh = reshape(1.0:30.0, 3, 10)                  # Base.ReshapedArray
            Sub  = @view Mat[:, 2:9]                          # SubArray, 3 x 8
            idx  = [3, 1, 5, 2, 4]
            c = Float64[]
            @test apply!!(c, getindex, Mat, 2, idx) === c && c == Mat[2, idx]
            @test apply!!(c, getindex, Resh, 3, idx) === c && c == Resh[3, idx]
            sidx = [1, 4, 2, 8]
            @test apply!!(c, getindex, Sub, 2, sidx) === c && c == Sub[2, sidx]
            # range column index sizes the cache correctly
            @test apply!!(c, getindex, Mat, 1, 2:5) === c && c == Mat[1, 2:5]
            # boolean column mask: output length is count(mask), sized from size(A, 2)
            mask = falses(size(Mat, 2)); mask[[1, 5, 10]] .= true
            @test apply!!(c, getindex, Mat, 3, mask) === c && c == Mat[3, mask]
            @test apply!!(c, getindex, Resh, 2, mask) === c && c == Resh[2, mask]
            # a wrong-length column mask is a DimensionMismatch (checked vs size(A, 2))
            @test_throws DimensionMismatch apply!!(c, getindex, Mat, 1, falses(size(Mat, 2) - 1))
            # an already-sized view destination fills in place (no resize! on a SubArray)
            flat = zeros(20)
            cv = @view flat[1:5]
            @test apply!!(cv, getindex, Mat, 2, idx) === cv && cv == Mat[2, idx]
        end
end

@testitem "MutatingFunctions sized-view destinations" begin
    using MutatingFunctions, Test
    using LinearAlgebra
    using Random
    using Statistics: quantile

        @testset "already-sized view destination fills in place (no resize!)" begin
            # A `SubArray`/`@view` has no `resize!` method, so every resize-to-fit
            # form used to throw `MethodError: resize!(::SubArray, ::Int)` even when
            # the view was ALREADY the right length. It must now fill in place — the
            # load-bearing case is a single flat backing array handing out contiguous
            # views as pooled buffers (one Enzyme shadow array instead of N).
            flat = zeros(20)
            v(lo, hi) = @view flat[lo:hi]   # a correctly-sized view into the backing array

            # generic fallback (unregistered f returning a Vector)
            myf(x) = x .+ 1
            c = v(1, 3); @test apply!!(c, myf, [1.0, 2.0, 3.0]) === c && c == [2.0, 3.0, 4.0]

            # zeros / ones / fill
            c = v(1, 3); @test apply!!(c, zeros, 3) === c && c == zeros(3)
            c = v(1, 2); @test apply!!(c, ones, 2) === c && c == ones(2)
            c = v(1, 4); @test apply!!(c, fill, 7.0, 4) === c && c == fill(7.0, 4)

            # map
            c = v(1, 3); @test apply!!(c, map, x -> 2x, [1.0, 2.0, 3.0]) === c && c == [2.0, 4.0, 6.0]

            # sort / reverse
            c = v(1, 3); @test apply!!(c, sort, [3.0, 1.0, 2.0]) === c && c == [1.0, 2.0, 3.0]
            c = v(1, 3); @test apply!!(c, reverse, [1.0, 2.0, 3.0]) === c && c == [3.0, 2.0, 1.0]

            # cumsum / cumprod / accumulate
            c = v(1, 3); @test apply!!(c, cumsum, [1.0, 2.0, 3.0]) === c && c == [1.0, 3.0, 6.0]
            c = v(1, 3); @test apply!!(c, cumprod, [1.0, 2.0, 3.0]) === c && c == [1.0, 2.0, 6.0]
            c = v(1, 3); @test apply!!(c, accumulate, +, [1.0, 2.0, 3.0]) === c && c == [1.0, 3.0, 6.0]

            # collect / copy
            c = v(1, 3); @test apply!!(c, collect, 1.0:3.0) === c && c == [1.0, 2.0, 3.0]
            c = v(1, 2); @test apply!!(c, copy, [4.0, 5.0]) === c && c == [4.0, 5.0]

            # broadcast (Vector and N-D views)
            c = v(1, 3); @test apply!!(c, broadcast, +, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) === c &&
                  c == [5.0, 7.0, 9.0]
            M = zeros(4, 4); cm = @view M[1:2, 1:2]
            @test apply!!(cm, broadcast, +, [1.0 2.0; 3.0 4.0], [10.0 20.0; 30.0 40.0]) === cm &&
                  cm == [11.0 22.0; 33.0 44.0]

            # getindex gather (integer + boolean mask), the snag's second repro
            A = [10.0, 20.0, 30.0, 40.0]
            c = v(1, 3); @test apply!!(c, getindex, A, [2, 4, 1]) === c && c == [20.0, 40.0, 10.0]
            c = v(1, 2); @test apply!!(c, getindex, A, Bool[1, 0, 1, 0]) === c && c == [10.0, 30.0]

            # rand / randn / quantile — the last resize-to-fit forms; a sized view
            # destination must fill in place, not throw resize!(::SubArray, ::Int)
            c = v(1, 5); @test apply!!(c, rand, 5) === c && length(c) == 5 && all(0 .<= c .<= 1)
            c = v(1, 3); @test apply!!(c, randn, 3) === c && length(c) == 3
            c = v(1, 3); p = [0.25, 0.5, 0.75]
            @test apply!!(c, quantile, collect(1.0:4.0), p) === c && c == quantile(1.0:4.0, p)
        end
end

@testitem "MutatingFunctions forwarding methods specialize" begin
    using MutatingFunctions, Test

    # The signature Julia compiles a call with these argument types into (the
    # method instance a call that is not inlined runs), next to the call's own
    # concrete signature. `kw=true` reads the `Core.kwcall` method a call with
    # keywords reaches.
    function compiled_and_called(f, argtypes...; kw=false)
        if kw
            kws = typeof((; s=1))
            m = which(Core.kwcall, Tuple{kws, typeof(f), argtypes...})
            atype = Tuple{typeof(Core.kwcall), kws, typeof(f), argtypes...}
        else
            m = which(f, Tuple{argtypes...})
            atype = Tuple{typeof(f), argtypes...}
        end
        env = ccall(:jl_type_intersection_with_env, Any, (Any, Any), atype, m.sig)[2]
        (Core.Compiler.get_compileable_sig(m, atype, env), atype)
    end
    specialized(f, argtypes...; kwargs...) = ==(compiled_and_called(f, argtypes...; kwargs...)...)

    struct Step end            # a callable struct, like a ReactiveKernels step
    (::Step)(xs...; kwargs...) = first(xs)
    V, I, M = Vector{Float64}, Vector{Int}, Matrix{Float64}
    forward(f, args...; kwargs...) = f(args...; kwargs...)   # the 0.1.0 spelling

    forward_n(f::F, args::Vararg{Any,N}) where {F,N} = f(args...)
    @testset "control: an ordinary forwarder is compiled for widened types" begin
        @test !specialized(forward, Step, V, I; kw=true)       # … Vector{Float64}, Vararg{Any}
        @test !specialized(forward, typeof(sum), V)            # f compiled as ::Function
        # `Vararg{Any,N}` alone still compiles a function or type argument as
        # `::Function` / `::Type`: the reason the forwarding methods are generated
        @test !specialized(forward_n, Step, typeof(+), V)
        @test !specialized(forward_n, Step, Type{Float64}, Int)
    end

    # array arguments, and arguments that are themselves functions or types
    arrays(n) = ntuple(k -> isodd(k) ? V : I, n)
    mixed(n) = ntuple(k -> (V, typeof(+), I, Type{Float64}, Float64, typeof(sin), Step, M)[mod1(k, 8)], n)
    @testset "apply!!(nothing | cache, f, args...), with and without keywords" begin
        for C in (Nothing, V), F in (Step, typeof(sum)), args in (arrays.(0:10)..., mixed.(1:12)...)
            @test specialized(apply!!, C, F, args...)
            @test specialized(apply!!, C, F, args...; kw=true)
        end
    end

    # ReactiveKernels keeps a cache slot for a step only when `apply!!` selects a
    # registered form for its types, which it tells from the copy fallback by
    # method identity: `which(apply!!, Tuple{T, typeof(op), argtypes...})` is or
    # is not `which(apply!!, Tuple{Any,Any,Vararg{Any}})`. 0.1.1's one method per
    # arity broke that for 1-8 arguments, and its non-allocating AD preparation
    # threw (snag `apply-0-1-1-nona-d289b9ca`).
    @testset "every unregistered call resolves to the one generic method" begin
        generic = which(apply!!, Tuple{Any,Any,Vararg{Any}})
        nothing_form = which(apply!!, Tuple{Nothing,Any,Vararg{Any}})
        @test generic !== nothing_form
        bc = typeof(Base.broadcasted(+, zeros(2), zeros(2)))
        sub = typeof(view(zeros(3), [1, 2]))
        for C in (V, M, Float64, bc, sub, Any), F in (Step, typeof(sum), typeof(Base.broadcasted), typeof(view)),
                args in (arrays.(0:10)..., mixed.(1:12)...)
            @test which(apply!!, Tuple{C, F, args...}) === generic
            @test which(apply!!, Tuple{Nothing, F, args...}) === nothing_form
        end
        # and a registered form is not it
        @test which(apply!!, Tuple{V, typeof(zeros), Int}) !== generic
        @test which(apply!!, Tuple{V, typeof(broadcast), typeof(+), V, V}) !== generic
    end

    @testset "registered forms still win over the forwarding methods" begin
        # A representative call for each registered form (type variables at their
        # bound, `Any` as `Float64`, varargs expanded to 1-3 arguments) must still
        # dispatch to that form, not to a forwarding method.
        bound(t) = t isa TypeVar ? (t.ub === Any ? Float64 : t.ub) : t
        nforms = 0
        for m in methods(apply!!)
            m.module === MutatingFunctions || continue
            ps = collect(Base.unwrap_unionall(m.sig).parameters)[2:end]
            ps[2] isa TypeVar && continue                      # a forwarding method
            nforms += 1
            fixed = [bound(p) for p in ps if !(p isa Core.TypeofVararg)]
            va = filter(p -> p isa Core.TypeofVararg, ps)
            for extra in (isempty(va) ? (0:0) : (1:3))
                @test which(apply!!, Tuple{fixed..., fill(Float64, extra)...}) === m
            end
        end
        @test nforms >= 20
        # The keyword forms (`sort`, `cumsum`, …) also win over the generic
        # `Core.kwcall` method.
        kws = typeof((; rev=true))
        generic_kw = which(Core.kwcall, Tuple{kws, typeof(apply!!), Any, Any, Vararg{Any}})
        @test which(Core.kwcall, Tuple{kws, typeof(apply!!), V, typeof(sort), V}) !== generic_kw
        @test which(Core.kwcall, Tuple{kws, typeof(apply!!), V, typeof(cumsum), V}) !== generic_kw
    end

    @testset "the generated methods compute what the 0.1.0 methods did" begin
        @test apply!!(nothing, broadcast, +, 1.0, [1.0, 2.0]) == [2.0, 3.0]
        @test apply!!(nothing, convert, Float64, 3) === 3.0
        @test apply!!(nothing, map, sin, [0.0]) == [0.0]
        @test apply!!(nothing, (xs...) -> sum(xs), 1:9...) == 45
        @test apply!!(nothing, () -> 7) == 7
        @test apply!!(nothing, (x; s=1) -> s * x, 2; s=3) == 6
        @test apply!!(nothing, (x; s=1) -> s * x, 2) == 2
        c = Float64[]
        @test apply!!(c, broadcast, +, 1.0, [1.0, 2.0]) === c && c == [2.0, 3.0]
        @test apply!!(c, (xs...) -> collect(Float64, xs), 1:9...) === c && c == 1.0:9.0
        @test apply!!(c, (x; s=1) -> s .* x, [1.0, 2.0]; s=3) === c && c == [3.0, 6.0]
        @test apply!!(c, sort, [3.0, 1.0, 2.0]; rev=true) === c && c == [3.0, 2.0, 1.0]
        @test apply!!(c, sum, [1.0, 2.0]) === 3.0                  # scalar passthrough
        @test apply!!(c, sum, [1.0 2.0; 3.0 4.0]; dims=1) == [4.0 6.0]   # shape mismatch passthrough
        @test_throws MethodError apply!!(c, zeros, 3; nonsense=1)
    end

    @testset "registered forms that forward a function" begin
        for n in 1:7
            @test specialized(apply!!, V, typeof(broadcast), typeof(+), ntuple(_ -> V, n)...)
            @test specialized(apply!!, M, typeof(broadcast), typeof(+), ntuple(_ -> M, n)...)
            @test specialized(apply!!, V, typeof(map), typeof(+), ntuple(_ -> V, n)...)
        end
        @test specialized(apply!!, V, typeof(accumulate), typeof(+), V)
        @test specialized(apply!!, V, typeof(accumulate), typeof(+), V; kw=true)
    end

    @test isempty(Test.detect_ambiguities(MutatingFunctions))
end
