using MutatingFunctions, Test
using LinearAlgebra
using Random
using Statistics: quantile

@testset "MutatingFunctions" begin
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
    end
end
