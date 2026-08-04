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
    end
end
