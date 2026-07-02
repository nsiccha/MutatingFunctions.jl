using MutatingFunctions, Test
using Statistics: quantile

@testset "MutatingFunctions" begin
    @testset "nothing cache = allocating" begin
        @test apply!(nothing, zeros, 3) == zeros(3)
        @test apply!(nothing, map, x -> x^2, 1:3) == [1, 4, 9]
    end

    @testset "cache reuse (Base forms)" begin
        c = Float64[]
        @test apply!(c, zeros, 3) === c
        @test c == zeros(3)
        @test apply!(c, ones, 2) === c && c == ones(2)
        @test apply!(c, fill, 7.0, 4) === c && c == fill(7.0, 4)
        d = Int[]
        @test apply!(d, map, x -> 2x, 1:3) === d && d == [2, 4, 6]
    end

    @testset "generic fallback" begin
        myalloc(n) = collect(1:n) .+ 0.0        # unregistered, returns a Vector
        c = Float64[]
        @test apply!(c, myalloc, 3) === c && c == [1.0, 2.0, 3.0]
        @test apply!(c, sum, 1:5) == 15         # scalar result → passthrough
    end

    @testset "Statistics extension: quantile" begin
        c = Float64[]
        p = [0.25, 0.5, 0.75]
        @test apply!(c, quantile, collect(1.0:4.0), p) === c
        @test c == quantile(1.0:4.0, p)
    end
end
