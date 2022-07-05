using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISPolyhedralBarrier.jl")
else
    using CEGISPolyhedralBarrier
end
CPB = CEGISPolyhedralBarrier

p1 = CPB.Polyhedron()
CPB.add_halfspace!(p1, [1, 1], 1)

@testset "polyhedron" begin
    @test [1, -2.1] ∈ p1
    @test [1, -1.9] ∉ p1
    @test !CPB.near([1, -1.9], p1, 0.1/sqrt(3) - 1e-8)
    @test CPB.near([1, -1.9], p1, 0.1/sqrt(3) + 1e-8)
end

p2 = CPB.Polyhedron()
CPB.add_halfspace!(p2, [-1, -1], -2)
p3 = p1 ∩ p2

@testset "polyhedron" begin
    @test [0, -2.1] ∉ p3
    @test !CPB.near([0, -2.1], p3, 0.1/sqrt(6) - 1e-8)
    @test CPB.near([0, -2.1], p3, 0.1/sqrt(6) + 1e-8)
end

nothing