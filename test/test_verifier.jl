using LinearAlgebra
using StaticArrays
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISPolyhedralBarrier.jl")
else
    using CEGISPolyhedralBarrier
end
CPB = CEGISPolyhedralBarrier
PolyFunc = CPB.PolyFunc
MultiPolyFunc = CPB.MultiPolyFunc
System = CPB.System
Verifier = CPB.Verifier

solver() = Model(optimizer_with_attributes(
    HiGHS.Optimizer, "output_flag"=>false
))

xmax = 1e3

# Set #1
sys = System{1}()
pf_dom = PolyFunc{1}()
CPB.add_af!(pf_dom, SVector(-1.0), 0.0)
A = @SMatrix [0.5]
b = @SVector [1.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

mpf_inv = MultiPolyFunc{1,1}()
CPB.add_af!(mpf_inv, 1, SVector(1.0), -1.0)

mpf_BF = MultiPolyFunc{1,1}()
CPB.add_af!(mpf_BF, 1, SVector(-1.0), 0.5)

mpf_safe = MultiPolyFunc{1,1}()

verif = Verifier(mpf_safe, mpf_inv, mpf_BF, sys, xmax)
x, r, loc = CPB.verify_safe(verif, solver)

@testset "verify safe: empty" begin
    @test r == -Inf
    @test x === SVector(NaN)
    @test loc == 0
end

mpf_safe = MultiPolyFunc{1,1}()
CPB.add_af!(mpf_safe, 1, SVector(1.0), -0.25)

verif = Verifier(mpf_safe, mpf_inv, mpf_BF, sys, xmax)
x, r, loc = CPB.verify_safe(verif, solver)

@testset "verify safe: infeasible" begin
    @test r == -Inf
    @test x === SVector(NaN)
    @test loc == 0
end

mpf_safe = MultiPolyFunc{1,1}()
CPB.add_af!(mpf_safe, 1, SVector(1.0), -1.0)

verif = Verifier(mpf_safe, mpf_inv, mpf_BF, sys, xmax)
x, r, loc = CPB.verify_safe(verif, solver)

@testset "verify safe: unsafe" begin
    @test r == 0.5
    @test x === SVector(1.0)
    @test loc == 1
end

# Set #2
sys = System{2}()
pf_dom = PolyFunc{2}()
CPB.add_af!(pf_dom, SVector(-1.0, 0.0), -1.0)
A = @SMatrix [0.0 1.0; -1.0 0.0]
b = @SVector [0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 2)

mpf_inv = MultiPolyFunc{2,2}()
CPB.add_af!(mpf_inv, 1, SVector(0.0, 1.0), -1.0)

mpf_BF = MultiPolyFunc{2,2}()
CPB.add_af!(mpf_BF, 1, SVector(1.0, 0.0), -1.0)

mpf_safe = MultiPolyFunc{2,2}()
CPB.add_af!(mpf_safe, 1, SVector(0.0, -1.0), -1.0)

verif = Verifier(mpf_safe, mpf_inv, mpf_BF, sys, xmax)
x, r, loc = CPB.verify_safe(verif, solver)

@testset "verify safe: empty" begin
    @test r == -Inf
    @test x === SVector(NaN, NaN)
    @test loc == 0
end

verif = Verifier(mpf_safe, mpf_inv, mpf_BF, sys, xmax)
x, r, loc = CPB.verify_BF(verif, solver)

@testset "verify BF: empty" begin
    @test r == -Inf
    @test x === SVector(NaN, NaN)
    @test loc == 0
end

mpf_BF = MultiPolyFunc{2,2}()
CPB.add_af!(mpf_BF, 2, SVector(1.0, 0.0), -1.0)

verif = Verifier(mpf_safe, mpf_inv, mpf_BF, sys, xmax)
x, r, loc = CPB.verify_BF(verif, solver)

@testset "verify BF: satisfied" begin
    @test r == 0.0
    @test x[2] ≈ 1
    @test loc == 1
end

pf_dom = PolyFunc{2}()
CPB.add_af!(pf_dom, SVector(-1.0, 0.0), -1.0)
A = @SMatrix [0.0 1.0; -1.0 0.0]
b = @SVector [0.0, -1.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 2)

mpf_inv = MultiPolyFunc{2,2}()
CPB.add_af!(mpf_inv, 1, SVector(1.0, 0.0), -2.0)

mpf_BF = MultiPolyFunc{2,2}()

mpf_safe = MultiPolyFunc{2,2}()
CPB.add_af!(mpf_safe, 1, SVector(0.0, -1.0), -1.0)
CPB.add_af!(mpf_safe, 2, SVector(0.0, -1.0), -1.0)

verif = Verifier(mpf_safe, mpf_inv, mpf_BF, sys, xmax)
x, r, loc = CPB.verify_safe(verif, solver)

@testset "verify safe: unsafe" begin
    @test r ≈ 2
    @test x[1] ≈ 2
    @test loc == 1
end

nothing