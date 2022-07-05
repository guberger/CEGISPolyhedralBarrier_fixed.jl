using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISPolyhedralBarrier.jl")
else
    using CEGISPolyhedralBarrier
end
CPB = CEGISPolyhedralBarrier
Polyhedron = CPB.Polyhedron
PolyFunc = CPB.PolyFunc
MultiPolyFunc = CPB.MultiPolyFunc
PosPredicate = CPB.PosPredicate
LiePredicate = CPB.LiePredicate

solver() = Model(optimizer_with_attributes(
    HiGHS.Optimizer, "output_flag"=>false
))

## Parameters
N = 2

## Pos infeasible
verif = CPB.Verifier()
domain = Polyhedron()
CPB.add_halfspace!(domain, [1, 1], -1)
CPB.add_halfspace!(domain, [-1, -1], 2)
CPB.add_predicate!(verif, PosPredicate(N, domain, 1))

mpf = MultiPolyFunc(2)
afs_ = [([-0.5, 0.5], -0.5), ([1, 0], 0)]
for af_ in afs_
    CPB.add_af!(mpf, 1, af_...)
end

xmax = 1e3
x, r, loc = CPB.verify_pos(verif, mpf, xmax, 1e4, solver)

@testset "verify pos infeasible" begin
    @test r == -Inf
    @test isempty(x)
    @test loc == 0
end

## Pos false #1 #2 #3
verif = CPB.Verifier()
domain = Polyhedron()
CPB.add_halfspace!(domain, [1, 1], -1)
CPB.add_predicate!(verif, PosPredicate(N, domain, 1))

mpf = MultiPolyFunc(2)
afs_ = [([-0.5, 0.5], -0.5), ([1, 0], 0)]
for af_ in afs_
    CPB.add_af!(mpf, 1, af_...)
end

xmax = 1e3
x, r, loc = CPB.verify_pos(verif, mpf, xmax, 1e4, solver)

@testset "verify pos false #1" begin
    @test r ≈ (xmax + 1)/3
    @test norm(x, Inf) ≈ xmax
    @test x ∈ domain
    @test loc == 1
end

domain = Polyhedron()
CPB.add_af!(mpf, 2, [4, 0], 1)
CPB.add_predicate!(verif, PosPredicate(N, domain, 2))

x, r, loc = CPB.verify_pos(verif, mpf, xmax, 1e4, solver)

@testset "verify pos false #2" begin
    @test r ≈ 4*xmax - 1
    @test loc == 2
end

CPB.add_af!(mpf, 2, [4, 0], 4000)

x, r, loc = CPB.verify_pos(verif, mpf, xmax, 1e4, solver)

@testset "verify pos false #3" begin
    @test r ≈ (xmax + 1)/3
    @test loc == 1
end

# Pos true
verif = CPB.Verifier()
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, -1], 1)
CPB.add_predicate!(verif, PosPredicate(N, domain, 1))

mpf = MultiPolyFunc(2)
afs_ = [([-0.5, 0.5], 0.5), ([1, 0], 0)]
for af_ in afs_
    CPB.add_af!(mpf, 1, af_...)
end

xmax = 1e3
x, r, loc = CPB.verify_pos(verif, mpf, xmax, 1e4, solver)

@testset "verify pos true" begin
    @test r ≈ -1/2
    @test x ≈ [1/2, 1/2]
    @test x ∈ domain
    @test loc == 1
end

## Lie infeasible
verif = CPB.Verifier()
domain = Polyhedron()
A = [0.5 0.0; 1.0 1.0]
b = [1, 0]
CPB.add_predicate!(verif, LiePredicate(N, domain, 1, A, b, 1))

mpf = MultiPolyFunc(2)
afs_ = [([-1.0, 0.0], 1), ([1.0, 0.0], 1)]
for af_ in afs_
    CPB.add_af!(mpf, 1, af_...)
end

x, r, loc = CPB.verify_lie(verif, mpf, 1e3, 1e4, solver)

@testset "verify lie infeasible" begin
    @test r == -Inf
    @test isempty(x)
    @test loc == 0
end

## Lie false #1
verif = CPB.Verifier()
domain = Polyhedron()
A = [0.5 0.0; 1.0 1.0]
b = [1, 0]
CPB.add_predicate!(verif, LiePredicate(N, domain, 1, A, b, 1))

mpf = MultiPolyFunc(2)
afs_ = [([-1.0, 0.0], -1), ([1.0, 0.0], -1)]
for af_ in afs_
    CPB.add_af!(mpf, 1, af_...)
end

x, r, loc = CPB.verify_lie(verif, mpf, 1e3, 1e4, solver)

@testset "verify lie false #1" begin
    @test r ≈ 1/2
    @test x[1] ≈ 1
    @test x ∈ domain
    @test loc == 1
end

## Lie false #2
verif = CPB.Verifier()
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0)
A = [0.0 0.5; 0.5 0.1]
b = [0, 1]
CPB.add_predicate!(verif, LiePredicate(N, domain, 1, A, b, 1))

mpf = MultiPolyFunc(2)
afs_ = [([-1, 0], -1), ([1, 0], -1), ([0, -1], -1), ([0, 1], -1)]
for af_ in afs_
    CPB.add_af!(mpf, 1, af_...)
end

x, r, loc = CPB.verify_lie(verif, mpf, 1e3, 1e4, solver)

@testset "verify lie false #2" begin
    @test r ≈ 0.6
    @test x ≈ [1, 1]
    @test x ∈ domain
    @test loc == 1
end

## Lie true #1
verif = CPB.Verifier()
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0)
A = [0.0 0.5; 0.5 0.1]
b = [0.0, -0.5]
CPB.add_predicate!(verif, LiePredicate(N, domain, 1, A, b, 1))

mpf = MultiPolyFunc(2)
afs_ = [([-1, 0], -1), ([1, 0], -1), ([0, -1], -1), ([0, 1], -1)]
for af_ in afs_
    CPB.add_af!(mpf, 1, af_...)
end

x, r, loc = CPB.verify_lie(verif, mpf, 1e3, 1e4, solver)

@testset "verify lie false #2" begin
    @test r ≈ -0.4
    @test x ≈ [0, -1]
    @test x ∈ domain
    @test loc == 1
end

## Lie multiple #1
verif = CPB.Verifier()
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, -1], -1)
CPB.add_halfspace!(domain, [-1, 1], -1)
A = [0.5 -0.25; 0.1 0.5]
b = [0.0, 0.5]
CPB.add_predicate!(verif, LiePredicate(N, domain, 2, A, b, 1))

mpf = MultiPolyFunc(2)
afs_ = [([1, 0], -1), ([0, -1], -1), ([0, 1], -1)]
for af_ in afs_
    CPB.add_af!(mpf, 1, af_...)
    CPB.add_af!(mpf, 2, af_...)
end

x, r, loc = CPB.verify_lie(verif, mpf, 1e3, 1e4, solver)

@testset "verify lie multiple #1" begin
    @test r ≈ 0.1
    @test x ≈ [1, 1]
    @test x ∈ domain
    @test loc == 2
end

mpf = MultiPolyFunc(2)
afs_ = [([1, 0], -1), ([0, -1], -1), ([0, 1], -1)]
for af_ in afs_
    CPB.add_af!(mpf, 2, af_...)
end
CPB.add_af!(mpf, 1, afs_[1]...)

x, r, loc = CPB.verify_lie(verif, mpf, 1e3, 1e4, solver)

@testset "verify lie multiple #2" begin
    @test r ≈ -0.25
    @test x ≈ [1, -1]
    @test x ∈ domain
    @test loc == 2
end

nothing