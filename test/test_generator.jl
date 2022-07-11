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
Generator = CPB.Generator
NegEvidence = CPB.NegEvidence
PosEvidence = CPB.PosEvidence
PolyFunc = CPB.PolyFunc
_norm(pf::PolyFunc) = maximum(af -> max(norm(af.a, Inf), af.β), pf.afs)

solver() = Model(optimizer_with_attributes(
    HiGHS.Optimizer, "output_flag"=>false
))

## Empty
gen = CPB.Generator{2}((1, 1, 1))

mpf, r = CPB.compute_mpf_robust(gen, solver)

@testset "compute mpf empty" begin
    @test length(mpf.pfs) == 3
    @test all(pf -> length(pf.afs) == 1, mpf.pfs)
    @test r ≈ 10
end

## 1 Loc #1
gen = CPB.Generator{2}((1,))

CPB.add_evidence!(gen, NegEvidence(1, SVector(0.0, 0.0)))
CPB.add_evidence!(gen, PosEvidence(1, 1, SVector(0.5, 0.0)))

mpf, r = CPB.compute_mpf_robust(gen, solver)

@testset "compute mpf 1 Loc #1" begin
    @test maximum(pf -> _norm(pf), mpf.pfs) ≈ 1
    @test r ≈ 0.5/2.5
end

## 1 Loc #2
gen = CPB.Generator{2}((1,))

CPB.add_evidence!(gen, NegEvidence(1, SVector(0.0, 0.0)))
CPB.add_evidence!(gen, PosEvidence(1, 1, SVector(8.0, 0.0)))
CPB.add_evidence!(gen, NegEvidence(1, SVector(4.0, 0.0)))

mpf, r = CPB.compute_mpf_robust(gen, solver)

@testset "compute mpf 1 Loc #2" begin
    @test maximum(pf -> _norm(pf), mpf.pfs) ≈ 1
    @test r ≈ 1/19
end

## 1 Loc #3
gen = CPB.Generator{2}((1,))

CPB.add_evidence!(gen, NegEvidence(1, SVector(0.0, 0.0)))
CPB.add_evidence!(gen, PosEvidence(1, 1, SVector(6.0, 0.0)))
CPB.add_evidence!(gen, PosEvidence(1, 1, SVector(8.0, 0.0)))

mpf, r = CPB.compute_mpf_robust(gen, solver)

@testset "compute mpf 1 Loc #3" begin
    @test maximum(pf -> _norm(pf), mpf.pfs) ≈ 1
    @test r ≈ 6/8
end

## 2 Loc #1
gen = CPB.Generator{2}((1, 1))

CPB.add_evidence!(gen, NegEvidence(1, SVector(2.0, 0.0)))
CPB.add_evidence!(gen, NegEvidence(1, SVector(-2.0, 0.0)))
CPB.add_evidence!(gen, PosEvidence(2, 1, SVector(2.0, 0.0)))
CPB.add_evidence!(gen, NegEvidence(2, SVector(4.0, 0.0)))

mpf, r = CPB.compute_mpf_robust(gen, solver)

@testset "compute mpf 2 Loc #1" begin
    @test maximum(pf -> _norm(pf), mpf.pfs) ≈ 1
    @test r ≈ 1/11
end

## 2 Loc #2
gen = CPB.Generator{2}((2, 1))

CPB.add_evidence!(gen, NegEvidence(1, SVector(2.0, 0.0)))
CPB.add_evidence!(gen, NegEvidence(1, SVector(-2.0, 0.0)))
CPB.add_evidence!(gen, PosEvidence(1, 1, SVector(4.0, 0.0)))
CPB.add_evidence!(gen, PosEvidence(1, 2, SVector(-4.0, 0.0)))
CPB.add_evidence!(gen, NegEvidence(2, SVector(1.0, 0.0)))
CPB.add_evidence!(gen, NegEvidence(1, SVector(2.0, 0.0)))

mpf, r = CPB.compute_mpf_robust(gen, solver)

@testset "compute mpf 2 Loc #2" begin
    @test maximum(pf -> _norm(pf), mpf.pfs) ≈ 1
    @test r ≈ 1/11
end

nothing