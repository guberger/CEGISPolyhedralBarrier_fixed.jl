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
System = CPB.System
PointSet = CPB.PointSet
PolyFunc = CPB.PolyFunc
MultiPolyFunc = CPB.MultiPolyFunc
_norm(pf::PolyFunc) = maximum(af -> max(norm(af.a, Inf), af.β), pf.afs)

function HiGHS._check_ret(ret::Cint) 
    if ret != Cint(0) && ret != Cint(1)
        error(
            "Encountered an error in HiGHS (Status $(ret)). Check the log " * 
            "for details.", 
        )
    end 
    return 
end 

solver() = Model(optimizer_with_attributes(
    HiGHS.Optimizer, "output_flag"=>false
))

_methods = (CPB.Depth1st(), CPB.DepthMin(), CPB.RadMax(), CPB.ObjMin())
_PRs = ("full", "none", 2)

## Set #1
sys = System{1}()

pf_dom = PolyFunc{1}()
CPB.add_af!(pf_dom, SVector(-1.0), 0.0)
CPB.add_af!(pf_dom, SVector(1.0), -2.0)
A = @SMatrix [-1.0]
b = @SVector [1.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 2)

iset = PointSet{1,2}()
CPB.add_point!(iset, 1, SVector(-1.0))
CPB.add_point!(iset, 1, SVector(1.0))

mpf_safe = MultiPolyFunc{1,2}()
CPB.add_af!(mpf_safe, 2, SVector(1.0), -1.1)

mpf_inv = MultiPolyFunc{1,2}()

lear = CPB.Learner((10, 10), sys, mpf_safe, mpf_inv, iset)
CPB.set_tol!(lear, :rad, 10)
CPB.set_param!(lear, :xmax, 1e2)

@testset "set tol and param" begin
    @test_throws AssertionError CPB.set_tol!(lear, :dumb, 0)
    @test_throws AssertionError CPB.set_param!(lear, :dumb, 0)
    @test lear.tols[:rad] ≈ 10
    @test lear.params[:xmax] ≈ 100
end

lear = CPB.Learner((0, 1), sys, mpf_safe, mpf_inv, iset)

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 1, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov #1: max iter" begin
            @test status == CPB.MAX_ITER_REACHED
            @test length(gen.neg_evids) + length(gen.pos_evids) == 2
        end
    end
end

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 30, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov #1: found" begin
            @test status == CPB.BARRIER_FOUND
        end
    end
end

# Set #2
sys = System{1}()

pf_dom = PolyFunc{1}()
A = @SMatrix [-1.0]
b = @SVector [0.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 2)

iset = PointSet{1,2}()
CPB.add_point!(iset, 1, SVector(-1.0))
CPB.add_point!(iset, 1, SVector(1.0))

mpf_safe = MultiPolyFunc{1,2}()
CPB.add_af!(mpf_safe, 2, SVector(1.0), -1.1)

mpf_inv = MultiPolyFunc{1,2}()
CPB.add_af!(mpf_inv, 1, SVector(1.0), 0.0)

lear = CPB.Learner((0, 1), sys, mpf_safe, mpf_inv, iset)

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 30, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov #2: rad too small" begin
            @test status == CPB.BARRIER_INFEASIBLE
        end
    end
end

lear = CPB.Learner((1, 1), sys, mpf_safe, mpf_inv, iset)

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 30, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov #2: found" begin
            @test status == CPB.BARRIER_FOUND
        end
    end
end

# Set #3
sys = System{1}()

pf_dom = PolyFunc{1}()
CPB.add_af!(pf_dom, SVector(1.0), 0.0)
A = @SMatrix [-1.0]
b = @SVector [0.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 2)

pf_dom = PolyFunc{1}()
CPB.add_af!(pf_dom, SVector(-1.0), 0.0)
CPB.add_af!(pf_dom, SVector(1.0), -3.0)
A = @SMatrix [1.0]
b = @SVector [-1.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 2)

iset = PointSet{1,2}()
CPB.add_point!(iset, 1, SVector(-1.0))
CPB.add_point!(iset, 1, SVector(1.0))

mpf_safe = MultiPolyFunc{1,2}()
CPB.add_af!(mpf_safe, 2, SVector(1.0), -1.1)

mpf_inv = MultiPolyFunc{1,2}()
CPB.add_af!(mpf_inv, 2, SVector(1.0), -100.0)

lear = CPB.Learner((1, 1), sys, mpf_safe, mpf_inv, iset)

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 30, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov #3: rad too small" begin
            @test status == CPB.BARRIER_INFEASIBLE
        end
    end
end

lear = CPB.Learner((2, 1), sys, mpf_safe, mpf_inv, iset)

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 100, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov #3: found" begin
            @test status == CPB.BARRIER_FOUND
        end
    end
end

nothing