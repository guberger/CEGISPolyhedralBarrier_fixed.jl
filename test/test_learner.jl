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
System = CPB.System
PointSet = CPB.PointSet
PolyFunc = CPB.PolyFunc
_norm(pf::PolyFunc) = maximum(lf -> max(norm(lf.lin, Inf), lf.off), pf.afs)

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

## Learner Disc
sys = System()

domain = Polyhedron()
CPB.add_halfspace!(domain, [-1], 0)
CPB.add_halfspace!(domain, [1], -2)
A = reshape([-1], 1, 1)
b = [1]
CPB.add_piece!(sys, domain, 1, A, b, 2)

iset = PointSet{2}()
CPB.add_point!(iset, 1, [-1])
CPB.add_point!(iset, 1, [1])

uset = PointSet{2}()
CPB.add_point!(uset, 2, [1.1])
CPB.add_point!(uset, 2, [2.1])

lear = CPB.Learner{1}((10, 10), sys, iset, uset)
CPB.set_tol!(lear, :radius, 10)
CPB.set_param!(lear, :xmax, 1e2)

@testset "set tol and param" begin
    @test_throws AssertionError CPB.set_tol!(lear, :dumb, 0)
    @test_throws AssertionError CPB.set_param!(lear, :dumb, 0)
    @test lear.tols[:radius] ≈ 10
    @test lear.params[:xmax] ≈ 100
end

lear = CPB.Learner{1}((10, 10), sys, iset, uset)

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 1, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov disc: max iter" begin
            @test status == CPB.MAX_ITER_REACHED
            @test length(gen.neg_evids) + length(gen.pos_evids) == 0
        end
    end
end

lear = CPB.Learner{1}((0, 1), sys, iset, uset)

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 30, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov disc: found" begin
            @test status == CPB.BARRIER_FOUND
        end
    end
end

sys = System()

domain = Polyhedron()
CPB.add_halfspace!(domain, [1], 0)
A = reshape([-1], 1, 1)
b = [0]
CPB.add_piece!(sys, domain, 1, A, b, 2)

lear = CPB.Learner{1}((-1, 1), sys, iset, uset)

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 30, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov disc: rad too small" begin
            @test status == CPB.BARRIER_INFEASIBLE
        end
    end
end

lear = CPB.Learner{1}((1, 1), sys, iset, uset)

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 30, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov disc: found" begin
            @test status == CPB.BARRIER_FOUND
        end
    end
end

sys = System()

domain = Polyhedron()
CPB.add_halfspace!(domain, [1], 0)
A = reshape([-1], 1, 1)
b = [0]
CPB.add_piece!(sys, domain, 1, A, b, 2)

domain = Polyhedron()
CPB.add_halfspace!(domain, [-1], 0)
CPB.add_halfspace!(domain, [1], -3)
A = reshape([1], 1, 1)
b = [-1]
CPB.add_piece!(sys, domain, 1, A, b, 2)

lear = CPB.Learner{1}((0, 0), sys, iset, uset)

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 30, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov disc: rad too small" begin
            @test status == CPB.BARRIER_INFEASIBLE
        end
    end
end

lear = CPB.Learner{1}((2, 1), sys, iset, uset)

for method in _methods
    for PR in _PRs
        local status, mpf, gen = CPB.learn_lyapunov!(
            lear, 30, solver, solver, PR=PR, method=method
        )
        @testset "learn lyapunov disc: found" begin
            @test status == CPB.BARRIER_FOUND
        end
    end
end

nothing