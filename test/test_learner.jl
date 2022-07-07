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
InitialSet = CPB.InitialSet
UnsafeSet = CPB.UnsafeSet
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

## Learner Disc
sys = System()

domain = Polyhedron()
CPB.add_halfspace!(domain, [-1], 0)
CPB.add_halfspace!(domain, [1], -2)
A = reshape([-1], 1, 1)
b = [1]
CPB.add_piece!(sys, domain, 1, A, b, 2)

iset = InitialSet{2}()
CPB.add_point!(iset, 1, [-1])
CPB.add_point!(iset, 1, [1])

uset = UnsafeSet{2}()
udom = Polyhedron()
CPB.add_halfspace!(udom, [-1], 1.1)
CPB.add_domain!(uset, 2, udom)

lear = CPB.Learner{1}((10, 10), sys, iset, uset)
CPB.set_tol!(lear, :rad, 10)
CPB.set_param!(lear, :xmax, 1e2)

@testset "set tol and param" begin
    @test_throws AssertionError CPB.set_tol!(lear, :dumb, 0)
    @test_throws AssertionError CPB.set_param!(lear, :dumb, 0)
    @test lear.tols[:rad] ≈ 10
    @test lear.params[:xmax] ≈ 100
end

lear = CPB.Learner{1}((10, 10), sys, iset, uset)
status, mpf, gen = CPB.learn_lyapunov!(lear, 1, solver, solver)

@testset "learn lyapunov disc: max iter" begin
    @test status == CPB.MAX_ITER_REACHED
    @test length(gen.pos_evids) + length(gen.lie_evids) == 0
end

lear = CPB.Learner{1}((0, 1), sys, iset, uset)
status, mpf, gen = CPB.learn_lyapunov!(lear, 30, solver, solver, PR=2)

@testset "learn lyapunov disc: found" begin
    @test status == CPB.BARRIER_FOUND
end

sys = System()

domain = Polyhedron()
CPB.add_halfspace!(domain, [1], 0)
A = reshape([-1], 1, 1)
b = [0]
CPB.add_piece!(sys, domain, 1, A, b, 2)

lear = CPB.Learner{1}((0, 1), sys, iset, uset)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, 30, solver, solver, method=CPB.DepthMin()
)

@testset "learn lyapunov disc: rad too small" begin
    @test status == CPB.BARRIER_INFEASIBLE
end

lear = CPB.Learner{1}((1, 1), sys, iset, uset)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, 30, solver, solver, method=CPB.DepthMin()
)

@testset "learn lyapunov disc: found" begin
    @test status == CPB.BARRIER_FOUND
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

lear = CPB.Learner{1}((1, 1), sys, iset, uset)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, 30, solver, solver, PR="none", method=CPB.RadiusMax()
)

@testset "learn lyapunov disc: rad too small" begin
    @test status == CPB.BARRIER_INFEASIBLE
end

lear = CPB.Learner{1}((2, 1), sys, iset, uset)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, 30, solver, solver, PR="full", method=CPB.RadiusMax()
)

@testset "learn lyapunov disc: found" begin
    @test status == CPB.BARRIER_FOUND
end

nothing