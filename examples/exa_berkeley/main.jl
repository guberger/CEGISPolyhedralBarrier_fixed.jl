module ExampleSubwayHard

using LinearAlgebra
using StaticArrays
using JuMP
using Gurobi
using PyPlot

include("../../src/CEGISPolyhedralBarrier.jl")
CPB = CEGISPolyhedralBarrier
System = CPB.System
PointSet = CPB.PointSet
PolyFunc = CPB.PolyFunc
MultiPolyFunc = CPB.MultiPolyFunc

include("../utils/plotting2D.jl")

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

# vars = [invalid, unowned, nonexclusive, exclusive]

mpf_inv = MultiPolyFunc{4,1}()
CPB.add_af!(mpf_inv, 1, SVector(-1.0, 0.0, 0.0, 0.0), -10.0)
CPB.add_af!(mpf_inv, 1, SVector(1.0, 0.0, 0.0, 0.0), -10.0)
CPB.add_af!(mpf_inv, 1, SVector(0.0, -1.0, 0.0, 0.0), -10.0)
CPB.add_af!(mpf_inv, 1, SVector(0.0, 1.0, 0.0, 0.0), -10.0)
CPB.add_af!(mpf_inv, 1, SVector(0.0, 0.0, -1.0, 0.0), -10.0)
CPB.add_af!(mpf_inv, 1, SVector(0.0, 0.0, 1.0, 0.0), -10.0)
CPB.add_af!(mpf_inv, 1, SVector(0.0, 0.0, 0.0, -1.0), -10.0)
CPB.add_af!(mpf_inv, 1, SVector(0.0, 0.0, 0.0, 1.0), -10.0)

sys = System{4}()

pf_dom = PolyFunc{4}()
CPB.add_af!(pf_dom, SVector(-1.0, 0.0, 0.0, 0.0), 1.0) # i ≥ 1
A = SMatrix{4,4,Float64}([
    1 0 0 0;
    0 1 0 0;
    0 0 1 1;
    0 0 0 0
])
b = SVector{4,Float64}([-1, 1, 0, 1])
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

pf_dom = PolyFunc{4}()
CPB.add_af!(pf_dom, SVector(0.0, -1.0, -1.0, 0.0), 1.0) # u + n ≥ 1
A = SMatrix{4,4,Float64}([
    1 1 1 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 1
])
b = SVector{4,Float64}([-1, 0, 0, 1])
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

pf_dom = PolyFunc{4}()
CPB.add_af!(pf_dom, SVector(-1.0, 0.0, 0.0, 0.0), 1.0) # i ≥ 1
A = SMatrix{4,4,Float64}([
    1 1 1 1;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0
])
b = SVector{4,Float64}([-1, 0, 0, 1])
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

iset = PointSet{4,1}()
CPB.add_point!(iset, 1, SVector(1.0, 0.0, 0.0, 0.0))

mpf_safe = MultiPolyFunc{4,1}()
CPB.add_af!(mpf_safe, 1, SVector(-1.0, -1.0, -1.0, -1.0), 0.0)

## Learner
lear = CPB.Learner((1,), sys, mpf_safe, mpf_inv, iset)
CPB.set_tol!(lear, :rad, 1e-2)
CPB.set_tol!(lear, :dom, 1e-8)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, Inf, solver, solver, PR="full", method=CPB.RadMax()
)

display(status)

foreach(pf -> display(pf), mpf.pfs)

end # module