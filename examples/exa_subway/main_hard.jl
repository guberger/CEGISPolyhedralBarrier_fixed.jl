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

# vars = [#beacon, #second, delay]
# locs = [ontime, late, onbrake, stopped]
_EYE_ = SMatrix{3,3}(1.0*I)
_RES_ = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0]

mpf_inv = MultiPolyFunc{3,4}()
for loc in (1, 2, 4)
    CPB.add_af!(mpf_inv, loc, SVector(-1.0, 0.0, 0.0), -0.0)
    CPB.add_af!(mpf_inv, loc, SVector(1.0, 0.0, 0.0), -100.0)
    CPB.add_af!(mpf_inv, loc, SVector(0.0, -1.0, 0.0), -0.0)
    CPB.add_af!(mpf_inv, loc, SVector(0.0, 1.0, 0.0), -100.0)
    CPB.add_af!(mpf_inv, loc, SVector(0.0, 0.0, -1.0), 0.0)
    CPB.add_af!(mpf_inv, loc, SVector(0.0, 0.0, 1.0), 0.0)
end
CPB.add_af!(mpf_inv, 3, SVector(-1.0, 0.0, 0.0), -0.0)
CPB.add_af!(mpf_inv, 3, SVector(1.0, 0.0, 0.0), -100.0)
CPB.add_af!(mpf_inv, 3, SVector(0.0, -1.0, 0.0), -0.0)
CPB.add_af!(mpf_inv, 3, SVector(0.0, 1.0, 0.0), -100.0)
CPB.add_af!(mpf_inv, 3, SVector(0.0, 0.0, -1.0), 0.0)
CPB.add_af!(mpf_inv, 3, SVector(0.0, 0.0, 1.0), -10.0)

sys = System{3}()

# ontime -> ontime: time advance
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(1.0, -1.0, 0.0), -10.0) # b ≤ s + 10
CPB.add_af!(pf_dom, SVector(-1.0, 1.0, 0.0), -9.0) # b ≥ s - 9
b = @SVector [0.0, 1.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, _EYE_, b, 1)

# ontime -> ontime: beacon advance
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(1.0, -1.0, 0.0), -9.0) # b ≤ s + 9
CPB.add_af!(pf_dom, SVector(-1.0, 1.0, 0.0), -10.0) # b ≥ s - 10
b = @SVector [1.0, 0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, _EYE_, b, 1)

# ontime -> late: run_late
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(1.0, -1.0, 0.0), 10.0) # b = s - 10
CPB.add_af!(pf_dom, SVector(-1.0, 1.0, 0.0), -10.0)
b = @SVector [0.0, 0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, _EYE_, b, 2)

# late -> late: beacon advance
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(1.0, -1.0, 0.0), 1.0) # b ≤ s - 1
b = @SVector [1.0, 0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 2, _EYE_, b, 2)

# late -> ontime: back_on_time
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(1.0, -1.0, 0.0), 0.0) # b == s
CPB.add_af!(pf_dom, SVector(-1.0, 1.0, 0.0), 0.0)
b = @SVector [0.0, 0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 2, _EYE_, b, 1)

# ontime -> onbrake: become_early
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(1.0, -1.0, 0.0), -10.0) # b = s + 10
CPB.add_af!(pf_dom, SVector(-1.0, 1.0, 0.0), 10.0)
b = @SVector [0.0, 0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, _EYE_, b, 3)

# onbrake -> onbrake: time advance
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(-1.0, 1.0, 0.0), 1.0) # b ≥ s + 1
b = @SVector [0.0, 1.0, 0.0]
CPB.add_piece!(sys, pf_dom, 3, _EYE_, b, 3)

# onbrake -> onbrake: beacon and delay advance
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(-1.0, 1.0, 0.0), 0.0) # b ≥ s
CPB.add_af!(pf_dom, SVector(0.0, 0.0, 1.0), -9.0) # d ≤ 9
b = @SVector [1.0, 0.0, 1.0]
CPB.add_piece!(sys, pf_dom, 3, _EYE_, b, 3)

# onbrake -> ontime: back_on_time
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(1.0, -1.0, 0.0), 0.0) # b == s
CPB.add_af!(pf_dom, SVector(-1.0, 1.0, 0.0), 0.0)
b = @SVector [0.0, 0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 3, _RES_, b, 1)

# onbrake -> stopped: stop
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(-1.0, 1.0, 0.0), 0.0) # b ≥ s
CPB.add_af!(pf_dom, SVector(0.0, 0.0, 1.0), -10.0) # d == 10
CPB.add_af!(pf_dom, SVector(0.0, 0.0, -1.0), 10.0)
b = @SVector [0.0, 0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 3, _RES_, b, 4)

# stopped -> stopped: time advance
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(-1.0, 1.0, 0.0), 1.0) # b ≥ s + 1
b = @SVector [0.0, 1.0, 0.0]
CPB.add_piece!(sys, pf_dom, 4, _EYE_, b, 4)

# stopped -> ontime: back_on_time
pf_dom = PolyFunc{3}()
CPB.add_af!(pf_dom, SVector(1.0, -1.0, 0.0), 0.0) # b == s
CPB.add_af!(pf_dom, SVector(-1.0, 1.0, 0.0), 0.0)
b = @SVector [0.0, 0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 4, _EYE_, b, 1)

iset = PointSet{3,4}()
CPB.add_point!(iset, 1, SVector(1.0, 0.0, 0.0))
CPB.add_point!(iset, 1, SVector(0.0, 1.0, 0.0))
CPB.add_point!(iset, 2, SVector(0.0, 10.0, 0.0))
CPB.add_point!(iset, 3, SVector(10.0, 0.0, 0.0))

mpf_safe = MultiPolyFunc{3,4}()
for loc = 1:4
    CPB.add_af!(mpf_safe, loc, SVector(-1.0, 1.0, 0.0), -21.0)
    CPB.add_af!(mpf_safe, loc, SVector(1.0, -1.0, 0.0), -21.0)
end

## Learner
lear = CPB.Learner((0, 0, 1, 0), sys, mpf_safe, mpf_inv, iset)
CPB.set_tol!(lear, :rad, 1e-4)
CPB.set_tol!(lear, :dom, 1e-8)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, Inf, solver, solver, PR="full", method=CPB.DepthMin()
)

display(status)

foreach(pf -> display(pf), mpf.pfs)

end # module