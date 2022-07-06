module ExampleSubwayMedium

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../src/CEGISPolyhedralBarrier.jl")
CPB = CEGISPolyhedralBarrier
Polyhedron = CPB.Polyhedron
System = CPB.System
InitialSet = CPB.InitialSet
UnsafeSet = CPB.UnsafeSet
State = CPB.State
Region = CPB.Region

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

## Parameters
nvar = 3
# vars = [#beacon, #second, delay]
nloc = 4
# locs = [ontime, late, onbrake, stopped]
_EYE_ = Matrix{Bool}(I, 3, 3)
_RES_ = [1 0 0; 0 1 0; 0 0 0]

Inv1 = Polyhedron()
CPB.add_halfspace!(Inv1, [-1, 0, 0], 0)
CPB.add_halfspace!(Inv1, [1, 0, 0], -40)
CPB.add_halfspace!(Inv1, [0, -1, 0], 0)
CPB.add_halfspace!(Inv1, [0, 1, 0], -40)
CPB.add_halfspace!(Inv1, [0, 0, -1], 0)
CPB.add_halfspace!(Inv1, [0, 0, 1], 0)
Inv2 = Polyhedron()
CPB.add_halfspace!(Inv2, [-1, 0, 0], 0)
CPB.add_halfspace!(Inv2, [1, 0, 0], -40)
CPB.add_halfspace!(Inv2, [0, -1, 0], 0)
CPB.add_halfspace!(Inv2, [0, 1, 0], -40)
CPB.add_halfspace!(Inv2, [0, 0, -1], 0)
CPB.add_halfspace!(Inv2, [0, 0, 1], -10)

sys = System()

# ontime -> ontime: time advance
domain = Polyhedron()
CPB.add_halfspace!(domain, [1, -1, 0], -10) # b ≤ s + 10
CPB.add_halfspace!(domain, [-1, 1, 0], -9) # b ≥ s - 9
b = [0, 1, 0]
CPB.add_piece!(sys, domain ∩ Inv1, 1, _EYE_, b, 1)

# ontime -> ontime: beacon advance
domain = Polyhedron()
CPB.add_halfspace!(domain, [1, -1, 0], -9) # b ≤ s + 9
CPB.add_halfspace!(domain, [-1, 1, 0], -10) # b ≥ s - 10
b = [1, 0, 0]
CPB.add_piece!(sys, domain ∩ Inv1, 1, _EYE_, b, 1)

# ontime -> late: run_late
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 1, 0], -10) # b = s - 10
CPB.add_halfspace!(domain, [1, -1, 0], 10)
b = [0, 0, 0]
CPB.add_piece!(sys, domain ∩ Inv1, 1, _EYE_, b, 2)

# late -> late: beacon advance
domain = Polyhedron()
CPB.add_halfspace!(domain, [1, -1, 0], 1) # b ≤ s - 1
b = [1, 0, 0]
CPB.add_piece!(sys, domain ∩ Inv1, 2, _EYE_, b, 2)

# late -> ontime: back_on_time
domain = Polyhedron()
CPB.add_halfspace!(domain, [1, -1, 0], 0) # b = s
CPB.add_halfspace!(domain, [-1, 1, 0], 0)
b = [0, 0, 0]
CPB.add_piece!(sys, domain ∩ Inv1, 2, _EYE_, b, 1)

# ontime -> onbrake: become_early
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 1, 0], 10) # b = s + 10
CPB.add_halfspace!(domain, [1, -1, 0], -10)
b = [0, 0, 0]
CPB.add_piece!(sys, domain ∩ Inv1, 1, _EYE_, b, 3)

# onbrake -> onbrake: time advance
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 1, 0], 1) # b ≥ s + 1
b = [0, 1, 0]
CPB.add_piece!(sys, domain ∩ Inv2, 3, _EYE_, b, 3)

# onbrake -> ontime: back_on_time
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 1, 0], 0) # b = s
CPB.add_halfspace!(domain, [1, -1, 0], 0)
b = [0, 0, 0]
CPB.add_piece!(sys, domain ∩ Inv2, 3, _RES_, b, 1)

# onbrake -> stopped: stop
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 1, 0], 0) # b ≥ s
CPB.add_halfspace!(domain, [0, 0, -1], 10) # d = 10
CPB.add_halfspace!(domain, [0, 0, 1], -10)
b = [0, 0, 0]
CPB.add_piece!(sys, domain ∩ Inv2, 3, _RES_, b, 4)

# stopped -> stopped: time advance
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 1, 0], 1) # b ≥ s + 1
b = [0, 1, 0]
CPB.add_piece!(sys, domain ∩ Inv1, 4, _EYE_, b, 4)

# stopped -> ontime: back_on_time
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 1, 0], 0) # b = s
CPB.add_halfspace!(domain, [1, -1, 0], 0)
b = [0, 0, 0]
CPB.add_piece!(sys, domain ∩ Inv1, 4, _EYE_, b, 1)

iset = InitialSet()
CPB.add_state!(iset, 1, [0, 0, 0])
CPB.add_state!(iset, 2, [0, 10, 0])
CPB.add_state!(iset, 3, [10, 0, 0])

uset = UnsafeSet()
udom = Polyhedron()
CPB.add_halfspace!(udom, [-1, 1, 0], 11)
CPB.add_region!(uset, 1, udom ∩ Inv1)
CPB.add_region!(uset, 2, udom ∩ Inv1)
CPB.add_region!(uset, 3, udom ∩ Inv2)
udom = Polyhedron()
CPB.add_halfspace!(udom, [1, -1, 0], 11)
CPB.add_region!(uset, 1, udom ∩ Inv1)
CPB.add_region!(uset, 2, udom ∩ Inv1)
CPB.add_region!(uset, 3, udom ∩ Inv2)

lear = CPB.Learner(nvar, nloc, sys, iset, uset, 0, 0)
CPB.set_tol!(lear, :rad, 1e-4)
CPB.set_tol!(lear, :dom, 1e-6)
CPB.set_param!(lear, :bigM, 1e3)

status, mpf, niter = CPB.learn_lyapunov!(lear, 1000, solver, solver)

# with unsafe set `|b - s| ≤ 15`: 176 iterations
# `tol_dom = 1e-6`

# with unsafe set `|b - s| ≤ 11`: 267 iterations
# `tol_dom = 1e-6`

display(status)

end # module