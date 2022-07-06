module ExampleBerkeley

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
nvar = 4
# vars = [invalid, unowned, nonexclusive, exclusive]
nloc = 1

Inv = Polyhedron()
CPB.add_halfspace!(Inv, [-1, 0, 0, 0], -10)
CPB.add_halfspace!(Inv, [1, 0, 0, 0], -10)
CPB.add_halfspace!(Inv, [0, -1, 0, 0], -10)
CPB.add_halfspace!(Inv, [0, 1, 0, 0], -10)
CPB.add_halfspace!(Inv, [0, 0, -1, 0], -10)
CPB.add_halfspace!(Inv, [0, 0, 1, 0], -10)
CPB.add_halfspace!(Inv, [0, 0, 0, -1], -10)
CPB.add_halfspace!(Inv, [0, 0, 0, 1], -10)

sys = System()

domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0, 0, 0], 1) # i ≥ 1
A = [1 0 0 0; 0 1 0 0; 0 0 1 1; 0 0 0 0]
b = [-1, 1, 0, 0]
CPB.add_piece!(sys, domain ∩ Inv, 1, A, b, 1)

domain = Polyhedron()
CPB.add_halfspace!(domain, [0, -1, -1, 0], 1) # u + n ≥ 1
A = [1 1 1 0; 0 0 0 0; 0 0 0 0; 0 0 0 1]
b = [-1, 0, 0, 1]
CPB.add_piece!(sys, domain ∩ Inv, 1, A, b, 1)

domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0, 0, 0], 1) # i ≥ 1
A = [1 1 1 1; 0 0 0 0; 0 0 0 0; 0 0 0 0]
b = [-1, 0, 0, 1]
CPB.add_piece!(sys, domain ∩ Inv, 1, A, b, 1)

iset = InitialSet()
CPB.add_state!(iset, 1, [0, 0, 0, 1])
CPB.add_state!(iset, 1, [0, 0, 0, 10])

uset = UnsafeSet()

udom = Polyhedron()
CPB.add_halfspace!(udom, [1, 0, 0, 0], 0.5)
CPB.add_region!(uset, 1, udom ∩ Inv)

udom = Polyhedron()
CPB.add_halfspace!(udom, [0, 1, 0, 0], 0.5)
CPB.add_region!(uset, 1, udom ∩ Inv)

udom = Polyhedron()
CPB.add_halfspace!(udom, [0, 0, 1, 0], 0.5)
CPB.add_region!(uset, 1, udom ∩ Inv)

udom = Polyhedron()
CPB.add_halfspace!(udom, [0, 0, 0, 1], -0.5)
CPB.add_region!(uset, 1, udom ∩ Inv)

udom = Polyhedron()
CPB.add_halfspace!(udom, [1, 1, 1, 1], -0.5)
CPB.add_region!(uset, 1, udom ∩ Inv)

lear = CPB.Learner(nvar, nloc, sys, iset, uset, 0, 0)
CPB.set_tol!(lear, :rad, 1e-4)
CPB.set_tol!(lear, :dom, 1e-6)
CPB.set_param!(lear, :bigM, 1e3)

status, mpf, niter = CPB.learn_lyapunov!(lear, 1000, solver, solver)

display(status)

display(mpf)

end # module