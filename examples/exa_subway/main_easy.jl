module ExampleSubwayEasy

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../src/CEGISPolyhedralBarrier.jl")
CPB = CEGISPolyhedralBarrier
Halfspace = CPB.Halfspace
AffForm = CPB.AffForm
Point = CPB.Point
Polyhedron = CPB.Polyhedron
PolyFunc = CPB.PolyFunc
System = CPB.System
InitialSet = CPB.InitialSet
UnsafeSet = CPB.UnsafeSet
State = CPB.State
Region = CPB.Region

include("../utils/plotting2D.jl")

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

## Parameters
nvar = 2
# vars = [#beacon, #second]
nloc = 3
# locs = [ontime, late, onbrake]
_EYE_ = Matrix{Bool}(I, 2, 2)

box = Polyhedron()
CPB.add_halfspace!(box, [-1, 0], -20)
CPB.add_halfspace!(box, [1, 0], -20)
CPB.add_halfspace!(box, [0, -1], -20)
CPB.add_halfspace!(box, [0, 1], -20)

sys = System()

# ontime -> ontime: time advance
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0)
CPB.add_halfspace!(domain, [0, -1], 0)
CPB.add_halfspace!(domain, [1, -1], -10) # b ≤ s + 10
CPB.add_halfspace!(domain, [-1, 1], -9) # b ≥ s - 9
b = [0, 1]
CPB.add_piece!(sys, domain ∩ box, 1, _EYE_, b, 1)

# ontime -> ontime: beacon advance
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0)
CPB.add_halfspace!(domain, [0, -1], 0)
CPB.add_halfspace!(domain, [1, -1], -9) # b ≤ s + 9
CPB.add_halfspace!(domain, [-1, 1], -10) # b ≥ s - 10
b = [1, 0]
CPB.add_piece!(sys, domain ∩ box, 1, _EYE_, b, 1)

# ontime -> late: run_late
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0)
CPB.add_halfspace!(domain, [0, -1], 0)
CPB.add_halfspace!(domain, [-1, 1], -10) # b = s - 10
CPB.add_halfspace!(domain, [1, -1], 10)
b = [0, 0]
CPB.add_piece!(sys, domain ∩ box, 1, _EYE_, b, 2)

# late -> late: beacon advance
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0)
CPB.add_halfspace!(domain, [0, -1], 0)
CPB.add_halfspace!(domain, [1, -1], 1) # b ≤ s - 1
b = [1, 0]
CPB.add_piece!(sys, domain ∩ box, 2, _EYE_, b, 2)

# late -> ontime: back_on_time
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0)
CPB.add_halfspace!(domain, [0, -1], 0)
CPB.add_halfspace!(domain, [1, -1], 0) # b = s
CPB.add_halfspace!(domain, [-1, 1], 0)
b = [0, 0]
CPB.add_piece!(sys, domain ∩ box, 2, _EYE_, b, 1)

# ontime -> onbrake: become_early
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0)
CPB.add_halfspace!(domain, [0, -1], 0)
CPB.add_halfspace!(domain, [-1, 1], 10) # b = s + 10
CPB.add_halfspace!(domain, [1, -1], -10)
b = [0, 0]
CPB.add_piece!(sys, domain ∩ box, 1, _EYE_, b, 3)

# onbrake -> onbrake: time advance
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0)
CPB.add_halfspace!(domain, [0, -1], 0)
CPB.add_halfspace!(domain, [-1, 1], 1) # b ≥ s + 1
b = [0, 1]
CPB.add_piece!(sys, domain ∩ box, 3, _EYE_, b, 3)

# onbrake -> ontime: back_on_time
domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0)
CPB.add_halfspace!(domain, [0, -1], 0)
CPB.add_halfspace!(domain, [-1, 1], 0) # b = s
CPB.add_halfspace!(domain, [1, -1], 0)
b = [0, 0]
CPB.add_piece!(sys, domain ∩ box, 3, _EYE_, b, 1)

iset = InitialSet()
CPB.add_state!(iset, 1, [0, 0])
CPB.add_state!(iset, 2, [0, 10])
CPB.add_state!(iset, 3, [10, 0])

uset = UnsafeSet()
udom = Polyhedron()
CPB.add_halfspace!(udom, [-1, 1], 11)
CPB.add_region!(uset, 1, udom ∩ box)
CPB.add_region!(uset, 2, udom ∩ box)
CPB.add_region!(uset, 3, udom ∩ box)
udom = Polyhedron()
CPB.add_halfspace!(udom, [1, -1], 11)
CPB.add_region!(uset, 1, udom ∩ box)
CPB.add_region!(uset, 2, udom ∩ box)
CPB.add_region!(uset, 3, udom ∩ box)

# Illustration
fig = figure(0, figsize=(15, 8))
ax_ = fig.subplots(
    nrows=1, ncols=3,
    gridspec_kw=Dict("wspace"=>0.2, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal")
)

xlims = (-20, 20)
ylims = (-20, 20)

for ax in ax_
    ax.set_xlim(xlims...)
    ax.set_ylim(ylims...)
    ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)
end

for state in iset.states
    plot_point!(ax_[state.loc], state.point, mc="gold")
end
for loc = 1:nloc
    points = [state.point for state in iset.states if state.loc == loc]
    plot_vrep!(ax_[loc], points, fc="yellow", ec="yellow")
end

for region in uset.regions
    plot_hrep!(
        ax_[region.loc], region.domain.halfspaces, nothing, fc="red", ec="red"
    )
end

for piece in sys.pieces
    plot_hrep!(
        ax_[piece.loc1], piece.domain.halfspaces, nothing,
        fa=0.25, fc="green", ew=0.5
    )
end

lear = CPB.Learner(nvar, nloc, sys, iset, uset, 0, 0)
CPB.set_tol!(lear, :rad, 1e-4)
CPB.set_param!(lear, :bigM, 1e3)

status, mpf, niter = CPB.learn_lyapunov!(lear, 1000, solver, solver)

display(status)

for loc = 1:nloc
    plot_level!(ax_[loc], mpf.pfs[loc].afs, [(-21, -21), (21, 21)])
end

end # module