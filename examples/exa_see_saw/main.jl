module ExampleSeeSaw

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
nloc = 1

box = Polyhedron()
CPB.add_halfspace!(box, [-1, 0], -20)
CPB.add_halfspace!(box, [1, 0], -20)
CPB.add_halfspace!(box, [0, -1], -20)
CPB.add_halfspace!(box, [0, 1], -20)

sys = System()

domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 5)
CPB.add_halfspace!(domain, [1, 0], -7)
A = Matrix{Bool}(I, 2, 2)
b = [2, 1]
CPB.add_piece!(sys, domain ∩ box, 1, A, b, 1)

domain = Polyhedron()
CPB.add_halfspace!(domain, [1, 0], -4)
A = Matrix{Bool}(I, 2, 2)
b = [1, 2]
CPB.add_piece!(sys, domain ∩ box, 1, A, b, 1)

domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 7)
CPB.add_halfspace!(domain, [1, 0], -9)
A = Matrix{Bool}(I, 2, 2)
b = [1, 3]
CPB.add_piece!(sys, domain ∩ box, 1, A, b, 1)

domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 9)
A = Matrix{Bool}(I, 2, 2)
b = [2, 1]
CPB.add_piece!(sys, domain ∩ box, 1, A, b, 1)

iset = InitialSet()
CPB.add_state!(iset, 1, [0, 0])

uset = UnsafeSet()
udom = Polyhedron()
CPB.add_halfspace!(udom, [-1, 0], 1)
CPB.add_halfspace!(udom, [0, 1], 0)
CPB.add_region!(uset, 1, udom ∩ box)
udom = Polyhedron()
CPB.add_halfspace!(udom, [-1, 0], 10)
CPB.add_halfspace!(udom, [0, 1], -4)
CPB.add_region!(uset, 1, udom ∩ box)

## Plotting

# Illustration
fig = figure(0, figsize=(10, 8))
ax = fig.add_subplot(aspect="equal")
ax_ = (ax,)

xlims = (-22, 22)
ylims = (-22, 22)

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
    plot_vrep!(ax_[loc], points, fc="yellow", ec="black")
end

for region in uset.regions
    plot_hrep!(
        ax_[region.loc], region.domain.halfspaces, nothing, fc="red", ec="red"
    )
end

for piece in sys.pieces
    plot_hrep!(
        ax_[piece.loc1], piece.domain.halfspaces, nothing, fa=0.1
    )
end

## Learner
lear = CPB.Learner(nvar, nloc, sys, iset, uset, 0, 0)
CPB.set_tol!(lear, :rad, 1e-4)
CPB.set_tol!(lear, :bigM, 1e3)

tracerec = CPB.TraceRecorder()
status, mpf, niter = CPB.learn_lyapunov!(
    lear, 1000, solver, solver, tracerec=tracerec
)

display(status)

for loc = 1:nloc
    plot_level!(ax_[loc], mpf.pfs[loc].afs, [(-21, -21), (21, 21)])
end

fig = figure(1, figsize=(15, 8))
ax_ = fig.subplots(
    nrows=4, ncols=5,
    gridspec_kw=Dict("wspace"=>0.2, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal")
)

xlims = (-22, 22)
ylims = (-22, 22)

for ax in ax_
    ax.set_xlim(xlims...)
    ax.set_ylim(ylims...)
    ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)
end

iters = ceil.(Int, range(1, niter, length=20))

for (i, iter) in enumerate(iters)
    mpf = tracerec.mpf_list[iter]
    for loc = 1:nloc
        plot_level!(ax_[i], mpf.pfs[loc].afs, [(-21, -21), (21, 21)])
    end
    for evid in tracerec.pos_evids_list[iter]
        plot_point!(ax_[i], evid.point, mc="red")
    end
    for evid in tracerec.lie_evids_list[iter]
        plot_point!(ax_[i], evid.point1, mc="green")
        plot_point!(ax_[i], evid.point2, mc="orange")
    end
end

end # module