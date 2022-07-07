module ExampleSubwayEasy

using LinearAlgebra
using JuMP
using Gurobi
using PyPlot

include("../../src/CEGISPolyhedralBarrier.jl")
CPB = CEGISPolyhedralBarrier
Polyhedron = CPB.Polyhedron
PolyFunc = CPB.PolyFunc
System = CPB.System
InitialSet = CPB.InitialSet
UnsafeSet = CPB.UnsafeSet

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

iset = InitialSet{3}()
CPB.add_point!(iset, 1, [0, 0])
CPB.add_point!(iset, 2, [0, 10])
CPB.add_point!(iset, 3, [10, 0])

uset = UnsafeSet{3}()
udom = Polyhedron()
CPB.add_halfspace!(udom, [-1, 1], 11)
CPB.add_domain!(uset, 1, udom ∩ box)
# CPB.add_domain!(uset, 2, udom ∩ box)
CPB.add_domain!(uset, 3, udom ∩ box)
udom = Polyhedron()
CPB.add_halfspace!(udom, [1, -1], 11)
CPB.add_domain!(uset, 1, udom ∩ box)
CPB.add_domain!(uset, 2, udom ∩ box)
CPB.add_domain!(uset, 3, udom ∩ box)

# Illustration
fig = figure(0, figsize=(15, 8))
ax_ = fig.subplots(
    nrows=1, ncols=3,
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

for (loc, points) in enumerate(iset.points_list)
    for point in points
        plot_point!(ax_[loc], point, mc="gold")
    end
    plot_vrep!(ax_[loc], points, fc="yellow", ec="yellow")
end

for (loc, domains) in enumerate(uset.domains_list)
    for domain in domains
        plot_hrep!(
            ax_[loc], domain.halfspaces, nothing, fc="red", ec="red"
        )
    end
end

for piece in sys.pieces
    plot_hrep!(
        ax_[piece.loc1], piece.domain.halfspaces, nothing, fa=0.1
    )
end

## Learner
lear = CPB.Learner{2}((2, 1, 2), sys, iset, uset)
CPB.set_tol!(lear, :rad, 1e-3)
CPB.set_tol!(lear, :dom, 1e-8)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, Inf, solver, solver, PR=500, method=CPB.RandLeaf()
)

display(status)

for (loc, pf) in enumerate(mpf.pfs)
    plot_level!(ax_[loc], pf.afs, [(-21, -21), (21, 21)], fa=0.1, ew=0.5)
end

for evid in gen.pos_evids
    plot_point!(ax_[evid.loc], evid.point, mc="orange")
end

for evid in gen.lie_evids
    plot_point!(ax_[evid.loc], evid.point, mc="purple")
end

end # module