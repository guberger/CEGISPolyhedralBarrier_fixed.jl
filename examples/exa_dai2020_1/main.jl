module ExampleIllustrative

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

box = Polyhedron()
CPB.add_halfspace!(box, [-1, 0], -2)
CPB.add_halfspace!(box, [1, 0], -2)
CPB.add_halfspace!(box, [0, -1], -2)
CPB.add_halfspace!(box, [0, 1], -2)

sys = System()

domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0.05)
CPB.add_halfspace!(domain, [0, 1], 0.05)
A = [-0.999 0; -0.139 0.341]
b = [0, 0]
CPB.add_piece!(sys, domain ∩ box, 1, A, b, 1)

domain = Polyhedron()
CPB.add_halfspace!(domain, [-1, 0], 0.05)
CPB.add_halfspace!(domain, [0, -1], 0.05)
A = [0.436 0.323; 0.388 -0.049]
b = [0, 0]
CPB.add_piece!(sys, domain ∩ box, 1, A, b, 1)

domain = Polyhedron()
CPB.add_halfspace!(domain, [1, 0], 0.05)
CPB.add_halfspace!(domain, [0, 1], 0.05)
A = [-0.457 0.215; 0.491 0.49]
b = [0, 0]
CPB.add_piece!(sys, domain ∩ box, 1, A, b, 1)

domain = Polyhedron()
CPB.add_halfspace!(domain, [1, 0], 0.05)
CPB.add_halfspace!(domain, [0, -1], 0.05)
A = [-0.022 0.344; 0.458 0.327141]
b = [0, 0]
CPB.add_piece!(sys, domain ∩ box, 1, A, b, 1)

iset = InitialSet{1}()
init_points = ([-0.5, -0.5],)
for point in init_points
    CPB.add_point!(iset, 1, point)
end

uset = UnsafeSet{1}()
udom = Polyhedron()
CPB.add_halfspace!(udom, [-1, 0], 1.8)
CPB.add_domain!(uset, 1, udom ∩ box)

# Illustration
fig = figure(0, figsize=(10, 8))
ax = fig.add_subplot(aspect="equal")
ax_ = (ax,)

xlims = (-2.2, 2.2)
ylims = (-2.2, 2.2)

for ax in ax_
    ax.set_xlim(xlims...)
    ax.set_ylim(ylims...)
    ax.set_xticks(-2:1:2)
    ax.set_yticks(-2:1:2)
    ax.tick_params(axis="both", labelsize=15)
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
lear = CPB.Learner{2}((3,), sys, iset, uset)
CPB.set_tol!(lear, :rad, 1e-1)
CPB.set_tol!(lear, :dom, 1e-8)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, Inf, solver, solver, PR=2, method=CPB.RadiusMax()
)

display(status)

for (loc, pf) in enumerate(mpf.pfs)
    plot_level!(ax_[loc], pf.afs, [(-10, -10), (10, 10)], fa=0.1, ew=0.5)
end

for evid in gen.pos_evids
    plot_point!(ax_[evid.loc], evid.point, mc="orange")
end

for evid in gen.lie_evids
    plot_point!(ax_[evid.loc], evid.point, mc="purple")
end

end # module