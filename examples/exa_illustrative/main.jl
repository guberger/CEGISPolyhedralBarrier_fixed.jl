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
PointSet = CPB.PointSet

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
CPB.add_halfspace!(domain, [0, -1], 0.5)
A = [0.5 0.0; 0.0 0.5]
b = [0, 0]
CPB.add_piece!(sys, domain ∩ box, 1, A, b, 2)

domain = Polyhedron()
CPB.add_halfspace!(domain, [1, 0], 0)
A = Matrix{Bool}(I, 2, 2)
b = [0.0, 0.5]
CPB.add_piece!(sys, domain ∩ box, 2, A, b, 1)

iset = PointSet{2}()
in_points = ([-1, -1], [-1, 1], [1, -1], [1, 1])
for point in in_points
    CPB.add_point!(iset, 1, point)
end

uset = PointSet{2}()
ex_points = ([-2, 1], [-2, 2], [-1, 1], [-1, 2])
for point in ex_points
    CPB.add_point!(uset, 2, point)
end

# Illustration
fig = figure(0, figsize=(15, 8))
ax_ = fig.subplots(
    nrows=1, ncols=2,
    gridspec_kw=Dict("wspace"=>0.2, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal")
)

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
    plot_chull!(ax_[loc], points, fc="yellow", ec="yellow")
end

for (loc, points) in enumerate(uset.points_list)
    for point in points
        plot_point!(ax_[loc], point, mc="red")
    end
    plot_chull!(ax_[loc], points, fc="red", ec="red")
end

for piece in sys.pieces
    plot_interior!(
        ax_[piece.loc1], piece.domain.halfspaces, nothing, fa=0.1
    )
end

## Learner
lear = CPB.Learner{2}((1, 1), sys, iset, uset)
CPB.set_tol!(lear, :domain, 1e-8)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, Inf, solver, solver, method=CPB.ObjMin()
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

fig.savefig(string(
    @__DIR__, "/../figures/fig_exa_illustrative.png"
), dpi=200, transparent=false, bbox_inches="tight")

end # module