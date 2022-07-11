module ExampleIllustrative

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

mpf_inv = MultiPolyFunc{2,1}()
CPB.add_af!(mpf_inv, 1, SVector(-1.0, 0.0), -2.0)
CPB.add_af!(mpf_inv, 1, SVector(1.0, 0.0), -2.0)
CPB.add_af!(mpf_inv, 1, SVector(0.0, -1.0), -2.0)
CPB.add_af!(mpf_inv, 1, SVector(0.0, 1.0), -2.0)

sys = System{2}()

pf_dom = PolyFunc{2}()
CPB.add_af!(pf_dom, SVector(-1.0, 0.0), 0.05)
CPB.add_af!(pf_dom, SVector(0.0, 1.0), 0.05)
A = @SMatrix [-0.999 0.0; -0.139 0.341]
b = @SVector [0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

pf_dom = PolyFunc{2}()
CPB.add_af!(pf_dom, SVector(-1.0, 0.0), 0.05)
CPB.add_af!(pf_dom, SVector(0.0, -1.0), 0.05)
A = @SMatrix [0.436 0.323; 0.388 -0.049]
b = @SVector [0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

pf_dom = PolyFunc{2}()
CPB.add_af!(pf_dom, SVector(1.0, 0.0), 0.05)
CPB.add_af!(pf_dom, SVector(0.0, 1.0), 0.05)
A = @SMatrix [-0.457 0.215; 0.491 0.49]
b = @SVector [0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

pf_dom = PolyFunc{2}()
CPB.add_af!(pf_dom, SVector(1.0, 0.0), 0.05)
CPB.add_af!(pf_dom, SVector(0.0, -1.0), 0.05)
A = @SMatrix [-0.022 0.344; 0.458 0.271]
b = @SVector [0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

iset = PointSet{2,1}()
CPB.add_point!(iset, 1, SVector(-0.5, -0.5))

mpf_safe = MultiPolyFunc{2,1}()
CPB.add_af!(mpf_safe, 1, SVector(1.0, 0.0), -1.8)

# Illustration
fig = figure(0, figsize=(10, 8))
ax = fig.add_subplot(aspect="equal")
ax_ = (ax,)

xlims = (-2.2, 2.2)
ylims = (-2.2, 2.2)
lims = [(-40, -40), (40, 40)]

for ax in ax_
    ax.set_xlim(xlims...)
    ax.set_ylim(ylims...)
    ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)
end

for (loc, points) in enumerate(iset.points_list)
    for point in points
        plot_point!(ax_[loc], point, mc="gold")
    end
end

for (loc, pf) in enumerate(mpf_safe.pfs)
    plot_level!(ax_[loc], pf.afs, lims, fc="green", fa=0.1, ec="green")
end

for (loc, pf) in enumerate(mpf_inv.pfs)
    plot_level!(ax_[loc], pf.afs, lims, fc="none", ec="yellow")
end

for piece in sys.pieces
    plot_level!(
        ax_[piece.loc1], piece.pf_dom.afs, lims, fc="blue", fa=0.1, ec="blue"
    )
end

## Learner
lear = CPB.Learner((2,), sys, mpf_safe, mpf_inv, iset)
CPB.set_tol!(lear, :rad, 1e-1)
CPB.set_tol!(lear, :dom, 1e-8)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, Inf, solver, solver, PR=2, method=CPB.RadMax()
)

display(status)

for (loc, pf) in enumerate(mpf.pfs)
    plot_level!(ax_[loc], pf.afs, lims, fc="red", ec="red", fa=0.1, ew=0.5)
end

for evid in gen.pos_evids
    plot_point!(ax_[evid.loc], evid.point, mc="orange")
end

for evid in gen.neg_evids
    plot_point!(ax_[evid.loc], evid.point, mc="purple")
end

fig.savefig(string(
    @__DIR__, "/../figures/fig_exa_dai2020_1.png"
), dpi=200, transparent=false, bbox_inches="tight")

end # module