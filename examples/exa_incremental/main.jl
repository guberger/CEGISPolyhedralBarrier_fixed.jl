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
CPB.add_af!(pf_dom, SVector(-1.0, 0.0), 0.0)
CPB.add_af!(pf_dom, SVector(0.0, 1.0), 0.0)
A = @SMatrix [-0.999 0.0; -0.139 0.341]
b = @SVector [0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

pf_dom = PolyFunc{2}()
CPB.add_af!(pf_dom, SVector(-1.0, 0.0), 0.0)
CPB.add_af!(pf_dom, SVector(0.0, -1.0), 0.0)
A = @SMatrix [0.436 0.323; 0.388 -0.049]
b = @SVector [0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

pf_dom = PolyFunc{2}()
CPB.add_af!(pf_dom, SVector(1.0, 0.0), 0.0)
CPB.add_af!(pf_dom, SVector(0.0, 1.0), 0.0)
A = @SMatrix [-0.457 0.215; 0.491 0.49]
b = @SVector [0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

pf_dom = PolyFunc{2}()
CPB.add_af!(pf_dom, SVector(1.0, 0.0), 0.0)
CPB.add_af!(pf_dom, SVector(0.0, -1.0), 0.0)
A = @SMatrix [-0.022 0.344; 0.458 0.271]
b = @SVector [0.0, 0.0]
CPB.add_piece!(sys, pf_dom, 1, A, b, 1)

iset = PointSet{2,1}()
CPB.add_point!(iset, 1, SVector(-1.0, 0.0))
CPB.add_point!(iset, 1, SVector(1.0, 0.0))
CPB.add_point!(iset, 1, SVector(0.0, -1.0))
CPB.add_point!(iset, 1, SVector(0.0, 1.0))

# Illustration
fig = figure(0, figsize=(10, 8))
ax_ = fig.subplots(
    nrows=2, ncols=2,
    gridspec_kw=Dict("wspace"=>0.2, "hspace"=>0.1),
    subplot_kw=Dict("aspect"=>"equal")
)

xlims = (-2.2, 2.2)
ylims = (-2.2, 2.2)
lims = [(-40, -40), (40, 40)]

for ax in ax_
    ax.set_xlim(xlims...)
    ax.set_ylim(ylims...)
    ax.plot(0, 0, marker="x", ms=10, c="black", mew=2.5)
    for point in iset.points_list[1]
        plot_point!(ax, point, mc="gold")
    end
    for piece in sys.pieces
        plot_level!(
            ax, piece.pf_dom.afs, lims, fc="blue", fa=0.1, ec="blue"
        )
    end
end

# Trajectory
nstep = 20
set1 = PointSet{2,1}()
for point in iset.points_list[1]
    CPB.add_point!(set1, 1, point)
end
set2 = PointSet{2,1}()
set_traj = PointSet{2,1}()

for step = 1:nstep
    global set1, set2
    empty!(set2)
    CPB.compute_post!(set2, set1, sys, 1e-8)
    for point in set2.points_list[1]
        CPB.add_point!(set_traj, 1, point)
    end
    set1, set2 = set2, set1
end

for point in set_traj.points_list[1]
    plot_point!(ax_[1], point, mc="black", ms=2.5)
end

## Learner

# Step 1
mpf_safe = MultiPolyFunc{2,1}()
CPB.add_af!(mpf_safe, 1, SVector(-1.0, 1.0), -1.8)

plot_level!(ax_[1], mpf_safe.pfs[1].afs, lims, fc="green", fa=0.1, ec="green")
plot_level!(ax_[1], mpf_inv.pfs[1].afs, lims, fc="none", ec="yellow")

lear = CPB.Learner((1,), sys, mpf_safe, mpf_inv, iset)
CPB.set_tol!(lear, :rad, 1e-4)
CPB.set_tol!(lear, :dom, 1e-8)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, Inf, solver, solver, PR=100, method=CPB.RadMax()
)
@assert status == CPB.BARRIER_FOUND

plot_level!(ax_[1], mpf.pfs[1].afs, lims, fc="red", ec="red", fa=0.1, ew=0.5)
for evid in gen.pos_evids
    plot_point!(ax_[1], evid.point, mc="orange")
end
for evid in gen.neg_evids
    plot_point!(ax_[1], evid.point, mc="purple")
end

# Step 2
for af in Iterators.flatten((mpf.pfs[1].afs, mpf_safe.pfs[1].afs))
    CPB.add_af!(mpf_inv, 1, af)
end

mpf_safe = MultiPolyFunc{2,1}()
CPB.add_af!(mpf_safe, 1, SVector(1.0, -1.0), -1.8)

plot_level!(ax_[2], mpf_safe.pfs[1].afs, lims, fc="green", fa=0.1, ec="green")
plot_level!(ax_[2], mpf_inv.pfs[1].afs, lims, fc="none", ec="yellow")

lear = CPB.Learner((1,), sys, mpf_safe, mpf_inv, iset)
CPB.set_tol!(lear, :rad, 1e-4)
CPB.set_tol!(lear, :dom, 1e-8)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, Inf, solver, solver, PR=100, method=CPB.RadMax()
)
@assert status == CPB.BARRIER_FOUND

plot_level!(ax_[2], mpf.pfs[1].afs, lims, fc="red", ec="red", fa=0.1, ew=0.5)
for evid in gen.pos_evids
    plot_point!(ax_[2], evid.point, mc="orange")
end
for evid in gen.neg_evids
    plot_point!(ax_[2], evid.point, mc="purple")
end

# Step 3
for af in Iterators.flatten((mpf.pfs[1].afs, mpf_safe.pfs[1].afs))
    CPB.add_af!(mpf_inv, 1, af)
end

mpf_safe = MultiPolyFunc{2,1}()
CPB.add_af!(mpf_safe, 1, SVector(-1.0, -1.0), -1.8)

plot_level!(ax_[3], mpf_safe.pfs[1].afs, lims, fc="green", fa=0.1, ec="green")
plot_level!(ax_[3], mpf_inv.pfs[1].afs, lims, fc="none", ec="yellow")

lear = CPB.Learner((1,), sys, mpf_safe, mpf_inv, iset)
CPB.set_tol!(lear, :rad, 1e-4)
CPB.set_tol!(lear, :dom, 1e-8)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, Inf, solver, solver, PR=100, method=CPB.RadMax()
)
@assert status == CPB.BARRIER_FOUND

plot_level!(ax_[3], mpf.pfs[1].afs, lims, fc="red", ec="red", fa=0.1, ew=0.5)
for evid in gen.pos_evids
    plot_point!(ax_[3], evid.point, mc="orange")
end
for evid in gen.neg_evids
    plot_point!(ax_[3], evid.point, mc="purple")
end

# Step 4
for af in Iterators.flatten((mpf.pfs[1].afs, mpf_safe.pfs[1].afs))
    CPB.add_af!(mpf_inv, 1, af)
end

mpf_safe = MultiPolyFunc{2,1}()
CPB.add_af!(mpf_safe, 1, SVector(1.0, 1.0), -1.8)

plot_level!(ax_[4], mpf_safe.pfs[1].afs, lims, fc="green", fa=0.1, ec="green")
plot_level!(ax_[4], mpf_inv.pfs[1].afs, lims, fc="none", ec="yellow")

lear = CPB.Learner((1,), sys, mpf_safe, mpf_inv, iset)
CPB.set_tol!(lear, :rad, 1e-4)
CPB.set_tol!(lear, :dom, 1e-8)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, Inf, solver, solver, PR=100, method=CPB.RadMax()
)
@assert status == CPB.BARRIER_FOUND

plot_level!(ax_[4], mpf.pfs[1].afs, lims, fc="red", ec="red", fa=0.1, ew=0.5)
for evid in gen.pos_evids
    plot_point!(ax_[4], evid.point, mc="orange")
end
for evid in gen.neg_evids
    plot_point!(ax_[4], evid.point, mc="purple")
end

# Verif
for af in Iterators.flatten((mpf.pfs[1].afs, mpf_safe.pfs[1].afs))
    CPB.add_af!(mpf_inv, 1, af)
end

mpf_safe = mpf_inv
mpf_inv = MultiPolyFunc{2,1}()

plot_level!(ax_[4], mpf_safe.pfs[1].afs, lims, fc="none", fa=0.1, ec="black")

lear = CPB.Learner((0,), sys, mpf_safe, mpf_inv, iset)
CPB.set_tol!(lear, :rad, 1e-4)
CPB.set_tol!(lear, :dom, 1e-8)
status, mpf, gen = CPB.learn_lyapunov!(
    lear, Inf, solver, solver, PR="full", method=CPB.RadMax()
)
@assert status == CPB.BARRIER_FOUND

# Figure
fig.savefig(string(
    @__DIR__, "/../figures/fig_exa_dai2020_1.png"
), dpi=200, transparent=false, bbox_inches="tight")

end # module