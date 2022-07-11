include("geometry.jl")

function plot_level!(ax, afs, lims; fc="green", fa=0.5, ec="green", ew=1.0)
    A = zeros(length(afs), 2)
    b = zeros(length(afs))
    for (i, af) in enumerate(afs)
        A[i, 1], A[i, 2] = (af.lin...,)
        b[i] = -af.off
    end
    _plot_hrep!(ax, A, b, lims, fc, fa, ec, ew)
end

function _plot_hrep!(ax, A, b, lims, fc, fa, ec, ew)
    if !isnothing(lims)
        A = vcat(A, -Matrix{Bool}(I, 2, 2), Matrix{Bool}(I, 2, 2))
        b = vcat(b, -collect(lims[1]), collect(lims[2]))
    end
    verts = compute_vertices_hrep(A, b)
    isempty(verts) && return
    polylist = matplotlib.collections.PolyCollection((verts,))
    fca = matplotlib.colors.colorConverter.to_rgba(fc, alpha=fa)
    polylist.set_facecolor(fca)
    polylist.set_edgecolor(ec)
    polylist.set_linewidth(ew)
    ax.add_collection(polylist)
end

function plot_chull!(ax, points; fc="blue", fa=0.5, ec="blue", ew=1.0)
    isempty(points) && return
    P = zeros(length(points), 2)
    for (i, point) in enumerate(points)
        P[i, 1], P[i, 2] = (point...,)
    end
    _plot_vrep!(ax, P, fc, fa, ec, ew)
end

function _plot_vrep!(ax, P, fc, fa, ec, ew)
    verts = compute_vertices_vrep(P)
    polylist = matplotlib.collections.PolyCollection((verts,))
    fca = matplotlib.colors.colorConverter.to_rgba(fc, alpha=fa)
    polylist.set_facecolor(fca)
    polylist.set_edgecolor(ec)
    polylist.set_linewidth(ew)
    ax.add_collection(polylist)
end

function plot_point!(ax, point; mc="blue", ms=15)
    ax.plot(point..., marker=".", ms=ms, c=mc)
end