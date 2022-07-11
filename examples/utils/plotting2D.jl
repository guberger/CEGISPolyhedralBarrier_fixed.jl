include("geometry.jl")

function plot_level!(ax, afs, lims; fc="green", fa=0.5, ec="green", ew=1.0)
    A = zeros(length(afs), 2)
    b = zeros(length(afs))
    for (i, af) in enumerate(afs)
        A[i, 1], A[i, 2] = (af.a...,)
        b[i] = -af.β
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

function plot_point!(ax, point; mc="blue", ms=15)
    ax.plot(point..., marker=".", ms=ms, c=mc)
end