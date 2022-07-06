using LinearAlgebra
using PyCall
const spatial = pyimport_conda("scipy.spatial", "scipy")
const optimize = pyimport_conda("scipy.optimize", "scipy")

function compute_vertices_hrep(A, b)
    @assert (size(A, 1),) == size(b)
    nvar = size(A, 2)
    M = hcat(A, -b)
    A_ub = hcat(A, map(r -> norm(r), eachrow(A)))
    c_obj = zeros(nvar + 1)
    c_obj[nvar + 1] = -1
    bounds = ((nothing, nothing), (nothing, nothing), (nothing, 1))
    res = optimize.linprog(c_obj, A_ub=A_ub, b_ub=b, bounds=bounds)
    @assert res["success"] && res["status"] == 0
    res["fun"] > 0 && return Vector{Float64}[]
    x = res["x"][1:nvar]
    hs = spatial.HalfspaceIntersection(M, x)
    points = collect.(eachrow(hs.intersections))
    ch = spatial.ConvexHull(points)
    return [ch.points[i + 1, :] for i in ch.vertices]
end

function compute_vertices_vrep(P)
    nvar = size(P, 2)
    size(P, 1) == 1 && return fill(P[1, :], nvar)
    x = sum(P, dims=1)/size(P, 1)
    xv = x[:]
    Q = P .- x
    F = svd(P)
    r_ = findfirst(s -> s < F.S[1]/1e4, F.S)
    r = isnothing(r_) ? nvar : r_ - 1
    r == 0 && return fill(xv, nvar)
    V = F.V[:, 1:r]
    Qp = Q*V
    if r == 1
        v1 = minimum(Qp)
        v2 = maximum(Qp)
        return [V*[v1] + xv, V*[v2] + xv]
    end
    ch = spatial.ConvexHull(Qp)
    return [V*ch.points[i + 1, :] + xv for i in ch.vertices]
end