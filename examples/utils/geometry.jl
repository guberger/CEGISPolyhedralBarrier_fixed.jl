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