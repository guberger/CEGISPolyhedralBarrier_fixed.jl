module CEGISPolyhedralBarrier

using LinearAlgebra
using DataStructures
using StaticArrays
using JuMP

include("polyhedra.jl")

Point{N} = SVector{N,Float64}

struct AffForm
    lin::Vector{Float64}
    off::Float64
end
_eval(af::AffForm, point) = dot(af.lin, point) + af.off

struct PolyFunc
    afs::Vector{AffForm}
end

PolyFunc() = PolyFunc(AffForm[])
add_af!(pf::PolyFunc, af::AffForm) = push!(pf.afs, af)
add_af!(pf::PolyFunc, af_...) = add_af!(pf, AffForm(af_...))

struct MultiPolyFunc
    pfs::Vector{PolyFunc}
end

MultiPolyFunc(nloc::Int) = MultiPolyFunc([PolyFunc() for loc = 1:nloc])
add_af!(mpf::MultiPolyFunc, loc::Int, af_...) = add_af!(mpf.pfs[loc], af_...)

struct Piece
    domain::Polyhedron
    loc1::Int
    A::Matrix{Float64}
    b::Vector{Float64}
    loc2::Int
end

struct System
    pieces::Vector{Piece}
end

System() = System(Piece[])
add_piece!(sys::System, piece::Piece) = push!(sys.pieces, piece)
add_piece!(sys::System, piece_...) = add_piece!(sys, Piece(piece_...))

struct PointSet{M}
    points_list::NTuple{M,Vector{Vector{Float64}}}
end

PointSet{M}() where M = PointSet(ntuple(loc -> Vector{Float64}[], Val(M)))
add_point!(S::PointSet, loc, point) = push!(S.points_list[loc], point)

include("generator.jl")
include("verifier.jl")
include("learner.jl")

end # module
