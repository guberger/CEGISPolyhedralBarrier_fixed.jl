module CEGISPolyhedralBarrier

using LinearAlgebra
using DataStructures
using StaticArrays
using JuMP

Point{N} = SVector{N,Float64}

struct AffForm{N}
    a::Point{N}
    β::Float64
end
_eval(af::AffForm, point) = dot(af.a, point) + af.β

struct PolyFunc{N}
    afs::Vector{AffForm{N}}
end

PolyFunc{N}() where N = PolyFunc(AffForm{N}[])
add_af!(pf::PolyFunc, af::AffForm) = push!(pf.afs, af)
add_af!(pf::PolyFunc, af_...) = add_af!(pf, AffForm(af_...))
_neg(pf::PolyFunc, point, tol) = all(af -> _eval(af, point) ≤ tol, pf.afs)

struct MultiPolyFunc{N,M}
    pfs::NTuple{M,PolyFunc{N}}
end

MultiPolyFunc{N,M}() where {N,M} = MultiPolyFunc(
    ntuple(loc -> PolyFunc{N}(), Val(M))
)
add_af!(mpf::MultiPolyFunc, loc::Int, af_...) = add_af!(mpf.pfs[loc], af_...)

struct Piece{N}
    pf_dom::PolyFunc{N}
    loc1::Int
    A::SMatrix{N,N,Float64}
    b::SVector{N,Float64}
    loc2::Int
end

struct System{N}
    pieces::Vector{Piece{N}}
end

System{N}() where N = System(Piece{N}[])
add_piece!(sys::System, piece::Piece) = push!(sys.pieces, piece)
add_piece!(sys::System, piece_...) = add_piece!(sys, Piece(piece_...))

struct PointSet{N,M}
    points_list::NTuple{M,Vector{SVector{N,Float64}}}
end

PointSet{N,M}() where {N,M} = PointSet(
    ntuple(loc -> SVector{N,Float64}[], Val(M))
)
add_point!(S::PointSet, loc::Int, point) = push!(S.points_list[loc], point)
Base.empty!(S::PointSet) = empty!.(S.points_list)

include("generator.jl")
include("verifier.jl")
include("learner.jl")

end # module
