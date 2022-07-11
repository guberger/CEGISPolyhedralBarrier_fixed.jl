struct NegEvidence{N}
    loc::Int
    point::Point{N}
end

struct PosEvidence{N}
    loc::Int
    i::Int
    point::Point{N}
end

struct Generator{N,M}
    nafs::NTuple{M,Int}
    neg_evids::Vector{NegEvidence{N}}
    pos_evids::Vector{PosEvidence{N}}
end

Generator{N}(nafs::NTuple{M,Int}) where {N,M} = Generator{N,M}(
    nafs, NegEvidence{N}[], PosEvidence{N}[]
)

function add_evidence!(gen::Generator, evid::NegEvidence)
    push!(gen.neg_evids, evid)
end

function add_evidence!(gen::Generator, evid::PosEvidence)
    push!(gen.pos_evids, evid)
end

## Compute afs

struct _AF{N}
    a::SVector{N,VariableRef}
    β::VariableRef
end
_eval(af::_AF, point) = dot(point, af.a) + af.β

struct _PF{N}
    afs::Vector{_AF{N}}
end
_PF{N}(naf::Int) where N = _PF(Vector{_AF{N}}(undef, naf))

struct _MPF{N,M}
    pfs::NTuple{M,_PF{N}}
end

function _add_vars!(model, ::Val{N}, nafs::NTuple{M,Int}) where {N,M}
    pfs = map(naf -> _PF{N}(naf), nafs)
    for (loc, naf) in enumerate(nafs)
        for i = 1:naf
            a = SVector(ntuple(
                k -> @variable(model, lower_bound=-1, upper_bound=1), Val(N)
            ))
            β = @variable(model, lower_bound=-1, upper_bound=1)
            pfs[loc].afs[i] = _AF(a, β)
        end
    end
    r = @variable(model, upper_bound=10)
    return _MPF(pfs), r
end

function _add_geq_constr!(model, af, r, point, α)
    @constraint(model, _eval(af, point) - α*r ≥ 0)
end

function _add_leq_constr(model, af, r, point, α)
    @constraint(model, _eval(af, point) + α*r ≤ 0)
end

_value(af::_AF) = AffForm(value.(af.a), value(af.β))

abstract type GeneratorProblem end

function _compute_mpf(
        prob::GeneratorProblem, gen::Generator{N}, solver
    ) where N
    model = solver()
    mpf, r = _add_vars!(model, Val(N), gen.nafs)

    for evid in gen.neg_evids
        for af in mpf.pfs[evid.loc].afs
            _add_constr_prob!(prob, model, af, r, evid)
        end
    end

    for evid in gen.pos_evids
        af = mpf.pfs[evid.loc].afs[evid.i]
        _add_constr_prob!(prob, model, af, r, evid)
    end

    @objective(model, Max, r)

    optimize!(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    return MultiPolyFunc(map(
        pf -> PolyFunc([_value(af) for af in pf.afs]), mpf.pfs
    )), value(r)
end

## Robust

struct GeneratorRobust <: GeneratorProblem end

function _add_constr_prob!(
        ::GeneratorRobust, model, af, r, evid::NegEvidence
    )
    _add_leq_constr(model, af, r, evid.point, norm(evid.point, 1) + 1)
end

function _add_constr_prob!(
        ::GeneratorRobust, model, af, r, evid::PosEvidence
    )
    _add_geq_constr!(model, af, r, evid.point, norm(evid.point, 1) + 1)
end

function compute_mpf_robust(gen::Generator, solver)
    prob = GeneratorRobust()
    return _compute_mpf(prob, gen, solver)
end