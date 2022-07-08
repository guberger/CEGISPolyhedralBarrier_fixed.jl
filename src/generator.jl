struct InEvidence{N}
    loc::Int
    point::Point{N}
end

struct ExEvidence{N}
    loc::Int
    i::Int
    point::Point{N}
end

struct NegEvidence{N}
    loc::Int
    point::Point{N}
    α::Float64 # weight
end

struct PosEvidence{N}
    loc::Int
    i::Int
    point::Point{N}
    α::Float64 # weight
end

struct Generator{N,M}
    nafs::NTuple{M,Int}
    in_evids::Vector{InEvidence{N}}
    ex_evids::Vector{ExEvidence{N}}
    neg_evids::Vector{NegEvidence{N}}
    pos_evids::Vector{PosEvidence{N}}
end

Generator{N}(nafs::NTuple{M,Int}) where {N,M} = Generator{N,M}(
    nafs, InEvidence{N}[], ExEvidence{N}[],
    NegEvidence{N}[], PosEvidence{N}[]
)

function add_evidence!(gen::Generator, evid::InEvidence)
    push!(gen.in_evids, evid)
end

function add_evidence!(gen::Generator, evid::ExEvidence)
    push!(gen.ex_evids, evid)
end

function add_evidence!(gen::Generator, evid::NegEvidence)
    push!(gen.neg_evids, evid)
end

function add_evidence!(gen::Generator, evid::PosEvidence)
    push!(gen.pos_evids, evid)
end

## Compute afs

struct _AF
    lin::Vector{VariableRef}
    off::VariableRef
end
_eval(af::_AF, point) = dot(point, af.lin) + af.off

struct _PF
    afs::Vector{_AF}
end

function _add_vars!(model, N, nafs)
    pfs = Vector{_PF}(undef, length(nafs))
    for (loc, naf) in enumerate(nafs)
        pfs[loc] = _PF(Vector{_AF}(undef, naf))
        for i = 1:naf
            lin = @variable(model, [1:N], lower_bound=-1, upper_bound=1)
            off = @variable(model, lower_bound=-1, upper_bound=1)
            pfs[loc].afs[i] = _AF(lin, off)
        end
    end
    r = @variable(model, upper_bound=10)
    return pfs, r
end

function _add_geq_constr!(model, af, r, point, α)
    @constraint(model, _eval(af, point) - α*r ≥ 0)
end

function _add_leq_constr(model, af, r, point, α)
    @constraint(model, _eval(af, point) + α*r ≤ 0)
end

_value(af::_AF) = AffForm(value.(af.lin), value(af.off))

abstract type GeneratorProblem end

function _compute_mpf(
        prob::GeneratorProblem, gen::Generator{N}, solver
    ) where N
    model = solver()
    pfs, r = _add_vars!(model, N, gen.nafs)

    for evid in gen.in_evids
        for af in pfs[evid.loc].afs
            _add_constr_prob!(prob, model, af, r, evid)
        end
    end

    for evid in gen.ex_evids
        af = pfs[evid.loc].afs[evid.i]
        _add_constr_prob!(prob, model, af, r, evid)
    end

    for evid in gen.neg_evids
        for af in pfs[evid.loc].afs
            _add_constr_prob!(prob, model, af, r, evid)
        end
    end

    for evid in gen.pos_evids
        af = pfs[evid.loc].afs[evid.i]
        _add_constr_prob!(prob, model, af, r, evid)
    end

    @objective(model, Max, r)

    optimize!(model)

    @assert termination_status(model) == OPTIMAL
    @assert primal_status(model) == FEASIBLE_POINT

    return MultiPolyFunc([
        PolyFunc([_value(af) for af in pf.afs]) for pf in pfs
    ]), value(r)
end

## Robust

struct GeneratorRobust <: GeneratorProblem end

function _add_constr_prob!(
        ::GeneratorRobust, model, af, r, evid::InEvidence
    )
    _add_leq_constr(model, af, r, evid.point, 0)
end

function _add_constr_prob!(
        ::GeneratorRobust, model, af, r, evid::ExEvidence
    )
    _add_geq_constr!(model, af, r, evid.point, 0)
end

function _add_constr_prob!(
        ::GeneratorRobust, model, af, r, evid::NegEvidence
    )
    _add_leq_constr(model, af, r, evid.point, evid.α)
end

function _add_constr_prob!(
        ::GeneratorRobust, model, af, r, evid::PosEvidence
    )
    _add_geq_constr!(model, af, r, evid.point, evid.α)
end

function compute_mpf_robust(gen::Generator, solver)
    prob = GeneratorRobust()
    return _compute_mpf(prob, gen, solver)
end