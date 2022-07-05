struct NegEvidence{N}
    loc::Int
    point::Point{N}
end

struct PosEvidence{N}
    loc::Int
    i::Int
    point::Point{N}
    np::Float64
end

struct LieEvidence{N}
    loc::Int
    point::Point{N}
    np::Float64
end

struct Generator{N,M}
    nafs::NTuple{M,Int}
    neg_evids::Vector{NegEvidence{N}}
    pos_evids::Vector{PosEvidence{N}}
    lie_evids::Vector{LieEvidence{N}}
end

Generator{N}(nafs::NTuple{M,Int}) where {N,M} = Generator{N,M}(
    nafs, NegEvidence{N}[], PosEvidence{N}[], LieEvidence{N}[]
)
nvar(::Generator{N}) where N = N

function add_evidence!(gen::Generator, evid::NegEvidence)
    push!(gen.neg_evids, evid)
end

function add_evidence!(gen::Generator, evid::PosEvidence)
    push!(gen.pos_evids, evid)
end

function add_evidence!(gen::Generator, evid::LieEvidence)
    push!(gen.lie_evids, evid)
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

function _add_neg_constr!(model, af, point)
    @constraint(model, _eval(af, point) ≤ 0)
end

function _add_pos_constr!(model, af, r, point, α)
    @constraint(model, _eval(af, point) - α*r ≥ 0)
end

function _add_lie_constr(model, af, r, point, α)
    @constraint(model, _eval(af, point) + α*r ≤ 0)
end

_value(af::_AF) = AffForm(value.(af.lin), value(af.off))

abstract type GeneratorProblem end

function _compute_mpf(prob::GeneratorProblem, gen::Generator, solver)
    model = solver()
    pfs, r = _add_vars!(model, nvar(gen), gen.nafs)

    for evid in gen.neg_evids
        for af in pfs[evid.loc].afs
            _add_constr_prob!(prob, model, af, evid)
        end
    end

    for evid in gen.pos_evids
        af = pfs[evid.loc].afs[evid.i]
        _add_constr_prob!(prob, model, af, r, evid)
    end

    for evid in gen.lie_evids
        for af in pfs[evid.loc].afs
            _add_constr_prob!(prob, model, af, r, evid)
        end
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
        ::GeneratorRobust, model, af, evid::NegEvidence
    )
    _add_neg_constr!(model, af, evid.point)
end


function _add_constr_prob!(
        ::GeneratorRobust, model, af, r, evid::PosEvidence
    )
    _add_pos_constr!(model, af, r, evid.point, evid.np + 1)
end

function _add_constr_prob!(
        ::GeneratorRobust, model, af, r, evid::LieEvidence
    )
    _add_lie_constr(model, af, r, evid.point, evid.np + 1)
end

function compute_mpf_robust(gen::Generator, solver)
    prob = GeneratorRobust()
    return _compute_mpf(prob, gen, solver)
end