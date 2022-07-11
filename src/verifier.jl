struct Verifier{N,M}
    mpf_safe::MultiPolyFunc{N,M}
    mpf_inv::MultiPolyFunc{N,M}
    mpf_BF::MultiPolyFunc{N,M}
    sys::System{N}
    xmax::Float64
end

# Safe
function _verify_optim(
        af1_, af2, A::SMatrix{N,N}, b::SVector{N}, xmax, solver
    ) where N
    model = solver()
    x = SVector(ntuple(
        k -> @variable(model, lower_bound=-xmax, upper_bound=xmax), Val(N)
    ))

    for af1 in af1_
        @constraint(model, _eval(af1, x) ≤ 0)
    end

    @objective(model, Max, _eval(af2, A*x + b))

    optimize!(model)

    xopt = has_values(model) ? value.(x) : SVector(ntuple(k -> NaN, Val(N)))
    ropt = has_values(model) ? objective_value(model) : -Inf
    ps, ts = primal_status(model), termination_status(model)
    flag = ps == FEASIBLE_POINT && ts == OPTIMAL
    @assert flag || (ps == NO_SOLUTION && ts == INFEASIBLE)

    return xopt, ropt, flag
end

abstract type VerifierProblem end

function _verify(
        prob::VerifierProblem, verif::Verifier{N,M}, solver
    ) where {N,M}
    xopt::SVector{N,Float64} = SVector(ntuple(k -> NaN, Val(N)))
    ropt::Float64 = -Inf
    locopt::Int = 0
    for piece in verif.sys.pieces
        pf_dom, A, b = piece.pf_dom, piece.A, piece.b
        pf_safe = verif.mpf_safe.pfs[piece.loc1]
        pf_inv = verif.mpf_inv.pfs[piece.loc1]
        pf_BF = verif.mpf_BF.pfs[piece.loc1]
        af1_ = Iterators.flatten(
            (pf_dom.afs, pf_safe.afs, pf_inv.afs, pf_BF.afs)
        )
        for af2 in _get_af2_(prob, verif, piece)
            x, r, flag = _verify_optim(af1_, af2, A, b, verif.xmax, solver)
            if flag && r > ropt
                xopt = x
                ropt = r
                locopt = piece.loc1
            end
        end
    end
    return xopt, ropt, locopt
end

struct VerifierSafe <: VerifierProblem end

_get_af2_(::VerifierSafe, verif, piece) = verif.mpf_safe.pfs[piece.loc2].afs

function verify_safe(verif::Verifier, solver)
    prob = VerifierSafe()
    return _verify(prob, verif, solver)
end

struct VerifierBF <: VerifierProblem end

_get_af2_(::VerifierBF, verif, piece) = verif.mpf_BF.pfs[piece.loc2].afs

function verify_BF(verif::Verifier, solver)
    prob = VerifierBF()
    return _verify(prob, verif, solver)
end