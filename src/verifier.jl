struct PosPredicate
    N::Int
    domain::Polyhedron
    loc::Int
end

struct LiePredicate
    N::Int
    domain::Polyhedron
    loc1::Int
    A::Matrix{Float64}
    b::Vector{Float64}
    loc2::Int
end

struct Verifier
    pos_predics::Vector{PosPredicate}
    lie_predics::Vector{LiePredicate}
end

Verifier() = Verifier(PosPredicate[], LiePredicate[])

function add_predicate!(verif::Verifier, predic::PosPredicate)
    push!(verif.pos_predics, predic)
end

function add_predicate!(verif::Verifier, predic::LiePredicate)
    push!(verif.lie_predics, predic)
end

## Optim problem

abstract type VerifierProblem end

function _add_variables!(model, N, xmax, rmax)
    x = @variable(model, [1:N], lower_bound=-xmax, upper_bound=xmax)
    r = @variable(model, upper_bound=rmax)
    return x, r
end

function _verify!(prob::VerifierProblem, pfs, xmax, rmax, solver)
    model = solver()
    x, r = _add_variables!(model, prob.N, xmax, rmax)

    _add_constrs_prob!(prob, model, x, r, pfs)

    @objective(model, Max, r)

    optimize!(model)

    xopt = has_values(model) ? value.(x) : Float64[]
    ropt = has_values(model) ? value(r) : -Inf
    ps, ts = primal_status(model), termination_status(model)
    flag = ps == FEASIBLE_POINT && ts == OPTIMAL
    @assert flag || (ps == NO_SOLUTION && ts == INFEASIBLE)

    return xopt, ropt, flag
end

## Verif Pos
struct VerifierPos <: VerifierProblem
    N::Int
    domain::Polyhedron
    loc::Int
end

function _add_constrs_prob!(prob::VerifierPos, model, x, r, pfs)
    for h in prob.domain.halfspaces
        @constraint(model, dot(h.a, x) + h.β ≤ 0)
    end
    for lf in pfs[prob.loc].afs
        @constraint(model, 0 ≥ _eval(lf, x) + r)
    end
end

function verify_pos(verif::Verifier, mpf::MultiPolyFunc, xmax, rmax, solver)
    xopt::Vector{Float64} = Float64[]
    ropt::Float64 = -Inf
    locopt::Int = 0
    for predic in verif.pos_predics
        prob = VerifierPos(predic.N, predic.domain, predic.loc)
        x, r, flag = _verify!(prob, mpf.pfs, xmax, rmax, solver)
        if flag && r > ropt
            xopt = x
            ropt = r
            locopt = predic.loc
        end
    end
    return xopt, ropt, locopt
end

## Verify Lie
struct VerifierLie <: VerifierProblem
    N::Int
    domain::Polyhedron
    loc1::Int
    A::Matrix{Float64}
    b::Vector{Float64}
    loc2::Int
    i2::Int
end

function _add_constrs_prob!(prob::VerifierLie, model, x, r, pfs)
    for h in prob.domain.halfspaces
        @constraint(model, dot(h.a, x) + h.β ≤ 0)
    end
    for lf1 in pfs[prob.loc1].afs
        @constraint(model, _eval(lf1, x) ≤ 0)
    end
    val2 = _eval(pfs[prob.loc2].afs[prob.i2], prob.A*x + prob.b)
    @constraint(model, val2 ≥ r)
end

function verify_lie(verif::Verifier, mpf::MultiPolyFunc, xmax, rmax, solver)
    xopt::Vector{Float64} = Float64[]
    ropt::Float64 = -Inf
    locopt::Int = 0
    for predic in verif.lie_predics
        for i2 in eachindex(mpf.pfs[predic.loc2].afs)
            prob = VerifierLie(
                predic.N, predic.domain,
                predic.loc1, predic.A, predic.b, predic.loc2, i2
            )
            x, r, flag = _verify!(prob, mpf.pfs, xmax, rmax, solver)
            if flag && r > ropt
                xopt = x
                ropt = r
                locopt = predic.loc1
            end
        end
    end
    return xopt, ropt, locopt
end