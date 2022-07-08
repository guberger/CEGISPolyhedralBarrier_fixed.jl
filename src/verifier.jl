struct Predicate
    N::Int
    domain::Polyhedron
    loc1::Int
    A::Matrix{Float64}
    b::Vector{Float64}
    loc2::Int
end

struct Verifier
    predics::Vector{Predicate}
end

Verifier() = Verifier(Predicate[])

function add_predicate!(verif::Verifier, predic::Predicate)
    push!(verif.predics, predic)
end

function _verify!(N, domain, A, b, pf1, af2, xmax, solver)
    model = solver()
    x = @variable(model, [1:N], lower_bound=-xmax, upper_bound=xmax)

    for h in domain.halfspaces
        @constraint(model, dot(h.a, x) + h.β ≤ 0)
    end
    for af1 in pf1.afs
        @constraint(model, _eval(af1, x) ≤ 0)
    end

    @objective(model, Max, _eval(af2, A*x + b))

    optimize!(model)

    xopt = has_values(model) ? value.(x) : Float64[]
    ropt = has_values(model) ? objective_value(model) : -Inf
    ps, ts = primal_status(model), termination_status(model)
    flag = ps == FEASIBLE_POINT && ts == OPTIMAL
    @assert flag || (ps == NO_SOLUTION && ts == INFEASIBLE)

    return xopt, ropt, flag
end

function verify_lie(verif::Verifier, mpf::MultiPolyFunc, xmax, rmax, solver)
    xopt::Vector{Float64} = Float64[]
    ropt::Float64 = -Inf
    locopt::Int = 0
    for predic in verif.predics
        N, domain, A, b = predic.N, predic.domain, predic.A, predic.b
        pf1 = mpf.pfs[predic.loc1]
        for af2 in mpf.pfs[predic.loc2].afs
            x, r, flag = _verify!(N, domain, A, b, pf1, af2, xmax, solver)
            if flag && r > ropt
                xopt = x
                ropt = r
                locopt = predic.loc1
            end
        end
    end
    return xopt, ropt, locopt
end