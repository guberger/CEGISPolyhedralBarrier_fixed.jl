function _verify(domain, A, b, pf1, af2, xmax, solver)
    N = size(A, 2)
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

function verify(mpf::MultiPolyFunc, sys::System, xmax, solver)
    xopt::Vector{Float64} = Float64[]
    ropt::Float64 = -Inf
    locopt::Int = 0
    for piece in sys.pieces
        domain, A, b = piece.domain, piece.A, piece.b
        pf1 = mpf.pfs[piece.loc1]
        for af2 in mpf.pfs[piece.loc2].afs
            x, r, flag = _verify(domain, A, b, pf1, af2, xmax, solver)
            if flag && r > ropt
                xopt = x
                ropt = r
                locopt = piece.loc1
            end
        end
    end
    return xopt, ropt, locopt
end