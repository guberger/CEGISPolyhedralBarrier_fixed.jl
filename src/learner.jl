## Learner

@enum StatusCode begin
    NOT_SOLVED = 0
    BARRIER_FOUND = 1
    BARRIER_INFEASIBLE = 2
    RADIUS_TOO_SMALL = 3
    MAX_ITER_REACHED = 4
end

## Learner

struct Learner{N,M}
    nafs::NTuple{M,Int}
    sys::System
    iset::InitialSet{M}
    uset::UnsafeSet{M}
    tols::Dict{Symbol,Float64}
    params::Dict{Symbol,Float64}
end
nvar(::Learner{N}) where N = N

function Learner{N}(nafs, sys, iset, uset) where N
    tols = Dict([
        :rad => eps(1.0),
        :pos => -eps(1.0),
        :lie => -eps(1.0),
        :dom => eps(1.0)
    ])
    params = Dict([
        :xmax => 1e3,
        :rmax => 1e2
    ])
    return Learner{N,length(nafs)}(nafs, sys, iset, uset, tols, params)
end

_setsafe!(D, k, v) = (@assert haskey(D, k); D[k] = v)
set_tol!(lear::Learner, s::Symbol, v) = _setsafe!(lear.tols, s, v)
set_param!(lear::Learner, s::Symbol, v) = _setsafe!(lear.params, s, v)

## Learn Barrier
function _add_evidences_neg!(gen, loc, point)
    x = Point{nvar(gen)}(point)
    add_evidence!(gen, NegEvidence(loc, x))
end

function _add_evidences_pos!(gen, i, loc, point)
    x = Point{nvar(gen)}(point)
    add_evidence!(gen, PosEvidence(loc, i, x, norm(x, Inf)))
end

function _add_evidences_lie!(gen, sys, loc1, point1, tol_dom)
    for piece in sys.pieces
        loc1 != piece.loc1 && continue
        !near(point1, piece.domain, tol_dom) && continue
        x2 = Point{nvar(gen)}(piece.A*point1 + piece.b)
        loc2 = piece.loc2
        add_evidence!(gen, LieEvidence(loc2, x2, norm(x2, Inf)))
    end
end

function _add_predicates_pos!(verif, N, uset)
    for (loc, domains) in enumerate(uset.domains_list)
        for domain in domains
            add_predicate!(verif, PosPredicate(N, domain, loc))
        end
    end
end

function _add_predicates_lie!(verif, N, sys)
    for piece in sys.pieces
        add_predicate!(verif, LiePredicate(
            N, piece.domain, piece.loc1, piece.A, piece.b, piece.loc2
        ))
    end
end

function learn_lyapunov!(
        lear::Learner, iter_max, solver_gen, solver_verif; PR="full"
    )
    @assert iter_max ≥ 1
    verif = Verifier()
    _add_predicates_pos!(verif, nvar(lear), lear.uset)
    _add_predicates_lie!(verif, nvar(lear), lear.sys)

    gen = Generator{nvar(lear)}(lear.nafs)
    for (loc, points) in enumerate(lear.iset.points_list)
        for point in points
            _add_evidences_neg!(gen, loc, point)
        end
    end
    neg_evids = gen.neg_evids
    # gen_queue = PriorityQueue((gen, 0, -Inf)=>-Inf)
    gen_queue = PriorityQueue((gen, 0, 0)=>0)

    iter = 0
    xmax, rmax = lear.params[:xmax], lear.params[:rmax]
    tol_dom = lear.tols[:dom]
    depth_max = 0
    mpf = MultiPolyFunc(0)

    # print rules
    _pr_full = PR == "full"
    _pr_none = PR == "none"
    _pr_part(PR, iter) =
        PR == "full" || (PR !="none" && mod(iter - 1, Int(PR)) == 0)

    while true
        if isempty(gen_queue)
            !_pr_none && println("Infeasible: queue empty")
            return BARRIER_INFEASIBLE, mpf, gen
        end

        iter += 1
        if iter > iter_max
            !_pr_none && println("Max iter exceeded: ", iter)
            return MAX_ITER_REACHED, mpf, gen
        end
        _pr_part(PR, iter) && println("Iter: ", iter)

        # Generator
        gen, depth, δ = dequeue!(gen_queue)
        depth_max = max(depth_max, depth)
        _pr_part(PR, iter) &&
            println("|--- depth: ", depth_max, ", δ: ", δ)
        mpf, r = compute_mpf_robust(gen, solver_gen)
        _pr_full && println("|--- radius: ", r)
        if r < lear.tols[:rad]
            _pr_full && println("Radius too small: ", r)
            continue
        end

        # Verifier
        _pr_full && print("|--- Verify pos... ")
        x, obj, loc = verify_pos(verif, mpf, xmax, rmax, solver_verif)
        # δ = -r
        δ = depth + 1
        if obj > lear.tols[:pos]
            _pr_full && println("CE found: ", x, ", ", loc, ", ", obj)
            for i = 1:lear.nafs[loc]
                pos_evids = copy(gen.pos_evids)
                lie_evids = gen.lie_evids
                gen2 = Generator(lear.nafs, neg_evids, pos_evids, lie_evids)
                _add_evidences_pos!(gen2, i, loc, x)
                enqueue!(gen_queue, (gen2, depth + 1, δ)=>δ)
            end
            continue
        else
            _pr_full && println("No CE found: ", obj)
        end
        _pr_full && print("|--- Verify lie... ")
        x, obj, loc = verify_lie(verif, mpf, xmax, rmax, solver_verif)
        if obj > lear.tols[:lie]
            _pr_full && println("CE found: ", x, ", ", loc, ", ", obj)
            for i = 1:lear.nafs[loc]
                pos_evids = copy(gen.pos_evids)
                lie_evids = gen.lie_evids
                gen2 = Generator(lear.nafs, neg_evids, pos_evids, lie_evids)
                _add_evidences_pos!(gen2, i, loc, x)
                enqueue!(gen_queue, (gen2, depth + 1, δ)=>δ)
            end
            pos_evids = gen.pos_evids
            lie_evids = copy(gen.lie_evids)
            gen2 = Generator(lear.nafs, neg_evids, pos_evids, lie_evids)
            _add_evidences_lie!(gen2, lear.sys, loc, x, tol_dom)
            enqueue!(gen_queue, (gen2, depth + 1, δ)=>δ)
            continue
        else
            _pr_full && println("No CE found: ", obj)
        end
        
        !_pr_none && println("No CE found")
        !_pr_none && println("Valid CLF: terminated")
        return BARRIER_FOUND, mpf, gen
    end
    @assert false
end