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
        :objmax => 1e2
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

abstract type TreeExploreMethod end
struct DepthMin <: TreeExploreMethod end
struct Depth1st <: TreeExploreMethod end
struct RadMax <: TreeExploreMethod end
struct ObjMin <: TreeExploreMethod end

Node{GT} = Tuple{GT,Int,Float64,Tuple{Float64,Float64}}

_make_queue(::Depth1st, GT) = Stack{Node{GT}}()
_make_queue(::DepthMin, GT) = Queue{Node{GT}}()
_make_queue(::RadMax, GT) = PriorityQueue{Node{GT},Float64}()
_make_queue(::ObjMin, GT) = PriorityQueue{Node{GT},Tuple{Float64,Float64}}()

_enqueue!(::Depth1st, Q, node) = push!(Q, node)
_enqueue!(::DepthMin, Q, node) = enqueue!(Q, node)
_enqueue!(::RadMax, Q, node) = enqueue!(Q, node=>-node[3])
_enqueue!(::ObjMin, Q, node) = enqueue!(Q, node=>node[4])

_dequeue!(::Depth1st, Q) = pop!(Q)
_dequeue!(::DepthMin, Q) = dequeue!(Q)
_dequeue!(::RadMax, Q) = dequeue!(Q)
_dequeue!(::ObjMin, Q) = dequeue!(Q)

max_depth(Q::Union{Stack,Queue}) = maximum(n->n[2], Q)
max_depth(Q::PriorityQueue) = maximum(n->n.first[2], Q)
max_rad(Q::Union{Stack,Queue}) = maximum(n->n[3], Q)
max_rad(Q::PriorityQueue) = maximum(n->n.first[3], Q)
min_obj(Q::Union{Stack,Queue}) = minimum(n->n[4], Q)
min_obj(Q::PriorityQueue) = minimum(n->n.first[4], Q)

function learn_lyapunov!(
        lear::Learner, iter_max, solver_gen, solver_verif;
        PR="full", method::TreeExploreMethod=Depth1st()
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
    node_queue = _make_queue(method, typeof(gen))
    _enqueue!(method, node_queue, (gen, 0, Inf, (Inf, Inf)))

    iter = 0
    xmax, objmax = lear.params[:xmax], lear.params[:objmax]
    tol_dom = lear.tols[:dom]
    mpf::MultiPolyFunc = MultiPolyFunc(0) # never used

    # print rules
    _pr_full(PR) = PR == "full"
    _pr_none(PR) = PR == "none"
    _pr_part(PR, iter) =
        PR == "full" ? true :
        PR =="none" ? false :
        mod(iter - 1, Int(PR)) == 0

    while true
        if isempty(node_queue)
            !_pr_none(PR) && println("Infeasible: queue empty")
            return BARRIER_INFEASIBLE, mpf, gen
        end

        iter += 1
        if iter > iter_max
            !_pr_none(PR) && println("Max iter exceeded: ", iter)
            return MAX_ITER_REACHED, mpf, gen
        end
        if _pr_part(PR, iter)
            println("Iter: ", iter)
            println(
                "|--- max: depth: ", max_depth(node_queue),
                ", rad: ", max_rad(node_queue), ", obj: ", min_obj(node_queue)
            )
        end

        # Generator
        gen, depth, = _dequeue!(method, node_queue)
        _pr_full(PR) && print("|--- depth: ", depth)
        mpf, r = compute_mpf_robust(gen, solver_gen)
        _pr_full(PR) && println(", rad: ", r)
        if r < lear.tols[:rad]
            _pr_full(PR) && println("Radius too small: ", r)
            continue
        end
        depth2 = depth + 1

        # Verifier
        _pr_full(PR) && print("|--- Verify pos... ")
        x, obj, loc = verify_pos(verif, mpf, xmax, objmax, solver_verif)
        if obj > lear.tols[:pos]
            _pr_full(PR) && println("CE found: ", x, ", ", loc, ", ", obj)
            for i = 1:lear.nafs[loc]
                pos_evids = copy(gen.pos_evids)
                lie_evids = gen.lie_evids
                gen2 = Generator(lear.nafs, neg_evids, pos_evids, lie_evids)
                _add_evidences_pos!(gen2, i, loc, x)
                _enqueue!(method, node_queue, (gen2, depth2, r, (obj, Inf)))
            end
            continue
        else
            _pr_full(PR) && println("No CE found: ", obj)
        end
        _pr_full(PR) && print("|--- Verify lie... ")
        x, obj, loc = verify_lie(verif, mpf, xmax, objmax, solver_verif)
        if obj > lear.tols[:lie]
            _pr_full(PR) && println("CE found: ", x, ", ", loc, ", ", obj)
            for i = 1:lear.nafs[loc]
                pos_evids = copy(gen.pos_evids)
                lie_evids = gen.lie_evids
                gen2 = Generator(lear.nafs, neg_evids, pos_evids, lie_evids)
                _add_evidences_pos!(gen2, i, loc, x)
                _enqueue!(method, node_queue, (gen2, depth2, r, (-Inf, obj)))
            end
            pos_evids = gen.pos_evids
            lie_evids = copy(gen.lie_evids)
            gen2 = Generator(lear.nafs, neg_evids, pos_evids, lie_evids)
            _add_evidences_lie!(gen2, lear.sys, loc, x, tol_dom)
            _enqueue!(method, node_queue, (gen2, depth2, r, (-Inf, obj)))
            continue
        else
            _pr_full(PR) && println("No CE found: ", obj)
        end
        
        !_pr_none(PR) && println("No CE found")
        !_pr_none(PR) && println("Valid CLF: terminated")
        return BARRIER_FOUND, mpf, gen
    end
    @assert false
end