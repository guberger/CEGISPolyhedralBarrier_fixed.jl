## Learner

@enum StatusCode begin
    NOT_SOLVED = 0
    BARRIER_FOUND = 1
    BARRIER_INFEASIBLE = 2
    RADIUS_TOO_SMALL = 3
    MAX_ITER_REACHED = 4
end

abstract type TreeExploreMethod end
struct DepthMin <: TreeExploreMethod end
struct Depth1st <: TreeExploreMethod end
struct RadMax <: TreeExploreMethod end
struct ObjMin <: TreeExploreMethod end

Node{GT} = Tuple{GT,Int,Float64,Float64}

_make_queue(::Depth1st, GT) = Stack{Node{GT}}()
_make_queue(::DepthMin, GT) = Queue{Node{GT}}()
_make_queue(::RadMax, GT) = PriorityQueue{Node{GT},Float64}()
_make_queue(::ObjMin, GT) = PriorityQueue{Node{GT},Float64}()

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

## Learner

struct Learner{N,M}
    nafs::NTuple{M,Int}
    sys::System{N}
    mpf_safe::MultiPolyFunc{N,M}
    mpf_inv::MultiPolyFunc{N,M}
    iset::PointSet{N,M}
    tols::Dict{Symbol,Float64}
    params::Dict{Symbol,Float64}
end

function Learner(nafs, sys, mpf_safe, mpf_inv, iset)
    tols = Dict([
        :rad => eps(1.0),
        :verif => -eps(1.0),
        :dom => eps(1.0)
    ])
    params = Dict([
        :xmax => 1e3,
    ])
    return Learner(nafs, sys, mpf_safe, mpf_inv, iset, tols, params)
end

_setsafe!(D, k, v) = (@assert haskey(D, k); D[k] = v)
set_tol!(lear::Learner, s::Symbol, v) = _setsafe!(lear.tols, s, v)
set_param!(lear::Learner, s::Symbol, v) = _setsafe!(lear.params, s, v)

function compute_post!(set2, set1, sys, tol_dom)
    for (loc1, points) in enumerate(set1.points_list)
        for piece in sys.pieces
            loc1 != piece.loc1 && continue
            for point in points
                !_neg(piece.pf_dom, point, tol_dom) && continue
                add_point!(set2, piece.loc2, piece.A*point + piece.b)
            end
        end
    end
end

# print rules
_pr_full(PR) = PR == "full"
_pr_none(PR) = PR == "none"
_pr_part(PR, iter) =
    PR == "full" ? true :
    PR =="none" ? false :
    mod(iter - 1, Int(PR)) == 0

function learn_lyapunov!(
        lear::Learner{N,M}, iter_max, solver_gen, solver_verif;
        PR="full", method::TreeExploreMethod=Depth1st()
    ) where {N,M}
    @assert iter_max ≥ 1

    gen = Generator{N}(lear.nafs)
    for (loc, points) in enumerate(lear.iset.points_list)
        for point in points
            add_evidence!(gen, NegEvidence(loc, point))
        end
    end
    node_queue = _make_queue(method, typeof(gen))
    _enqueue!(method, node_queue, (gen, 0, Inf, Inf))

    iter = 0
    xmax = lear.params[:xmax]
    tol_dom = lear.tols[:dom]
    mpf::MultiPolyFunc = MultiPolyFunc(map(
        naf -> PolyFunc([
            AffForm(SVector(ntuple(k -> NaN, Val(N))), NaN) for i = 1:naf
        ]), lear.nafs
    )) # never used
    set1 = PointSet{N,M}()
    set2 = PointSet{N,M}()

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
        verif = Verifier(lear.mpf_safe, lear.mpf_inv, mpf, lear.sys, xmax)

        # Verifier
        _pr_full(PR) && print("|--- verify safe... ")
        x, obj, loc = verify_safe(verif, solver_verif)
        if obj > lear.tols[:verif]
            _pr_full(PR) && println("CE found: ", x, ", ", loc, ", ", obj)
            for i = 1:lear.nafs[loc]
                gen2 = Generator(lear.nafs, gen.neg_evids, copy(gen.pos_evids))
                add_evidence!(gen2, PosEvidence(loc, i, x))
                _enqueue!(method, node_queue, (gen2, depth2, r, obj))
            end
            continue
        else
            _pr_full(PR) && println("No CE found: ", obj)
        end
        _pr_full(PR) && print("|--- verify BF... ")
        x, obj, loc = verify_BF(verif, solver_verif)
        if obj > lear.tols[:verif]
            _pr_full(PR) && println("CE found: ", x, ", ", loc, ", ", obj)
            for i = 1:lear.nafs[loc]
                gen2 = Generator(lear.nafs, gen.neg_evids, copy(gen.pos_evids))
                add_evidence!(gen2, PosEvidence(loc, i, x))
                _enqueue!(method, node_queue, (gen2, depth2, r, obj))
            end
            empty!(set1)
            empty!(set2)
            add_point!(set1, loc, x)
            compute_post!(set2, set1, lear.sys, tol_dom)
            gen2 = Generator(lear.nafs, copy(gen.neg_evids), gen.pos_evids)
            for (loc, points) in enumerate(set2.points_list)
                for point in points
                    add_evidence!(gen2, NegEvidence(loc, point))
                end
            end
            _enqueue!(method, node_queue, (gen2, depth2, r, obj))
            continue
        else
            _pr_full(PR) && println("No CE found: ", obj)
        end
        
        !_pr_none(PR) && println("Valid CLF: terminated")
        return BARRIER_FOUND, mpf, gen
    end
end