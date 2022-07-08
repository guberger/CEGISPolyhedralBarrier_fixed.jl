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
    sys::System
    iset::PointSet{M}
    uset::PointSet{M}
    tols::Dict{Symbol,Float64}
    params::Dict{Symbol,Float64}
end

function Learner{N}(nafs, sys, iset, uset) where N
    tols = Dict([
        :radius => eps(1.0),
        :verif => -eps(1.0),
        :domain => eps(1.0)
    ])
    params = Dict([
        :xmax => 1e3,
    ])
    return Learner{N,length(nafs)}(nafs, sys, iset, uset, tols, params)
end

_setsafe!(D, k, v) = (@assert haskey(D, k); D[k] = v)
set_tol!(lear::Learner, s::Symbol, v) = _setsafe!(lear.tols, s, v)
set_param!(lear::Learner, s::Symbol, v) = _setsafe!(lear.params, s, v)

function _add_images!(images_list, sys, loc1, point1, tol_domain)
    for piece in sys.pieces
        loc1 != piece.loc1 && continue
        !near(point1, piece.domain, tol_domain) && continue
        push!(images_list[piece.loc2], piece.A*point1 + piece.b)
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
    nafs_ = lear.nafs .+ 1

    gen = Generator{N}(nafs_)
    for (loc, points) in enumerate(lear.iset.points_list)
        for point in points
            add_evidence!(gen, InEvidence(loc, Point{N}(point)))
        end
    end
    for (loc, points) in enumerate(lear.uset.points_list)
        for point in points
            add_evidence!(gen, ExEvidence(loc, nafs_[loc], Point{N}(point)))
        end
    end
    node_queue = _make_queue(method, typeof(gen))
    _enqueue!(method, node_queue, (gen, 0, Inf, Inf))

    iter = 0
    xmax = lear.params[:xmax]
    tol_domain = lear.tols[:domain]
    mpf::MultiPolyFunc = MultiPolyFunc(0) # never used
    images_list = ntuple(i -> Vector{Float64}[], Val(M))

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
        if r < lear.tols[:radius]
            _pr_full(PR) && println("Radius too small: ", r)
            continue
        end
        depth2 = depth + 1

        # Verifier
        _pr_full(PR) && print("|--- verify... ")
        x, obj, loc = verify(mpf, lear.sys, xmax, solver_verif)
        if obj > lear.tols[:verif]
            _pr_full(PR) && println("CE found: ", x, ", ", loc, ", ", obj)
            point = Point{N}(x)
            α = norm(point, Inf) + 1
            for i = 1:lear.nafs[loc]
                gen2 = Generator(
                    nafs_, gen.in_evids, gen.ex_evids,
                    gen.neg_evids, copy(gen.pos_evids)
                )
                add_evidence!(gen2, PosEvidence(loc, i, point, α))
                _enqueue!(method, node_queue, (gen2, depth2, r, obj))
            end
            gen2 = Generator(
                nafs_, gen.in_evids, gen.ex_evids,
                copy(gen.neg_evids), gen.pos_evids
            )
            empty!.(images_list)
            _add_images!(images_list, lear.sys, loc, x, tol_domain)
            for (loc, images) in enumerate(images_list)
                for x in images
                    point = Point{N}(x)
                    α = norm(point, Inf) + 1
                    add_evidence!(gen2, NegEvidence(loc, point, α))
                end
            end
            _enqueue!(method, node_queue, (gen2, depth2, r, obj))
            continue
        else
            _pr_full(PR) && println("No CE found: ", obj)
        end
        
        !_pr_none(PR) && println("No CE found")
        !_pr_none(PR) && println("Valid CLF: terminated")
        return BARRIER_FOUND, mpf, gen
    end
end