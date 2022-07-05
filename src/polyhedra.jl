struct Halfspace
    a::Vector{Float64}
    β::Float64
end

Base.in(x, h::Halfspace) = dot(h.a, x) + h.β ≤ 0
near(x, h::Halfspace, tol) = dot(h.a, x) + h.β ≤ tol*sqrt(norm(h.a)^2 + h.β^2)

struct Polyhedron
    halfspaces::Vector{Halfspace}
end

Polyhedron() = Polyhedron(Halfspace[])

add_halfspace!(p::Polyhedron, h::Halfspace) = push!(p.halfspaces, h)    
add_halfspace!(p::Polyhedron, h_...) = add_halfspace!(p, Halfspace(h_...))

Base.in(x, p::Polyhedron) = all(h -> x ∈ h, p.halfspaces)
near(x, p::Polyhedron, tol) = all(h -> near(x, h, tol), p.halfspaces)

Base.intersect(p1::Polyhedron, p2::Polyhedron) = Polyhedron(
    vcat(p1.halfspaces, p2.halfspaces)
)