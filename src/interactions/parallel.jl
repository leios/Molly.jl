export Parallel

"""
    Parallel(; nl_only)

A simple interaction that moves all particles in the same direction for testing
"""
struct Parallel <: PairwiseInteraction
    nl_only::Bool
end

Parallel(; nl_only=false) = parallel(nl_only)

@inline @inbounds function force(inter::Parallel,
                                 dr,
                                 coord_i,
                                 coord_j,
                                 atom_i,
                                 atom_j,
                                 boundary)
    println(dr)
    return SVector((0.1, 0.1))
end
