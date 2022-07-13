export BitNeighborList, list_to_bitstring

using KernelAbstractions, CUDAKernels

"""
    BitNeighborList

Structure to contain neighborlists with a small memory footprint.
For this, the Neighborlist is stored as a bitstring in a "list of lists" format.
Each particle will have it's list concatenated with all other particles.
The Indices are the offset index within the NeighborBitString for each particle.

Note that BitArrays are not available on the GPU, so we use a CuArray{UInt8}.
"""

struct BitNeighborList
    bitstring::T where T <: Union{BitArray, Array{UInt8}, CuArray{UInt8}}
    indices::U where U <: Union{Array{UI}, CuArray{UI}} where UI <: Unsigned
end

# This function will take a list of lists and convert it into a vector of UInt8
function list_to_bitstring(l; AT = Array)

    # Step 1: pre-allocate the largest bit list on the CPU.
    #         we will later take a subset for the actual BitNeighborList

    # UT = ceil(log2(length(l))) finds the maximum number of bits to represent
    #     each element
    # length(l)^2 is the necessary number of elements for an all-to-all network
    bitsize = ceil(UInt, log2(length(l)))
    if bitsize <= 8
        UT = UInt8
    elseif bitsize == 16
        UT = UInt16
    elseif bitsize == 32
        UT = UInt32
    elseif bitsize == 64
        UT = UInt64
    elseif bitsize > 64
        error("Neighborlist cannot be compressed into UInt values")
    else
        error("Could not find appropriate size for NeighborList container")
    end

    max_bitstring = zeros(UT, length(l)^2)

    indices = zeros(UT, length(l)+1)

    offset = 0
    for i = 1:length(l)
        indices[i] = offset

        # counter for the inner list 
        count = 0
        for j = 1:length(l[i])
            # this is the number of bits needed to store the current value
            max_bitstring[offset+j] = trunc(UT, UInt(l[i][j]))
            count += 1
        end
        offset += count
    end

    indices[end] = length(l)

    return BitNeighborList(AT(max_bitstring[1:offset]), AT(indices))
end

# This will take an existing NeighborList and compress it to a BitNeighborList
function compress_neighborlist(nl::NeighborList; AT = Array)
    return list_to_bitstring(nl.list)
end

function compress_neighborlist(n::Nothing; kwargs...)
    return nothing
end

# TODO
function compress_neighborlist(nl::NeighborListVec; AT = Array)
end

# TODO: This function will create an encoded BitNeighborList from scratch
function encode_neighborlist(; AT = Array)
end

# This will take the neighbor bit string (nbs) and decode all neighbors 
# from idx_1, to idx_2
function decode_neighborlist(nbs::T, idx_1, idx_2;
                             AT = Array) where T <: Vector{UInt8}

end

function forces!(accelerations,
                 s::System, neighbor::Union{Nothing, BitNeighborList},
                 force; numcores = 4, numthreads = 256)

    if isa(s.coords, SVector) || isa(s.coords, Array)
        kernel! = forces_kernel(CPU(), numcores)
    else
        kernel! = forces_kernel(CUDADevice(), numthreads)
    end

    if neighbor == nothing
        kernel!(accelerations, s.coords, s.atoms, s.velocities,
                s.pairwise_inters, s.boundary, force,
                ndrange=length(length(s.coords)))
    else
        kernel!(accelerations, s.coords, s.atoms, s.velocities,
                s.pairwise_inters, s.boundary, force,
                neighbor.bitstring, neighbor.indices,
                force, ndrange=length(length(s.coords)))
    end
end

# this is for no-neighborlist computations
@kernel function forces_kernel(accelerations, coords, atoms, velocities, 
                               interaction, boundary, force)
    tid = @index(Global, Linear)
    lid = @index(Local, Linear)

    @inbounds n = size(accelerations)[2]

    FT = eltype(coords)

    @inbounds @uniform gs = @groupsize()[1]
    temp_acceleration = @localmem FT (gs, 4)

    for k = 1:n
        @inbounds temp_acceleration[lid, k] = 0
    end

    for j = 1:size(coords)[1]
        if j != tid
            dr = vector(coords[tid], coords[j], boundary)
            force(interaction, dr, coords[tid], coords[j],
                  atoms[tid], atoms[j], boundary)
        end
    end

    for k = 1:n
        @inbounds accelerations[tid,k] = temp_acceleration[lid,k]
    end
end

@kernel function forces_kernel(coords, atoms, velocities, interation, boundary,
                               bitstring, indices, force)
    tid = @index(Global, Linear)
    lid = @index(Local, Linear)

    @inbounds n = size(accelerations)[2]

    FT = eltype(coords)

    @inbounds @uniform gs = @groupsize()[1]
    temp_acceleration = @localmem FT (gs, 4)

    for k = 1:n
        @inbounds temp_acceleration[lid, k] = 0
    end

    for j = indices[tid]:indices[tid+1]
        if j != tid
            idx = bitstring[j]
            dr = vector(coords[tid], coords[idx], boundary)
            force(interaction, dr, coords[tid], coords[idx],
                  atoms[tid], atoms[idx], boundary)
        end
    end

    for k = 1:n
        @inbounds accelerations[tid,k] = temp_acceleration[lid,k]
    end
end
