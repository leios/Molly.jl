export BitNeighborList,

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

    # ceil(log2(length(l))) finds the maximum number of bits to represent
    #     each element
    # length(l)^2 is the necessary number of elements for an all-to-all network
    max_bits = ceil(UInt,ceil(log2(length(l)))*length(l)^2)

    # This creates a BitArray to be interpreted into UInt8s
    max_bitstring = BitArray(zeros(max_bits))

    indices = zeros(Uint, length(l))

    offset = 1
    for i = 1:length(l)
        indices[i] = offset
        total_bitnum = 0
        for j = 1:length(l[i])
            # this is the number of bits needed to store the current value
            bitnum = ceil(UInt, log2(l[i][j]))
            range = offset+total_bitnum:offset+total_bitnum+bitnum
            max_bitstring[range] .= trunc(UInt(l[i][j]),bitnum)
            total_bitnum += bitnum
        end
        offset += ceil(UInt, total_bitnum*0.125)*8
    end

    final_bitstring = zeros(UInt8, offset * 0.125)

    for i = 1:length(final_bitstring)
        final_bitstring[i] = UInt8(max_bitstring[(i-1)*8+1:i*8])
    end

    return BitNeighborList(AT(final_bitstring), AT(indices))
end

# This will take an existing NeighborList and compress it to a BitNeighborList
function compress_neighborlist(nl::NeighbotList; AT = Array)
    return list_to_bitstring(nl.list)
end

# TODO
function compress_neighborlist(nl::NeighbotListVec; AT = Array)
end

# TODO: This function will create an encoded BitNeighborList from scratch
function encode_neighborlist(; AT = Array)
end

# This will take the neighbor bit string (nbs) and decode all neighbors 
# from idx_1, to idx_2
function decode_neighborlist(nbs::T, idx_1, idx_2;
                             AT = Array) where T <: Vector{UInt8}

end
