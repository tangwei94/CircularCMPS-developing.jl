# copied from MPSKit.jl
const MPSBondTensor{S} = AbstractTensorMap{S,1,1} where {S<:EuclideanSpace}
const GenericMPSTensor{S,N} = AbstractTensorMap{S,N,1} where {S<:EuclideanSpace,N} 
const MPSTensor{S} = GenericMPSTensor{S,2} where {S<:EuclideanSpace}

_firstspace(t::AbstractTensorMap) = space(t, 1)
_lastspace(t::AbstractTensorMap) = space(t, numind(t))

#=
map every element in the tensormap to dfun(E)
allows us to create random tensormaps for any storagetype
=#
function fill_data!(a::TensorMap,dfun)
    E = eltype(a);

    for (k,v) in blocks(a)
        map!(x->dfun(E),v,v);
    end

    a
end
randomize!(a::TensorMap) = fill_data!(a,randn)

# default permutation of K matrix 
function K_permute(K::AbstractTensorMap{S, 2, 2}) where {S<:EuclideanSpace}
    return permute(K, (2, 3), (4, 1))
end

function K_permute_back(K::AbstractTensorMap{S, 2, 2}) where {S<:EuclideanSpace}
    return permute(K, (4, 1), (2, 3))
end



function K_otimes(A::MPSBondTensor, B::MPSBondTensor)
    @tensor Abar_otimes_B[-1, -2; -3, -4] := A'[-3, -1] * B[-2, -4]
    return Abar_otimes_B
end