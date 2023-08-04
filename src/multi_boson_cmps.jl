mutable struct MultiBosonCMPSData <: AbstractCMPSData 
    Q::MPSBondTensor
    Ms::Vector{<:MPSBondTensor}
end

function MultiBosonCMPSData(f, χ::Integer, d::Integer)
    Q = TensorMap(f, ComplexF64, ℂ^(χ^d), ℂ^(χ^d))
    Ms = MPSBondTensor{ComplexSpace}[]
    for ix in 1:d 
        push!(Ms, TensorMap(f, ComplexF64, ℂ^χ, ℂ^χ))
    end
    return MultiBosonCMPSData(Q, Ms)
end

# operations on the data. not on the cMPS
Base.:+(ψ::MultiBosonCMPSData, ϕ::MultiBosonCMPSData) = MultiBosonCMPSData(ψ.Q + ϕ.Q, ψ.Ms .+ ϕ.Ms)
Base.:-(ψ::MultiBosonCMPSData, ϕ::MultiBosonCMPSData) = MultiBosonCMPSData(ψ.Q - ϕ.Q, ψ.Ms .- ϕ.Ms)
Base.:*(ψ::MultiBosonCMPSData, x::Number) = MultiBosonCMPSData(ψ.Q * x, ψ.Ms .* x)
Base.:*(x::Number, ψ::MultiBosonCMPSData) = MultiBosonCMPSData(ψ.Q * x, ψ.Ms .* x)
LinearAlgebra.dot(ψ1::MultiBosonCMPSData, ψ2::MultiBosonCMPSData) = dot(ψ1.Q, ψ2.Q) + sum(dot.(ψ1.Ms, ψ2.Ms))
LinearAlgebra.norm(ψ::MultiBosonCMPSData) = sqrt(norm(dot(ψ, ψ)))

function Base.similar(ψ::MultiBosonCMPSData) 
    Q = similar(ψ.Q)
    Ms = [similar(M) for M in ψ.Ms]
    return MultiBosonCMPSData(Q, Ms)
end

function Base.zero(ψ::MultiBosonCMPSData) 
    Q = zero(ψ.Q)
    Ms = [zero(M) for M in ψ.Ms]
    return MultiBosonCMPSData(Q, Ms)
end

@inline get_χ(ψ::MultiBosonCMPSData) = dim(_firstspace(ψ.Ms[1]))
@inline get_d(ψ::MultiBosonCMPSData) = length(ψ.Ms)
TensorKit.space(ψ::MultiBosonCMPSData) = _firstspace(ψ.Q)

function CMPSData(ψ::MultiBosonCMPSData)
    χ, d = get_χ(ψ), get_d(ψ)
    Rs = map(1:d) do ix
        Rops = repeat(MPSBondTensor[id(ℂ^χ)], d)
        Rops[ix] = ψ.Ms[ix]
        δ = isomorphism(ℂ^(χ^d), (ℂ^χ)^d)
        return δ * foldr(⊗, Rops) * δ'
    end
    return CMPSData(ψ.Q, Rs)
end

# transfer matrix in the space (ℂ^χ)^d ⊗ (ℂ^χ)'^d. deprecated 
#function transfer_matrix(ϕ::MultiBosonCMPSData, ψ::MultiBosonCMPSData)
#    d = get_d(ψ)
#    function fK(v)
#        Tv = ψ.Q * v + v * ϕ.Q'
#        for ix in 1:d
#            Md, Mu = ψ.Ms[ix], ϕ.Ms[ix]
#            vindices = Vector(-1:-1:(-2*d)) 
#            vindices[ix] = 1
#            vindices[ix+d] = 2
#            Tv += permute(ncon([Md, v, Mu'], [[-ix, 1], vindices, [2, -ix-d]]), Tuple(1:d), Tuple(d+1:2d));
#        end
#        return Tv
#    end
#    return fK 
#end
#
#"""
#   transfer_matrix_dagger(ϕ::CMPSData{S}, ψ::CMPSData{S}) where S<:EuclideanSpace
#    
#   The Hermitian conjugate of the transfer matrix for <ϕ|ψ>.
#   target at left vector. 
#   T† = I + ϵK†. returns K†
#"""
#function transfer_matrix_dagger(ϕ::MultiBosonCMPSData, ψ::MultiBosonCMPSData) 
#    d = get_d(ψ)
#    function fK_dagger(v)
#        Tv = v * ψ.Q + ϕ.Q' * v 
#        for ix in 1:d
#            Md, Mu = ψ.Ms[ix], ϕ.Ms[ix]
#            vindices = Vector(-1:-1:(-2*d)) 
#            vindices[ix] = 1
#            vindices[ix+d] = 2
#            Tv += permute(ncon([Mu', v, Md], [[-ix, 1], vindices, [2, -ix-d]]), Tuple(1:d), Tuple(d+1:2d));
#        end
#        return Tv
#    end
#    return fK_dagger
#end
