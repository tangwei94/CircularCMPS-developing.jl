function otimes_rev_l(X::AbstractTensorMap{<:EuclideanSpace, 2, 2}, B::AbstractTensorMap{<:EuclideanSpace, 1, 1})
    @tensor rev[-1; -2] := X[-1 1; -2 2] * B'[2; 1]
    return rev
end
function otimes_rev_r(X::AbstractTensorMap{<:EuclideanSpace, 2, 2}, A::AbstractTensorMap{<:EuclideanSpace, 1, 1})
    @tensor rev[-1; -2] := X[1 -1; 2 -2] * A'[2; 1]
    return rev
end

function ChainRulesCore.rrule(::typeof(TensorKit.:⊗), A::MPSBondTensor, B::MPSBondTensor)
    function otimes_pushback(f̄wd)
        Ā = otimes_rev_l(f̄wd, B)
        B̄ = otimes_rev_r(f̄wd, A)
        return NoTangent(), Ā, B̄
    end
    return A⊗B, otimes_pushback
end

#function ChainRulesCore.rrule(::typeof(Base.:*), M::Matrix{<:AbstractTensorMap}, v::Vector{<:AbstractTensorMap})
#    Mv = [sum(M[ix, :] .⊗ v) for ix in eachindex(v)]
#    sizev = size(M)[1]
#    function times_pushback(f̄wd)
#        v̄ = Vector{AbstractTensorMap}(undef, sizev)
#        M̄ = Matrix{AbstractTensorMap}(undef, sizev, sizev)
#
#        for iy in 1:sizev 
#            v̄[iy] = sum(otimes_rev_r.(f̄wd, M[:, iy]))
#        end
#        for ix in 1:sizev, iy in 1:sizev
#            M̄[ix, iy] = otimes_rev_l(f̄wd[ix], v[iy])
#        end
#        return NoTangent(), M̄, v̄
#    end
#    return Mv, times_pushback
#end