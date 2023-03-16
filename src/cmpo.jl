abstract type AbstractCMPO end 

struct CMPO <: AbstractCMPO 
    Q::MPSBondTensor
    Rs::Vector{<:MPSBondTensor}
    Ls::Vector{<:MPSBondTensor}
    Ps::Matrix{<:MPSBondTensor}
end

@inline get_χ(T::CMPO) = dim(_firstspace(T.Q))

function Base.:*(M::Matrix{<:AbstractTensorMap}, v::Vector{<:AbstractTensorMap})
    Mv = AbstractTensorMap[]
    for ix in eachindex(v)
        push!(Mv, sum(M[ix, :] .⊗ v))
    end
    return Mv
end

#function Base.:*(v::Vector{<:AbstractTensorMap}, M::Matrix{<:AbstractTensorMap})
#    Mv = AbstractTensorMap[]
#    for ix in eachindex(v)
#        push!(Mv, sum(v .⊗ M[:, ix]))
#    end
#    return Mv
#end

"""
    Base.:*(T::CMPO, ψ::CMPSData)

    Act cMPO `T` on the cMPS `ψ`. The computation doesn't involve the length of the cMPS. 
"""
function Base.:*(T::CMPO, ψ::CMPSData)
    χψ, χT = get_χ(ψ), get_χ(T)
    χ = χT * χψ
    
    t_fuse = isomorphism(ℂ^χ, ℂ^χT*ℂ^χψ)
    Id_T = id(_firstspace(T.Q))
    Id_ψ = id(_firstspace(ψ.Q))

    Q = sum(T.Rs .⊗ ψ.Rs) + Id_T ⊗ ψ.Q + T.Q ⊗ Id_ψ
    Rs = T.Ls .⊗ Ref(Id_ψ) + T.Ps * ψ.Rs
    
    Q = t_fuse * Q * t_fuse'
    Rs = Ref(t_fuse) .* Rs .* Ref(t_fuse')
    return CMPSData(Q, Rs)
end

#"""
#    Base.:*(ψ::CMPSData, T::CMPO)
#
#    Act cMPO `T` on the cMPS `ψ` from the left. The computation doesn't involve the length of the cMPS. 
#"""
#function Base.:*(ψ::CMPSData, T::CMPO)
#    χψ, χT = get_χ(ψ), get_χ(T)
#    χ = χT * χψ
#    
#    t_fuse = isomorphism(ℂ^χ, ℂ^χψ*ℂ^χT)
#    Id_T = id(_firstspace(T.Q))
#    Id_ψ = id(_firstspace(ψ.Q))
#
#    Q = sum(ψ.Rs .⊗ T.Ls) + Id_ψ ⊗ T.Q + ψ.Q ⊗ Id_T
#    Rs = ψ.Rs * T.Ps + Ref(Id_ψ) .⊗ T.Rs
#    
#    Q = t_fuse * Q * t_fuse'
#    Rs = Ref(t_fuse) .* Rs .* Ref(t_fuse')
#    return CMPSData(Q, Rs)
#
#end

"""
    Base.:*(ϕ::CMPSData, ψ::CMPSData)

    We define the product between two `CMPSData`'s as the `K_mat` obtained from the two cMPS's. 
"""
function Base.:*(ϕ::CMPSData, ψ::CMPSData)
    return K_mat(ϕ, ψ)
end

function ln_ovlp(ϕ::CMPSData, ψ::CMPSData, L::Real)
    return finite_env(ϕ*ψ, L)[2]
end

function ln_ovlp(ϕ::CMPSData, T::CMPO, ψ::CMPSData, L::Real)
    return finite_env(ϕ * (T * ψ), L)[2]
end

function compress(ψ::CMPSData, χ::Integer, L::Real; maxiter::Integer=100, tol::Real=1e-8, verbosity::Integer=1, init=nothing, ϵ::Real=1e-6)
    # TODO. wrap
    if χ >= get_χ(ψ)
        @warn "no need to compress"
        return ψ
    end

    K = ψ * ψ
    expK, ln_norm = finite_env(K, L)

    @tensor M[-1; -2] := expK[1 -1 ; 1 -2]
    _, U = eig(M; sortby=λ->-real(λ))
    P = isometry(space(ψ), ℂ^χ)
    U1 = U * P
    U1inv = P' * inv(U) 

    Q1 = U1inv * ψ.Q * U1
    Rs1 = Ref(U1inv) .* ψ.Rs .* Ref(U1)
    ψ1 = CMPSData(Q1, Rs1)

    # variational optimization
    function _f(ϕ::CMPSData)
        return real(-ln_ovlp(ϕ, ψ, L) - ln_ovlp(ψ, ϕ, L) + ln_ovlp(ϕ, ϕ, L) + ln_norm)
    end

    if init !== nothing && _f(init) < _f(ψ1)
        ψ1 = init
    end
    ψ1 = ψ1 + ϵ * CMPSData(rand, χ, get_d(ψ)) #perturb

    function _fg(ϕ::CMPSData)
        fvalue = _f(ϕ)
        ∂ϕ = _f'(ϕ)
        dQ = zero(∂ϕ.Q) 
        dRs = ∂ϕ.Rs .- ϕ.Rs .* Ref(∂ϕ.Q)

        return fvalue, CMPSData(dQ, dRs) 
    end
    function inner(ϕ, ϕ1::CMPSData, ϕ2::CMPSData)
        return real(sum(dot.(ϕ1.Rs, ϕ2.Rs)))
    end
    function retract(ϕ::CMPSData, dϕ::CMPSData, α::Real)
        Rs = ϕ.Rs .+ α .* dϕ.Rs 
        Q = ϕ.Q - α * sum(adjoint.(ϕ.Rs) .* dϕ.Rs) - 0.5 * α^2 * sum(adjoint.(dϕ.Rs) .* dϕ.Rs)
        ϕ1 = CMPSData(Q, Rs)
        return ϕ1, dϕ
    end
    function scale!(dϕ::CMPSData, α::Number)
        dϕ.Q = dϕ.Q * α
        dϕ.Rs .= dϕ.Rs .* α
        return dϕ
    end
    function add!(dϕ::CMPSData, dϕ1::CMPSData, α::Number)
        dϕ.Q += dϕ1.Q * α
        dϕ.Rs .+= dϕ1.Rs .* α
        return dϕ
    end
    function precondition(ϕ::CMPSData, dϕ::CMPSData)
        fK = transfer_matrix(ϕ, ϕ)

        # solve the fixed point equation
        init = similar(ϕ.Q, _firstspace(ϕ.Q)←_firstspace(ϕ.Q))
        randomize!(init);
        _, vrs, _ = eigsolve(fK, init, 1, :LR)
        vr = vrs[1]

        δ = inner(ϕ, dϕ, dϕ)

        P = herm_reg_inv(vr, max(1e-12, 1e-3*δ)) 

        Q = dϕ.Q  
        Rs = dϕ.Rs .* Ref(P)

        return CMPSData(Q, Rs)
    end
    transport!(v, x, d, α, xnew) = v
    
    optalg_LBFGS = LBFGS(;maxiter=maxiter, gradtol=tol, verbosity=verbosity)

    ψ1 = left_canonical(ψ1)[2]
    ψ1, ln_fidel, grad, numfg, history = optimize(_fg, ψ1, optalg_LBFGS; retract = retract,
                                    precondition = precondition,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!
                                    );

    return ψ1#, ln_fidel, grad, numfg, history 
end

# only implemented for plain tensors
function direct_sum(A::MPSBondTensor, B::MPSBondTensor)
    χA, χB = dim(_firstspace(A)), dim(_firstspace(B))

    A_oplus_B = TensorMap(zeros, ComplexF64, ℂ^(χA+χB), ℂ^(χA+χB))
    A_oplus_B.data[1:χA, 1:χA] = A.data
    A_oplus_B.data[1+χA:end, 1+χA:end] = B.data
    return A_oplus_B
end

function direct_sum(ψ1::CMPSData, ψ2::CMPSData; α::Real=0)
    Id = id(domain(ψ2.Q))
    Q = direct_sum(ψ1.Q, ψ2.Q + α*Id)
    Rs = direct_sum.(ψ1.Rs, ψ2.Rs)
    return CMPSData(Q, Rs)
end