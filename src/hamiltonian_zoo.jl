abstract type AbstractHamiltonian end

struct SingleBosonLiebLiniger <: AbstractHamiltonian
    c::Real
    μ::Real
    L::Real
end

function ground_state(H::SingleBosonLiebLiniger, ψ0::CMPSData)
    if H.L == Inf
        function fE_inf(ψ::CMPSData)
            OH = kinetic(ψ) + H.c*point_interaction(ψ) - H.μ * particle_density(ψ)
            TM = TransferMatrix(ψ, ψ)
            envL = permute(left_env(TM), (), (1, 2))
            envR = permute(right_env(TM), (2, 1), ()) 
            return real(tr(envL * OH * envR) / tr(envL * envR))
        end
        @show "infinite system"

        return minimize(fE_inf, ψ0, CircularCMPSRiemannian(1000, 1e-9, 2)) # TODO. change this as input. 
    else
        @show "finite system of size $(H.L)"
        function fE_finiteL(ψ::CMPSData)
            OH = kinetic(ψ) + H.c*point_interaction(ψ) - H.μ * particle_density(ψ)
            expK, _ = finite_env(K_mat(ψ, ψ), H.L)
            return real(tr(expK * OH))
        end 

        return minimize(fE_finiteL, ψ0, CircularCMPSRiemannian(1000, 1e-9, 2)) # TODO. change this as input. 
    end
end

struct MultiBosonLiebLiniger <: AbstractHamiltonian
    cs::Matrix{<:Real}
    μs::Vector{<:Real}
    L::Real
end

#struct MultiBosonCMPSState
#    ψ::MultiBosonCMPSData
#    envL::MPSBondTensor
#    envR::MPSBondTensor
#    λ::Number
#    K::AbstractTensorMap
#    Kinv::AbstractTensorMap
#end

#function MultiBosonCMPSState(ψ::MultiBosonCMPSData)
#    ψn = CMPSData(ψ0)
#    K = K_permute(K_mat(ψn, ψn))
#    λ, EL = left_env(K)
#    λ, ER = right_env(K)
#    Kinv = Kmat_pseudo_inv(K, λ)
#    return MultiBosonCMPSState(ψ, EL, ER, λ, K, Kinv)
#end

function ground_state(H::MultiBosonLiebLiniger, ψ0::MultiBosonCMPSData; do_preconditioning::Bool=true, maxiter::Int=10000)
    if H.L == Inf
        cs = Matrix{ComplexF64}(H.cs)
        μs = Vector{ComplexF64}(H.μs)

        function fE_inf(ψ::MultiBosonCMPSData)
            ψn = CMPSData(ψ)
            OH = kinetic(ψn) + H.cs[1,1]* point_interaction(ψn, 1) + H.cs[2,2]* point_interaction(ψn, 2) + H.cs[1,2] * point_interaction(ψn, 1, 2) + H.cs[2,1] * point_interaction(ψn, 2, 1) - H.μs[1] * particle_density(ψn, 1) - H.μs[2] * particle_density(ψn, 2)
            TM = TransferMatrix(ψn, ψn)
            envL = permute(left_env(TM), (), (1, 2))
            envR = permute(right_env(TM), (2, 1), ()) 
            return real(tr(envL * OH * envR) / tr(envL * envR))
        end
        #function fE_inf(ψm::MultiBosonCMPSData)
        #    ψ = CMPSData(ψm)
        #    OH = kinetic(ψ) + point_interaction(ψ, cs) - particle_density(ψ, μs)

        #    TM = TransferMatrix(ψ, ψ)
        #    envL = permute(left_env(TM), (), (1, 2))
        #    envR = permute(right_env(TM), (2, 1), ()) 
        #    return real(tr(envL * OH * envR) / tr(envL * envR))
        #end
        @show "infinite system"
    
        function fgE(ψ::MultiBosonCMPSData)
            E = fE_inf(ψ)
            ∂ψ = fE_inf'(ψ) 
            return E, ∂ψ 
        end
    
        function inner(ψ, ψ1::MultiBosonCMPSData, ψ2::MultiBosonCMPSData)
            # be careful the cases with or without a factor of 2. depends on how to define the complex gradient
            return real(dot(ψ1, ψ2)) 
        end

        function retract(ψ::MultiBosonCMPSData, dψ::MultiBosonCMPSData, α::Real)
            Λs = ψ.Λs .+ α .* dψ.Λs 
            Q = ψ.Q + α * dψ.Q
            ψ1 = MultiBosonCMPSData(Q, Λs)
            return ψ1, dψ
        end

        function scale!(dψ::MultiBosonCMPSData, α::Number)
            dψ.Q = dψ.Q * α
            dψ.Λs .= dψ.Λs .* α
            return dψ
        end

        function add!(dψ::MultiBosonCMPSData, dψ1::MultiBosonCMPSData, α::Number) 
            dψ.Q += dψ1.Q * α
            dψ.Λs .+= dψ1.Λs .* α
            return dψ
        end

        # only for comparison
        function _no_precondition(ψ::MultiBosonCMPSData, dψ::MultiBosonCMPSData)
            return dψ
        end

        function _precondition(ψ0::MultiBosonCMPSData, dψ::MultiBosonCMPSData)
            # TODO. avoid the re-computation of K, λ, EL, ER, Kinv
            ψn = CMPSData(ψ0)
            K = K_permute(K_mat(ψn, ψn))
            λ, EL = left_env(K)
            λ, ER = right_env(K)
            Kinv = Kmat_pseudo_inv(K, λ)

            ϵ = max(1e-12, 1e-3*norm(dψ))
            mapped, _ = linsolve(X -> tangent_map(ψ0, X, EL, ER, Kinv) + ϵ*X, dψ, dψ; maxiter=250, ishermitian = true, isposdef = true, tol=ϵ)
            return mapped

            #χ, d = get_χ(ψ0), get_d(ψ0)
            #M = zeros(ComplexF64, χ^2+d*χ, χ^2+d*χ)
            #for ix in 1:(χ^2+d*χ)
            #    v = zeros(χ^2+d*χ)
            #    v[ix] = 1
          
            #    X = MultiBosonCMPSData(v, χ, d)
            #    v1 = vec(tangent_map(ψ0, X, EL, ER, Kinv))
            #    M[:, ix] = v1
            #end

            #λs, V = eigen(Hermitian(M))
            #λs[1] < -1e-9 && @warn "$(λs[1]) not positive definite"
            
            #dV = vec(dψ)
            #mappedV = V'[:, χ:end] * Diagonal(1 ./ (λs[χ:end] .+ ϵ)) * V[χ:end, :] * dV
            #return MultiBosonCMPSData(mappedV, χ, d)
        end

        transport!(v, x, d, α, xnew) = v

        optalg_LBFGS = LBFGS(;maxiter=maxiter, gradtol=1e-8, verbosity=2)

        if do_preconditioning
            @show "doing precondition"
            precondition = _precondition
        else
            @show "no precondition"
            precondition = _no_precondition
        end
        ψ1, E1, grad1, numfg1, history1 = optimize(fgE, ψ0, optalg_LBFGS; retract = retract,
                                        precondition = precondition,
                                        inner = inner, transport! =transport!,
                                        scale! = scale!, add! = add!
                                        );

        res1 = (ψ1, E1, grad1, numfg1, history1)
        return res1

    else
        @show "finite system of size $(H.L)"

        error("finite size not implemented yet.")
    end
end