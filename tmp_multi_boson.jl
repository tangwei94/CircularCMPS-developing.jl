using LinearAlgebra, TensorKit, KrylovKit
using TensorKitAD, ChainRules, Zygote 
using CairoMakie
using JLD2 
using OptimKit
using Revise
using CircularCMPS

c, μ = 1.21, 2.12
ψ0 = CMPSData(rand, 4, 1)
ϕ0 = MultiBosonCMPSData(rand, 4, 1)

function lieb_liniger_ground_state0(c::Real, μ::Real, ψ0::Union{CMPSData, Nothing}=nothing)
    function fE(ψ::CMPSData)
        OH = kinetic(ψ) + c*point_interaction(ψ) - μ * particle_density(ψ)
        TM = TransferMatrix(ψ, ψ)
        envL = permute(left_env(TM), (), (1, 2))
        envR = permute(right_env(TM), (2, 1), ()) 
        return real(tr(envL * OH * envR) / tr(envL * envR))
    end

    return CircularCMPS.minimize(fE, ψ0, CircularCMPSRiemannian(1000, 1e-9, 2))
end

function lieb_liniger_ground_state1(c::Real, μ::Real, ψ::Union{MultiBosonCMPSData, Nothing}=nothing)
    function fE(ψ::MultiBosonCMPSData)
        ψn = CMPSData(ψ)
        OH = kinetic(ψn) + c*point_interaction(ψn) - μ * particle_density(ψn)
        TM = TransferMatrix(ψn, ψn)
        envL = permute(left_env(TM), (), (1, 2))
        envR = permute(right_env(TM), (2, 1), ()) 
        return real(tr(envL * OH * envR) / tr(envL * envR))
    end
    
    function fgE(ψ::MultiBosonCMPSData)
        E = fE(ψ)
        ∂ψ = fE'(ψ) 
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
    function no_precondition(ψ::MultiBosonCMPSData, dψ::MultiBosonCMPSData)
        return dψ
    end

    function precondition(ψ0::MultiBosonCMPSData, dψ::MultiBosonCMPSData)
        ψn = CMPSData(ψ0)
        K = K_permute(ψn * ψn)
        _, ER = right_env(K)
        λ, EL = left_env(K)

        UR, SR, _ = tsvd(ER)
        UL, SL, _ = tsvd(EL)
        ER = UR * SR * UR'
        EL = UL * SL * UL'

        Kinv = Kmat_pseudo_inv(K, λ)
        
        ϵ = norm(dψ)
        mapped, _ = linsolve(X -> tangent_map(ψ0, X, EL, ER, Kinv) + ϵ*X, dψ, dψ; maxiter=250, ishermitian = true, isposdef = true, tol=0.1*ϵ)

        #χ, d = get_χ(ψ0), get_d(ψ0)
        #M = zeros(ComplexF64, χ^2+d*χ, χ^2+d*χ)
        #for ix in 1:(χ^2+d*χ)
        #    v = zeros(χ^2+d*χ)
        #    v[ix] = 1
        
        #    X = MultiBosonCMPSData(v, χ, d)
        #    v1 = vec(tangent_map(ψ0, X, EL, ER, Kinv))
        #    M[:, ix] = v1
        #end
        #a, _ = eigen(M)

        return mapped
    end

    transport!(v, x, d, α, xnew) = v

    optalg_LBFGS = LBFGS(;maxiter=1000, gradtol=1e-8, verbosity=2)

    ψ1, E1, grad1, numfg1, history1 = optimize(fgE, ψ, optalg_LBFGS; retract = retract,
                                    precondition = no_precondition,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!
                                    );
    ψ2, E2, grad2, numfg2, history2 = optimize(fgE, ψ, optalg_LBFGS; retract = retract,
                                    precondition = precondition,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!
                                    );

    res1 = (ψ1, E1, grad1, numfg1, history1)
    res2 = (ψ2, E2, grad2, numfg2, history2)
    return res1, res2
end

res0 = lieb_liniger_ground_state0(c, μ, ψ0);
res0[2]
res1, res2 = lieb_liniger_ground_state1(c, μ, ϕ0);
@show res1[2], res2[2]

function lieb_liniger_ground_state2(c::Real, μ::Real, ψ::Union{MultiBosonCMPSData, Nothing}=nothing)
    function fE(ψ::MultiBosonCMPSData)
        ψn = CMPSData(ψ)
        OH = kinetic(ψn) + c*point_interaction(ψn, 1) + 1.8*c*point_interaction(ψn, 2) + 0*c * point_interaction(ψn, 1, 2) - μ * particle_density(ψn, 1) - 0.4*μ * particle_density(ψn, 2)
        TM = TransferMatrix(ψn, ψn)
        envL = permute(left_env(TM), (), (1, 2))
        envR = permute(right_env(TM), (2, 1), ()) 
        return real(tr(envL * OH * envR) / tr(envL * envR))
    end
    
    function fgE(ψ::MultiBosonCMPSData)
        E = fE(ψ)
        ∂ψ = fE'(ψ) 
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
    function no_precondition(ψ::MultiBosonCMPSData, dψ::MultiBosonCMPSData)
        return dψ
    end

    function precondition(ψ0::MultiBosonCMPSData, dψ::MultiBosonCMPSData)
        ψn = CMPSData(ψ0)
        K = K_permute(K_mat(ψn, ψn))
        λ, EL = left_env(K)
        λ, ER = right_env(K)
        Kinv = Kmat_pseudo_inv(K, λ)

        ϵ = norm(dψ)
        mapped, _ = linsolve(X -> tangent_map(ψ0, X, EL, ER, Kinv) + ϵ*X, dψ, dψ; maxiter=250, ishermitian = true, isposdef = true, tol=0.1*ϵ)

        #χ, d = get_χ(ψ0), get_d(ψ0)
        #M = zeros(ComplexF64, χ^2+d*χ, χ^2+d*χ)
        #for ix in 1:(χ^2+d*χ)
        #    v = zeros(χ^2+d*χ)
        #    v[ix] = 1
        
        #    X = MultiBosonCMPSData(v, χ, d)
        #    v1 = vec(tangent_map(ψ0, X, EL, ER, Kinv))
        #    M[:, ix] = v1
        #end

        #if norm(M-M') > 1e-9
        #    @show norm(mapped)
        #    @warn "$(norm(M-M'))"
        #end
        #a, _ = eigen(Hermitian(M))
        #if a[1] < -1e-9
        #    @show a

        #    Kt = K_permute_back(K)
        #    Λs, _ = eigen(Kt)
        #    @show diag(Λs.data)
        #    @show norm(ψ0)
        #    @show norm(M)
        #    @show norm(Kinv), λ
        #    @save "not_positive_definite.jld2" ψ0
        #    error("not positive definite")
        #end

        return mapped
    end

    transport!(v, x, d, α, xnew) = v

    optalg_LBFGS1 = LBFGS(;maxiter=100, gradtol=1e-8, verbosity=2)
    optalg_LBFGS = LBFGS(;maxiter=10000, gradtol=1e-8, verbosity=2)

    ψ1, E1, grad1, numfg1, history1 = optimize(fgE, ψ, optalg_LBFGS1; retract = retract,
                                    precondition = no_precondition,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!
                                    );
    ψ2, E2, grad2, numfg2, history2 = optimize(fgE, ψ, optalg_LBFGS; retract = retract,
                                    precondition = precondition,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!
                                    );

    res1 = (ψ1, E1, grad1, numfg1, history1)
    res2 = (ψ2, E2, grad2, numfg2, history2)
    return res1, res2
end

ϕ0 = MultiBosonCMPSData(rand, 4, 2)
ψ = CMPSData(ϕ0)
particle_density(ψ, 2)

#to add to test. 
function fE(ψ::MultiBosonCMPSData)
    ψn = CMPSData(ψ)
    OH = kinetic(ψn) + c*point_interaction(ψn, 1) + c*point_interaction(ψn, 2) + 0.0*c * point_interaction(ψn, 1, 2) - μ * particle_density(ψn, 1) - μ * particle_density(ψn, 2)
    TM = TransferMatrix(ψn, ψn)
    envL = permute(left_env(TM), (), (1, 2))
    envR = permute(right_env(TM), (2, 1), ()) 
    return real(tr(envL * OH * envR) / tr(envL * envR))
end

ϕ0 = MultiBosonCMPSData(rand, 4, 2)
fE(ϕ0)
fE'(ϕ0)
res1, res2 = lieb_liniger_ground_state2(c, μ, ϕ0);
@show res1[2], res2[2]

#ϕ1 = MultiBosonCMPSData(rand, 8, 2)
ϕ1 = expand(ϕ0, 8)
res3, res4 = lieb_liniger_ground_state2(c, μ, ϕ1);
ϕ1 = res4[1]

ψ1 = CMPSData(ϕ1)
K1 = K_mat(ψ1, ψ1)
Λ, _ = eigen(K1)
@show Λ.data |> diag



@save "tmpsaving.jld2" res3 res4 