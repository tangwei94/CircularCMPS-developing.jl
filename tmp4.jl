using LinearAlgebra, TensorKit
using OptimKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

kept_states = 5

hz = 0.05
T, Wmat = xxz_af_cmpo(1; hz=hz)

α = 2^(1/4)
βs = 0.32 * α .^ (0:32)

β0 = βs[33]
@load "heisenberg/results/cooling/heisenberg_hz0.05_beta$(β0)-2.jld2" ψ
β = βs[33]

function power_iteration1(T::CMPO, Wmat::Matrix{<:Number}, β::Real, ψ::CMPSData, alg::PowerMethod)
    printstyled("\n[ power_iteration: doing power method with 
        β = $(β)
        maxiter_power = $(alg.maxiter_power) 
        spect_shifting = $(alg.spect_shifting)
        maxχ = $(alg.maxχ)
        tol_fidel=$(alg.tol_fidel)
        tol_ES=$(alg.tol_ES)
        verbosity=$(alg.verbosity)
        maxiter_compress=$(alg.maxiter_compress) \n"; bold=true, color=:red)

    f, E, var, fidel = Inf, Inf, Inf, Inf
    function _f(ϕ::CMPSData)
        K = Tψ2 * Tψ2
        expK, ln_norm = finite_env(K, β)
        return real(-ln_ovlp(ϕ, Tψ2, β) - ln_ovlp(Tψ2, ϕ, β) + ln_ovlp(ϕ, ϕ, β) + ln_norm)
    end
    ϕs = CMPSData[CircularCMPS.gauge_fixing(ψ, β)]
    xs0 = zeros(Float64, kept_states - 1)

    for ix in 1:alg.maxiter_power
        Tψ = left_canonical(T*ψ)[2]
        if alg.spect_shifting > 0
            ψ = left_canonical(ψ)[2]
            Tψ = direct_sum(Tψ, ψ; α=log(alg.spect_shifting)/β/2)
        end

        χ, err = CircularCMPS.suggest_χ(Tψ, β; tol=alg.tol_ES, maxχ=alg.maxχ, minχ=get_χ(ψ))
        printstyled("[ power_iteration: next χ: $(χ), possible error: $(err) \n"; bold=true)
        
        tol_compress = min(0.1*abs(fidel), 100*alg.tol_fidel) # gradually lowering the tolerance for compression to save time
        if length(ϕs) == kept_states
            function _tmpf(xs::Vector{<:Real})
                ϕx = sum(xs .* ϕs[1:end-1]) + (1-sum(xs)) * ϕs[end]
                K = Tψ * Tψ
                _, ln_norm = finite_env(K, β)
                return real(-ln_ovlp(ϕx, Tψ, β) - ln_ovlp(Tψ, ϕx, β) + ln_ovlp(ϕx, ϕx, β) + ln_norm)
            end
            function _tmpfg(xs::Vector{<:Real})
                gs = real.(_tmpf'(xs))
                return _tmpf(xs), gs
            end

            xs0, _, _, _, _ = optimize(_tmpfg, xs0, LBFGS(; maxiter=100, gradtol=tol_compress, verbosity=2))
            ϕx = sum(xs0 .* ϕs[1:end-1]) + (1-sum(xs0)) * ϕs[end]
            deleteat!(ϕs, 1)
        else
            ϕx = ψ
        end

        ψ1 = compress(Tψ, χ, β; init=ϕx, maxiter=alg.maxiter_compress, verbosity=alg.verbosity, tol=tol_compress)
        fidel = real(2*ln_ovlp(ψ, ψ1, β) - ln_ovlp(ψ, ψ, β) - ln_ovlp(ψ1, ψ1, β))

        if dim(space(ϕs[end])) == dim(space(ψ1)) 
            ϕ1 = CircularCMPS.gauge_fixing(ϕs[end], ψ1, β)
            push!(ϕs, ϕ1)
        else
            ϕs = CMPSData[CircularCMPS.gauge_fixing(ψ1, β)]
        end

        ψ = ψ1
        ψL = W_mul(Wmat, ψ)
        f = free_energy(T, ψL, ψ, β)
        E = energy(T, ψL, ψ, β)
        var = variance(T, ψ, β)
        printstyled("[ power_iteration: ix, f, E, var, fidel $(ix) $(f) $(E) $(var) $(fidel) \n"; color=:red)

        ix > 2 && abs(fidel) < alg.tol_fidel && break
    end
    return ψ, f, E, var
end

ψ1, _, _, _ =  power_iteration1(T, Wmat, β, ψ, PowerMethod(tol_ES=1e-8, tol_fidel=1e-8))
