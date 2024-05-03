@with_kw struct PowerMethod 
    maxiter_power::Int = 200
    spect_shifting::Real = 0 # useful when the system breaks the translational symmetry at low temperatures; otherwise it can be zero
    maxχ::Int = 40
    fixχ::Int = -1
    tol_fidel::Real = 1e-8 # The tolerance for fidelity is preferably not greater than 1e-8, otherwise the results will be inaccurate.
    tol_ES::Real = 1e-10 # For each tol_ES, run the cooling process from starting point again, instead of using the previous result as the initial state.
    maxiter_compress::Int = 250
    verbosity::Int = 1
end

function power_iteration(T::CMPO, Wmat::Matrix{<:Number}, β::Real, ψ::CMPSData, alg::PowerMethod)
    printstyled("\n[ power_iteration: doing power method with 
        β = $(β)
        maxiter_power = $(alg.maxiter_power) 
        spect_shifting = $(alg.spect_shifting)
        maxχ = $(alg.maxχ)
        fixχ = $(alg.fixχ)
        tol_fidel=$(alg.tol_fidel)
        tol_ES=$(alg.tol_ES)
        verbosity=$(alg.verbosity)
        maxiter_compress=$(alg.maxiter_compress) \n"; bold=true, color=:red)

    f, E, var, fidel = Inf, Inf, Inf, Inf

    for ix in 1:alg.maxiter_power

        Tψ = left_canonical(T*ψ)[2]
        if alg.spect_shifting > 0
            ψ = left_canonical(ψ)[2]
            Tψ = direct_sum(Tψ, ψ; α=log(alg.spect_shifting)/β/2)
        end

        if alg.fixχ < 0
            χ = suggest_χ(Wmat, Tψ, β; tol=alg.tol_ES, maxχ=alg.maxχ, minχ=get_χ(ψ))
        else 
            χ = alg.fixχ
        end

        printstyled("[ power_iteration: next χ: $(χ) \n"; bold=true)

        tol_compress = max(min(0.1*abs(fidel), 1e-6), 0.1*alg.tol_fidel) # gradually lowering the tolerance for compression to save time
        ψ1 = compress(Tψ, χ, β; init=ψ, maxiter=alg.maxiter_compress, verbosity=alg.verbosity, tol=tol_compress)
        fidel = real(2*ln_ovlp(ψ, ψ1, β) - ln_ovlp(ψ, ψ, β) - ln_ovlp(ψ1, ψ1, β))

        ψ = ψ1
        ψL = W_mul(Wmat, ψ)
        f = free_energy(T, ψL, ψ, β)
        E = energy(T, ψL, ψ, β)
        var = variance(T, ψ, β)
        printstyled("[ power_iteration: ix, f, E, var, fidel: $(ix) $(f) $(E) $(var) $(fidel) \n"; color=:red)

        ix > 2 && abs(fidel) < alg.tol_fidel && break
    end
    return ψ, f, E, var
end
