@with_kw struct PowerMethod 
    maxiter_power::Int = 200
    spect_shifting::Real = 1 
    maxχ::Int = 40
    tol_fidel::Real = 1e-8
    tol_ES::Real = 1e-6
    maxiter_compress::Int = 100
    verbosity::Int = 1
end

function power_iteration(T::CMPO, Wmat::Matrix{<:Number}, β::Real, ψ::CMPSData, alg::PowerMethod)
    printstyled("\n[ power_iteration: doing power method with 
        maxiter_power = $(alg.maxiter_power)          
        spect_shifting = $(alg.spect_shifting)          
        maxχ = $(alg.maxχ)          
        tol_fidel=$(alg.tol_fidel) 
        tol_ES=$(alg.tol_ES) 
        verbosity=$(alg.verbosity) 
        maxiter_compress=$(alg.maxiter_compress) \n"; bold=true, color=:red)

    f, E, var, fidel = Inf, Inf, Inf, Inf

    #χ, err = suggest_χ(ψ, β; tol=alg.tol_ES, maxχ=alg.maxχ, minχ=2)
    #printstyled("[ power_iteration: init χ: $(χ), possible error: $(err) \n"; bold=true)
    #ψ = compress(ψ, χ, β; init=ψ, maxiter=alg.maxiter_compress, verbosity=alg.verbosity)

    for ix in 1:alg.maxiter_power

        Tψ = left_canonical(T*ψ)[2]
        if alg.spect_shifting > 0
            ψ = left_canonical(ψ)[2]
            Tψ = direct_sum(Tψ, ψ; α=log(alg.spect_shifting)/β/2)
        end

        χ, err = suggest_χ(Tψ, β; tol=alg.tol_ES, maxχ=alg.maxχ, minχ=get_χ(ψ))
        printstyled("[ power_iteration: next χ: $(χ), possible error: $(err) \n"; bold=true)

        tol_compress = min(0.1*abs(fidel), 1e-6)
        ψ1 = compress(Tψ, χ, β; init=ψ, maxiter=alg.maxiter_compress, verbosity=alg.verbosity, tol=tol_compress)
        fidel = real(2*ln_ovlp(ψ, ψ1, β) - ln_ovlp(ψ, ψ, β) - ln_ovlp(ψ1, ψ1, β))

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
