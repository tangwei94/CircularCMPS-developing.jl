function power_iteration(T::CMPO, Wmat::Matrix{<:Number}, β::Real, ψ::CMPSData, tol::Real; maxiter::Int=200, spect_shifting::Real=1, maxχ::Int=40)
    printstyled("\n[ power_iteration: doing power method with tol=$(tol) \n"; bold=true)
    f, E, var = Inf, Inf, Inf
    for ix in 1:maxiter
        Tψ = left_canonical(T*ψ)[2]
        if spect_shifting > 0
            ψ = left_canonical(ψ)[2]
            Tψ = direct_sum(Tψ, ψ; α=log(spect_shifting)/β/2)
        end
        χ, err = suggest_χ(Tψ, β, tol; maxχ=maxχ, minχ=get_χ(ψ))
        printstyled("[ power_iteration: next χ: $(χ), possible error: $(err) \n"; bold=true)
        ψ1 = compress(Tψ, χ, β; init=ψ, maxiter=100)
        fidel = real(2*ln_ovlp(ψ, ψ1, β) - ln_ovlp(ψ, ψ, β) - ln_ovlp(ψ1, ψ1, β))

        ψ = ψ1
        ψL = W_mul(Wmat, ψ)
        f = free_energy(T, ψL, ψ, β)
        E = energy(T, ψL, ψ, β)
        var = variance(T, ψ, β)
        printstyled("[ power_iteration: ix, f, E, var, fidel $(ix) $(f) $(E) $(var) $(fidel) \n"; color=:red)

        abs(fidel) < tol && break
    end
    return ψ, f, E, var
end
