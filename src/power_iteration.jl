function power_iteration(T::CMPO, Wmat::Matrix{<:Number}, β::Real, ψ::CMPSData; maxiter::Int=200, DIIS_D::Int=5, spect_shifting::Real=1)
    printstyled("\n[ power_iteration: doing power method for $(space(ψ)) \n"; bold=true)
    χ = dim(space(ψ))
    tmpψs = CMPSData[]
    Δψs = CMPSData[]
    f, E, var = Inf, Inf, Inf
    for ix in 1:maxiter
        Tψ = left_canonical(T*ψ)[2]
        if spect_shifting > 0
            ψ = left_canonical(ψ)[2]
            Tψ = direct_sum(Tψ, ψ; α=log(spect_shifting)/β/2)
        end
        ψ1 = compress(Tψ, χ, β; init=ψ, maxiter=100)
        fidel = real(2*ln_ovlp(ψ, ψ1, β) - ln_ovlp(ψ, ψ, β) - ln_ovlp(ψ1, ψ1, β))

        Δ = Inf
        if ix >= 2 
            ψ1 = gauge_fixing(tmpψs[end], ψ1, β)
            Δψ = ψ1 - tmpψs[end]
            push!(Δψs, Δψ)
            Δ = norm(Δψ)
            printstyled("[ DIIS: norm(Δψ): $(Δ), fidel: $(fidel) \n"; color=:red, bold=true)
        end
        if fidel < -0.001 || Δ/χ > 0.01 
            printstyled("[ DIIS: too far from convergence. clean up \n"; color=:red, bold=true)
            tmpψs = CMPSData[]
            Δψs = CMPSData[]
        end
        if length(Δψs) >= 2 && norm(Δψs[end]) > norm(Δψs[end-1]) # handle the situation where Δ oscillates
            printstyled("[ DIIS: remove abnormal error estimates \n"; color=:red, bold=true)
            tmpψs = CMPSData[]
            Δψs = CMPSData[]
        end

        # DIIS 
        if length(Δψs) == DIIS_D
            printstyled("[ DIIS: using DIIS \n"; color=:red, bold=true)

            Bs = zeros(ComplexF64, DIIS_D, DIIS_D)
            for ix in 1:DIIS_D, iy in 1:DIIS_D 
                Bs[ix, iy] = dot(Δψs[ix].Q, Δψs[iy].Q) + 
                             sum(dot.(Δψs[ix].Rs, Δψs[iy].Rs))
                (ix == iy) && (Bs[ix, iy] *= 1.02)
            end
            Rhs = ones(ComplexF64, DIIS_D)
            cs = Bs \ Rhs
            cs = cs ./ sum(cs)
            ψ = sum(cs .* tmpψs)

            tmpψs = CMPSData[]
            Δψs = CMPSData[]
        else
            ψ = ψ1
        end

        if length(tmpψs) == 0
            push!(tmpψs, gauge_fixing(ψ, β))
        else 
            # ψ already gauge fixed
            push!(tmpψs, ψ)
        end

        ψL = W_mul(Wmat, ψ)
        f = free_energy(T, ψL, ψ, β)
        E = energy(T, ψL, ψ, β)
        var = variance(T, ψ, β)
        printstyled("\n[ power_iteration: ix, f, E, var $(ix) $(f) $(E) $(var) \n"; bold=true)

        abs(fidel) < 1e-8 && break
    end
    return ψ, f, E, var
end

function gauge_fixing(ϕ::CMPSData, β::Real)
    χ = dim(space(ϕ))
    _, U = eigen(ϕ.Q)
    α = ln_ovlp(ϕ, ϕ, β) / β
    ϕ1 = CMPSData(inv(U) * ϕ.Q * U - α /2 * id(ℂ^χ), Ref(inv(U)) .* ϕ.Rs .* Ref(U))
    return ϕ1
end
function gauge_fixing(ϕ1::CMPSData, ϕ2::CMPSData, β::Real; verbosity::Int=0, gradtol::Real=1e-10, maxiter::Int=100)
    χ = dim(space(ϕ1))

    ϕ1 = gauge_fixing(ϕ1, β)
    ϕ2 = gauge_fixing(ϕ2, β)
    function _f(V)
        ΔQ = V * ϕ2.Q * V' - ϕ1.Q
        ΔRs = Ref(V) .* ϕ2.Rs .* Ref(V') .- ϕ1.Rs
        return sqrt(norm(ΔQ)^2 + norm(ΔRs)^2)
    end
    function _fg(V)
        dV = _f'(V)
        gV = Unitary.project!(dV, V)
        return _f(V), gV
    end

    V = id(Matrix{ComplexF64}, ℂ^χ) 

    optalg_LBFGS = LBFGS(;gradtol=gradtol, maxiter=maxiter, verbosity=verbosity)
    V, fvalue, grad, numfg, history = optimize(_fg, V, optalg_LBFGS; 
                                                transport! = Unitary.transport!,
                                                retract = Unitary.retract,
                                                inner = Unitary.inner,
                                                scale! = Unitary.rmul!,
                                                add! =(V, gV, α) -> axpy!(α, gV, V))
    if norm(grad) > gradtol
        printstyled("[ DIIS: gauge fixing doesn't fully converge, gradnorm $(norm(grad))\n"; bold=true, color=:red)
    end

    ϕ2_f = CMPSData(V * ϕ2.Q * V', Ref(V) .* ϕ2.Rs .* Ref(V'))
    return ϕ2_f
end
