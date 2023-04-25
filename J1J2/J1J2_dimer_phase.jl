using LinearAlgebra, TensorKit
using ChainRules, TensorKitAD, TensorKitManifolds, OptimKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

function gauge_fixing(ϕ::CMPSData, β::Real)
    χ = dim(space(ϕ))
    _, U = eigen(ϕ.Q)
    α = ln_ovlp(ϕ, ϕ, β) / β
    ϕ1 = CMPSData(inv(U) * ϕ.Q * U - α /2 * id(ℂ^χ), Ref(inv(U)) .* ϕ.Rs .* Ref(U))
    return ϕ1
end
function gauge_fixing(ϕ1::CMPSData, ϕ2::CMPSData, β::Real; verbosity::Int=0, gradtol::Real=1e-12, maxiter::Int=100)
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

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)

χs = [3, 6, 9, 12]

ψ0 = CMPSData(T.Q, T.Ls)

α = 2^(1/4)
βs = 1.28 * α .^ (0:23)

steps = 1:100

# power method, shift spectrum
#tmpψs = CMPSData[]
#vars1 = Float64[]
#for β in βs[end-1:end-1]
#    global ψ0
#    ψ = ψ0
#    fs, Es, vars = Float64[], Float64[], Float64[]
#    ψs = CMPSData[]
#    for χ in χs
#        f, E, var = fill(-999, 3)
#        for ix in 1:40 
#            Tψ = left_canonical(T*ψ)[2]
#            ψ = left_canonical(ψ)[2]
#            Tψ = direct_sum(Tψ, ψ)
#            ψ1 = compress(Tψ, χ, β; init=ψ, maxiter=100)
#            fidel = real(2*ln_ovlp(ψ, ψ1, β) - ln_ovlp(ψ, ψ, β) - ln_ovlp(ψ1, ψ1, β))
#            ψ = ψ1
#            ψL = W_mul(Wmat, ψ)
#
#            push!(tmpψs, ψ)
#
#            f = free_energy(T, ψL, ψ, β)
#            E = energy(T, ψL, ψ, β)
#            var = variance(T, ψ, β)
#            @show χ, ix, f, E, (E-f)*β, var, fidel
#            push!(vars1, fidel)
#        end
#        push!(fs, f)
#        push!(Es, E)
#        push!(vars, var)
#        push!(ψs, ψ)
#
#        @save "J1J2/gauged-results/dimer_phase_beta$(β).jld2" fs Es vars ψs
#    end
#end

TensorKit.norm(ϕ::CMPSData) =  sqrt(norm(ϕ.Q)^2 + norm(ϕ.Rs)^2)

DIIS_D = 5

for β in βs[end:-1:end-10]
    global ψ0
    ψ = ψ0
    fs, Es, vars = Float64[], Float64[], Float64[]
    ψs = CMPSData[]
    for χ in χs
        f, E, var = fill(-999, 3)
        tmpψs = CMPSData[]
        Δψs = CMPSData[]
        fidels = Float64[]
        for ix in steps
            Tψ = left_canonical(T*ψ)[2]
            ψ = left_canonical(ψ)[2]
            Tψ = direct_sum(Tψ, ψ)
            ψ1 = compress(Tψ, χ, β; init=ψ, maxiter=100)
            fidel = real(2*ln_ovlp(ψ, ψ1, β) - ln_ovlp(ψ, ψ, β) - ln_ovlp(ψ1, ψ1, β))

            Δ = Inf
            if ix >= 2 
                ψ1 = gauge_fixing(tmpψs[end], ψ1, β)
                Δϕ = ψ1 - tmpψs[end]
                push!(Δψs, Δϕ)
                Δ = norm(Δϕ) 
                printstyled("[ DIIS: norm(Δψ): $(Δ), fidel:$(fidel) \n"; color=:red, bold=true)
            end
            if fidel < -0.01 || Δ/χ > 0.01 
                printstyled("[ DIIS: too far from convergence. clean up \n"; color=:red, bold=true)
                tmpψs = CMPSData[]#tmpψs[3:end]
                Δψs = CMPSData[]#Δψs[3:end] 
            end
            if length(Δψs) == 2 && norm(Δψs[2]) > norm(Δψs[1])
                printstyled("[ DIIS: abnormal error estimate. clean up \n"; color=:red, bold=true)
                tmpψs = CMPSData[]#tmpψs[2:end]
                Δψs = CMPSData[]#Δψs[2:end] 
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

                tmpψs = CMPSData[]#tmpψs[3:end]
                Δψs = CMPSData[]#Δψs[3:end] 
                #tmpψs = tmpψs[end-1:end]
                #Δψs = Δψs[end-1:end] 
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
            @show χ, ix, f, E, (E- f)*β, var, fidel
            push!(fidels, fidel)
            if abs(fidel) < 1e-8
                break 
            end
        end
        push!(fs, f)
        push!(Es, E)
        push!(vars, var)
        push!(ψs, ψ)
        @save "J1J2/gauged-results/dimer_phase_beta$(β).jld2" fs Es vars ψs
    end
end


#vs = Vector{Float64}[]
#for ix in 1:100
    #v1 = M * v
    #v = v1 / norm(v1)
    #push!(vs, v)
#end 

#Δvs = vs[2:end] - vs[1:end-1]
#Bs = zeros(ComplexF64, 10, 10)
#ix0 = 0
#for ix in 1:10, iy in 1:10
    #Bs[ix, iy] = dot(Δvs[ix+ix0], Δvs[iy+ix0]) 
    ##(ix == iy) && (Bs[ix, iy] *= 1.02)
#end
#Bs
#Rhs = ones(ComplexF64, 10)
#cs = Bs \ Rhs
#cs / norm(cs)
#vf = sum((cs / norm(cs)) .* vs[1:10])

#vf = vs[10]
#dot(M * vf, M* vf) * dot(vf, vf) / dot(M*vf, vf) / dot(vf, M*vf)