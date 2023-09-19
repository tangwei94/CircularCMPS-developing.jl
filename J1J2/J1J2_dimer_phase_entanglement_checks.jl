using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

J1, J2 = 1, 0.5
T, Wmat = heisenberg_j1j2_cmpo(J1, J2)
T1, Wmat1 = CircularCMPS.heisenberg_j1j2_cmpo_deprecated(J1, J2)

α = 2^(1/4)
βs = 0.64 * α .^ (0:27)

# importance scattering check
function show_analysis_results(ψL, ψ, β)

    χ2 = length(ψ.Q.data)

    #ΛRR = half_chain_singular_values(ψ, β)
    #ΛLL = half_chain_singular_values(ψL, β)
    #ΛRR_diag = diag(real.(ΛRR.data))
    #ΛLL_diag = diag(real.(ΛLL.data))

    scattering, reweighted = CircularCMPS.half_chain_singular_values_testtool(ψL, ψ, β);

    scattering_norms = norm.(scattering.data)
    scattering_norms /= maximum(scattering_norms)
    reweighted_norms = norm.(reweighted.data)
    reweighted_norms /= maximum(reweighted_norms)
    fig1, ax, hm = heatmap(log10.(scattering_norms), colorrange=(-3, 0), colormap=:Blues)
    Colorbar(fig1[:, end+1], hm)
    fig2, ax, hm = heatmap(log10.(reweighted_norms), colorrange=(-6, 0), colormap=:Blues)
    Colorbar(fig2[:, end+1], hm)

    function meas1(M)
        map(1:χ2) do ix0
            return sum(M[ix0, 1:end]) + sum(M[1:end, ix0])
        end
    end

    fig3, ax2, _ = lines(log10.(meas1(scattering_norms)))
    ax3 = Axis(fig3[2, 1])
    lines!(ax3, log10.(meas1(reweighted_norms)))
    #scatter!(ax3, log10.(norm.(ΛRR_diag)), marker='x')
    #scatter!(ax3, log10.(norm.(ΛLL_diag)), marker='o')

    return fig1, fig2, fig3
end

β = 68.88623433758424;
@load "J1J2/deprecated-check-again/data/dimer_phase_beta$(β).jld2" ψs 
ψ = ψs[end];
ψL = Wmat1 * ψ;

#@load "J1J2/deprecated-check-again/data/dimer_phase_beta$(β)_cooling.jld2" ψ 
#ψL = Wmat1 * ψ

β = βs[end];
@load "J1J2/gauged_cooling/data/dimer_phase_beta$(β).jld2" ψ 
ψL = Wmat * ψ;

Λ1, U1 = eigen(Hermitian(Wmat1))
α = 1
G = U1 * Diagonal(sqrt.(norm.(Λ1)) .^ α) 

G * Wmat * G' - Wmat1 |> norm
ψ1 = convert(Matrix, inv(G')) * ψ;
ψL1 = (G * Wmat * G') * ψ1;

fig, fig2, fig3 = show_analysis_results(ψL, ψ, β);
@show fig
@show fig2
fig, fig2, fig3 = show_analysis_results(ψL1, ψ1, β);
@show fig
@show fig2
#@show fig3

a = CircularCMPS.half_chain_singular_values_testtool(ψL, ψ, β) 


#--->
# tried to compute the "scattering matrix" for the J1J2 gauged and deprecated, but didn't find any significant difference between these two
# from the regular analysis plot, only the double-sided ES shows significant difference. but it is also unclear what this means
#<---