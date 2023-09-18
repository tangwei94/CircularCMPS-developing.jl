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

    ΛLR = half_chain_singular_values(ψL, ψ, β)
    ΛRR = half_chain_singular_values(ψ, β)
    ΛLL = half_chain_singular_values(ψL, β)

    ΛLR_diag = diag(real.(ΛLR.data)) 
    ΛRR_diag = diag(real.(ΛRR.data))
    ΛLL_diag = diag(real.(ΛLL.data))

    scattering, reweighted_scattering = CircularCMPS.half_chain_singular_values_testtool(ψL, ψ, β)

    fig, ax, hm = heatmap(log10.(norm.(scattering.data)), colorrange=(-6, 1), colormap=:Blues)
    Colorbar(fig[:, end+1], hm)
    @show fig

    function importance_meas(M)
        map(1:χ2) do ix0
            return sum(M[ix0, 1:end]) + sum(M[1:end, ix0])
        end
    end

    fig2, ax2, _ = lines(log10.(importance_meas(norm.(scattering.data))))
    scatter!(ax2, log10.(norm.(ΛRR_diag)), marker='x')
    scatter!(ax2, log10.(norm.(ΛLL_diag)), marker='o')

    ΛL, = eigh(scattering) 
    #fig3, ax, hm = heatmap(log10.(norm.((U*sqrt(S)).data)), colorrange=(-4, 0), colormap=:Blues)
    #Colorbar(fig3[:, end+1], hm)
    #@show fig3
     
    return fig, fig2, diag(ΛL.data)
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

fig, fig2, λL = show_analysis_results(ψL, ψ, β);
@show fig
@show fig2
@show λL





#--->
# tried to compute the "scattering matrix" for the J1J2 gauged and deprecated, but didn't find any significant difference between these two
# from the regular analysis plot, only the double-sided ES shows significant difference. but it is also unclear what this means
#<---