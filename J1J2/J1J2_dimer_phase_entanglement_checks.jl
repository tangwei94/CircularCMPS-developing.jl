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

    scattering, reweighted = CircularCMPS.half_chain_singular_values_testtool(ψL, ψ, β);

    scattering_norms = norm.(scattering.data)
    scattering_norms /= maximum(scattering_norms)
    reweighted_norms = norm.(reweighted.data)
    reweighted_norms /= maximum(reweighted_norms)
    fig1, ax, hm = heatmap(log10.(scattering_norms), colorrange=(-3, 0), colormap=:Blues)
    Colorbar(fig1[:, end+1], hm)
    fig2, ax, hm = heatmap(log10.(reweighted_norms), colorrange=(-6, 0), colormap=:Blues)
    Colorbar(fig2[:, end+1], hm)

    return fig1, fig2
end

β = 68.88623433758424;
@load "J1J2/deprecated-check-again/data/dimer_phase_beta$(β).jld2" ψs 
ψ = ψs[end];
ψL = Wmat1 * ψ;

ψ1 = ψ;
ψL1 = ψL;

β = βs[end];
@load "J1J2/gauged_cooling/data/dimer_phase_beta$(β).jld2" ψ 
ψL = Wmat * ψ;

fig, fig2 = show_analysis_results(ψL, ψ, β);
@show fig
save("J1J2/J1J2_dimer_phase_MA.pdf", fig)
fig, fig2 = show_analysis_results(ψL1, ψ1, β);
@show fig
save("J1J2/J1J2_dimer_phase_MA_deprecated.pdf", fig)

#--->
# tried to compute the "scattering matrix" for the J1J2 gauged and deprecated, but didn't find any significant difference between these two
# from the regular analysis plot, only the double-sided ES shows significant difference. but it is also unclear what this means
#<---