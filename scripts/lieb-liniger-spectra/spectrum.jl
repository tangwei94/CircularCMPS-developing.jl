using CircularCMPS
using JLD2 
using TensorKit, LinearAlgebra 
using LiebLinigerBA
using CairoMakie

#c, μ, L = 1, 1.42, 8
c, μ, L = 1, 1.426, 16

function fE(ψ::CMPSData)
    OH = kinetic(ψ) + c * point_interaction(ψ) - μ * particle_density(ψ)
    expK, _ = finite_env(K_mat(ψ, ψ), L)
    return real(tr(expK * OH))
end

function fN(ψ::CMPSData)
    ON = particle_density(ψ)
    expK, _ = finite_env(K_mat(ψ, ψ), L)
    return real(tr(expK * ON))
end

@load "tmpdata/cmps_c$(c)_mu$(μ)_L$(L).jld2" ψ3 ψ4 ψ5

ψ = ψ5
χ = get_χ(ψ)

Egs_cmps = fE(ψ) * L 
Ngs_cmps = fN(ψ) * L 

# variance of N
opN = sum(K_otimes.(ψ.Rs, ψ.Rs))
opN2 = sum(K_otimes.(ψ.Rs .* ψ.Rs, ψ.Rs .* ψ.Rs))
Kmat = K_mat(ψ, ψ)
expK, α = finite_env(Kmat, L)
C2 = Coeff2(Kmat, 0, L)
#VarN = C2(opN, opN) + tr(expK * opN2) + Ngs_cmps / L - Ngs_cmps 
VarN = C2(opN, opN) + Ngs_cmps / L - Ngs_cmps

momenta_cmps = [0.0]
energies_cmps = [Egs_cmps]
numbers_cmps = [Ngs_cmps]

for k in -3:3
    global momenta_cmps, energies_cmps, numbers_cmps
    @load "tmpdata1/excitation_c$(c)_mu$(μ)_L$(L)_k$(k)_chi$(χ).jld2" H1 N1 M1

    @show norm(H1 - H1') / χ^4 
    @show norm(N1 - N1') / χ^4
    @show norm(M1 - M1') / χ^4

    H̃1 = sqrt(inv(N1)) * H1 * sqrt(inv(N1))
    M̃1 = sqrt(inv(N1)) * M1 * sqrt(inv(N1))
    Es, Vs = eigen(Hermitian(H̃1))

    energies_cmps = vcat(energies_cmps, Es)
    momenta_cmps = vcat(momenta_cmps, fill(k*2*pi/L, length(Es)))
    numbers_cmps = vcat(numbers_cmps, real.(diag(Vs' * M̃1 * Vs)))
end

msk_cmps = abs.(numbers_cmps .- Ngs) .< 0.5 # use colors to mark the accuracy

### BA solution
# go through different particle numbers
@show get_mu(c, L, L)

Nmax = 50
ψs = [ground_state(c, L, Nx) for Nx in 2:Nmax]
Es = [energy(ψi, μ) for ψi in ψs]

# find the ground state
Egs, gs_index = findmin(Es)
ψgs = ψs[gs_index]
Ngs = particle_number(ψgs)

ψps = LLBAState[];
ψms = LLBAState[];
for n in 1:(Ngs÷2)
    push!(ψps, ground_state(c, L, Ngs+n))
    push!(ψms, ground_state(c, L, Ngs-n))
end

Egs = energy(ψgs, μ)
ϵE = abs((Egs_cmps - Egs) / Egs)
ϵN = abs((Ngs_cmps - Ngs) / Ngs)

ph_actions = [[0], [-1], [-2], [-1, -1], [-3], [-2, -1], [-1, -1, -1]];
states = LLBAState[];
ψph1 = ph_excitation(ψgs, [0], [-1]);
ΔE = energy(ψph1, μ) - Egs
v = ΔE / (2*pi / L)

for qL in ph_actions
    for qR in ph_actions
        for ψx in [ψgs; ψps; ψms]
            if min(length(qL), length(qR)) ≤ length(ψx.ns) / 2
                push!(states, ph_excitation(ψx, qL, qR))
            end
        end
    end
end

momenta = momentum.(states)
energies = energy.(states, μ)
numbers = [length(state.ns) for state in states]
scale_E(x::Float64) = (x - Egs) / ΔE
scale_E(ψ::LLBAState, μ::Float64) = (energy(ψ, μ) - Egs) / ΔE
scale_p(x::Float64) = x * L / (2 * pi)

msk_ba = abs.(numbers .- Ngs) .< 1e-4

# plot the spectrum. 
font1 = Makie.to_font("/home/wtang/.local/share/fonts/STIXTwoText-Regular.otf")
fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400), fonts=(; regular=font1))
#gf = fig[1, 1] = GridLayout() 
#gl = fig[2, 1] = GridLayout()

ax1 = Axis(fig[1, 1], 
        xlabel = L"p L / 2\pi - 0.05 (N-N_0)",
        ylabel = L"\text{rescaled } \Delta E", 
        xticks = -4:1:4,
        yticks = 0:1:3,
        #title = L"\text{full spectrum}"
        )

ylims!(ax1, (-0.15, 3.5))

sc1_ba = scatter!(ax1, scale_p.(momenta) .- 0.05 .* (numbers .- Ngs), scale_E.(energies), color=:red3, marker='X', markersize=15, label="BA")
sc1_cmps = scatter!(ax1, scale_p.(momenta_cmps) .- 0.05 .* (numbers_cmps .- Ngs), scale_E.(energies_cmps), color=:blue3, marker='O', markersize=15, label="cMPS")

axislegend(ax1, position=:lb, framevisible=true, labelsize=14)
@show fig

#ax2 = Axis(fig[2, 1], 
#        xlabel = L"p L / 2\pi",
#        ylabel = L"\text{rescaled } \Delta E", 
#        xticks = -4:1:4,
#        yticks = 0:1:3,
#        title = L"N = N_0 \text{ spectrum}"
#        )
#
#ylims!(ax2, (-0.15, 4.0))
#
#sc2_ba = scatter!(ax2, scale_p.(momenta[msk_ba]), scale_E.(energies[msk_ba]), color=:red3, marker='X', markersize=15)
#sc2_cmps = scatter!(ax2, scale_p.(momenta_cmps[msk_cmps]), scale_E.(energies_cmps[msk_cmps]), color=:blue3, marker='O', markersize=15)

@show fig

Makie.save("fig-cmps-spect-c$(c)-L$(L)-N$(Ngs)-chi$(χ).pdf", fig)