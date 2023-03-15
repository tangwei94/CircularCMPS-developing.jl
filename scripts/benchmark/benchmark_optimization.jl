using CircularCMPS
using JLD2
using TensorKit, LinearAlgebra, KrylovKit, OptimKit
using CairoMakie
using LiebLinigerBA

function lieb_liniger_ground_state_riemannian(c::Real, μ::Real, L::Real, ψ0::Union{CMPSData, Nothing}=nothing; precond_choice::Symbol=:default, max_steps::Int=10000)
    function fE(ψ::CMPSData)
        OH = kinetic(ψ) + c*point_interaction(ψ) - μ * particle_density(ψ)
        expK, _ = finite_env(K_mat(ψ, ψ), L)
        return real(tr(expK * OH))
    end

    # TODO. implement gradientcheck: check inner(d, g) = gradient with respect to alpha obtained from finite difference.
    function fgE(ψ::CMPSData)
        E = fE(ψ)
        ∂ψ = fE'(ψ) 
        dQ = zero(∂ψ.Q) #- sum(ψ.Rs' .* ∂ψ.Rs)
        dRs = ∂ψ.Rs .- ψ.Rs .* Ref(∂ψ.Q) #the second term makes sure it is a true gradient! 

        return E, CMPSData(dQ, dRs)
    end

    function inner(ψ, ψ1::CMPSData, ψ2::CMPSData)
        #return real(dot(ψ1.Q, ψ2.Q) + sum(dot.(ψ1.Rs, ψ2.Rs))) #TODO. clarify the cases with or withou factor of 2. depends on how to define the complex gradient
        return real(sum(dot.(ψ1.Rs, ψ2.Rs))) 
    end

    function retract(ψ::CMPSData, dψ::CMPSData, α::Real)
        Rs = ψ.Rs .+ α .* dψ.Rs 
        Q = ψ.Q - α * sum(ψ.Rs' .* dψ.Rs) - 0.5 * α^2 * sum(dψ.Rs' .* dψ.Rs)
        ψ1 = CMPSData(Q, Rs)
        #ψ1 = left_canonical(ψ1)[2]
        return ψ1, dψ
    end

    function scale!(dψ::CMPSData, α::Number)
        dψ.Q = dψ.Q * α
        dψ.Rs .= dψ.Rs .* α
        return dψ
    end

    function add!(dψ::CMPSData, dψ1::CMPSData, α::Number) 
        dψ.Q += dψ1.Q * α
        dψ.Rs .+= dψ1.Rs .* α
        return dψ
    end

    # only for comparison
    function no_precondition(ψ::CMPSData, dψ::CMPSData)
        return dψ
    end

    function precondition(ψ::CMPSData, dψ::CMPSData)
        fK = transfer_matrix(ψ, ψ)

        # solve the fixed point equation
        init = similar(ψ.Q, CircularCMPS._firstspace(ψ.Q)←CircularCMPS._firstspace(ψ.Q))
        randomize!(init);
        _, vrs, _ = eigsolve(fK, init, 1, :LR)
        vr = vrs[1]

        δ = inner(ψ, dψ, dψ)

        P = herm_reg_inv(vr, max(1e-12, 1e-3*δ)) 

        Q = dψ.Q  
        Rs = dψ.Rs .* Ref(P)

        return CMPSData(Q, Rs)
    end

    function precondition1(ψ::CMPSData, dψ::CMPSData)
        Kmat = K_mat(ψ, ψ)
        expK, _ = finite_env(Kmat, L)

        δ = inner(ψ, dψ, dψ)

        P = herm_reg_inv(permute(expK, (2, 4), (1,3)), max(1e-12, 1e-3*δ))
        P = permute(P, (1, 4), (2, 3))

        V = dψ.Q 
        Ws = MPSBondTensor[]
        for W in dψ.Rs
            @tensor W1[-1; -2] := W[1, 2] * P[2, -1, 1, -2]
            push!(Ws, W1)
        end

        return CMPSData(V, Ws)
    end

    if precond_choice == :default
        precondition_used = precondition
    elseif precond_choice == :new1
        precondition_used = precondition1 
    else
        precondition_used = no_precondition
    end

    transport!(v, x, d, α, xnew) = v

    optalg_LBFGS = LBFGS(;maxiter=max_steps, gradtol=1e-6, verbosity=2)

    ψ = left_canonical(ψ0)[2]
    ψ1, E, grad, numfg, history = optimize(fgE, ψ, optalg_LBFGS; retract = retract,
                                    precondition = precondition_used,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!
                                    );
    return ψ1, E, grad, numfg, history 
end

function lieb_liniger_ground_state_ordinary(c::Real, μ::Real, L::Real, ψ0::Union{CMPSData, Nothing}=nothing; max_steps::Int=10000)
    function fE(ψ::CMPSData)
        OH = kinetic(ψ) + c*point_interaction(ψ) - μ * particle_density(ψ)
        expK, _ = finite_env(K_mat(ψ, ψ), L)
        return real(tr(expK * OH))
    end

    function fgE(ψ::CMPSData)
        E = fE(ψ)
        ∂ψ = fE'(ψ) 
        dQ = ∂ψ.Q
        dRs = ∂ψ.Rs 

        return E, CMPSData(dQ, dRs)
    end

    function inner(ψ, ψ1::CMPSData, ψ2::CMPSData)
        return real(dot(ψ1.Q, ψ2.Q) + sum(dot.(ψ1.Rs, ψ2.Rs))) #TODO. clarify the cases with or withou factor of 2. depends on how to define the complex gradient
    end

    function retract(ψ::CMPSData, dψ::CMPSData, α::Real)
        Rs = ψ.Rs .+ α .* dψ.Rs 
        Q = ψ.Q + α * dψ.Q 
        ψ1 = CMPSData(Q, Rs)
        return ψ1, dψ
    end

    function scale!(dψ::CMPSData, α::Number)
        dψ.Q = dψ.Q * α
        dψ.Rs .= dψ.Rs .* α
        return dψ
    end

    function add!(dψ::CMPSData, dψ1::CMPSData, α::Number) 
        dψ.Q += dψ1.Q * α
        dψ.Rs .+= dψ1.Rs .* α
        return dψ
    end

    # only for comparison
    function no_precondition(ψ::CMPSData, dψ::CMPSData)
        return dψ
    end

    transport!(v, x, d, α, xnew) = v

    optalg_LBFGS = LBFGS(;maxiter=max_steps, gradtol=1e-6, verbosity=2)

    ψ = left_canonical(ψ0)[2]
    ψ1, E, grad, numfg, history = optimize(fgE, ψ, optalg_LBFGS; retract = retract,
                                    precondition = no_precondition,
                                    inner = inner, transport! =transport!,
                                    scale! = scale!, add! = add!
                                    );
    return ψ1, E, grad, numfg, history 

end

c, L = 1, 32

## BA solution
N = Int(L) 
μ = get_mu(c, L, N)

Nexact = N / L
ψgs = ground_state(c, L, N)
Eexact = energy(ψgs, μ) / L

## cMPS optimization
χ = 12
ψ0 = CMPSData(rand, χ, 1)
max_steps = 10000

ψ1, E1, grad1, numfg1, history1 = lieb_liniger_ground_state_riemannian(c, μ, L, ψ0, precond_choice = :default, max_steps=max_steps);
ψ2, E2, grad2, numfg2, history2 = lieb_liniger_ground_state_riemannian(c, μ, L, ψ0, precond_choice = :new1, max_steps=max_steps);
ψ3, E3, grad3, numfg3, history3 = lieb_liniger_ground_state_riemannian(c, μ, L, ψ0, precond_choice = :none, max_steps=max_steps);
ψ4, E4, grad4, numfg4, history4 = lieb_liniger_ground_state_ordinary(c, μ, L, ψ0, max_steps=max_steps);

@save "tmp_history_c$(c)_N$(N)_L$(L)_chi$(χ).jld2" history1 history2 history3 history4
@load "tmp_history_c$(c)_N$(N)_L$(L)_chi$(χ).jld2" history1 history2 history3 history4

error_in_E(x) = abs((x - Eexact) / Eexact) 
@show error_in_E.([E1, E2, E3, E4])

#font1 = Makie.to_font("/home/wtang/.local/share/fonts/cmunrm.ttf")
font2 = Makie.to_font("/home/wtang/.local/share/fonts/STIXTwoText-Regular.otf")

N1, _ = size(history1)
N2, _ = size(history2)
N3, _ = size(history3)
N4, _ = size(history4)
Nmin = 1

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 600), fonts=(; regular=font2))
gf = fig[1, 1] = GridLayout() 
gl = fig[2, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = L"\text{steps}",
        ylabel = L"\text{error in } E",
        yscale = log10, 
        )
lines!(ax1, Nmin:N4, error_in_E.(history4[Nmin:N4, 1]), label=L"\text{plain}")
lines!(ax1, Nmin:N3, error_in_E.(history3[Nmin:N3, 1]), label=L"\text{Riemannian, w/o precond.}")
lines!(ax1, Nmin:N1, error_in_E.(history1[Nmin:N1, 1]), label=L"\text{Riemannian, w/ precond v1.}")
lines!(ax1, Nmin:N2, error_in_E.(history2[Nmin:N2, 1]), label=L"\text{Riemannian, w/ precond v2.}")

@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = L"\text{steps}",
        ylabel = L"\text{grad norm}",
        yscale = log10, 
        )
lines!(ax2, Nmin:N4, history4[Nmin:N4, 2])
lines!(ax2, Nmin:N3, history3[Nmin:N3, 2])
lines!(ax2, Nmin:N1, history1[Nmin:N1, 2])
lines!(ax2, Nmin:N2, history2[Nmin:N2, 2])

@show fig

for (label, layout) in zip(["(a)", "(b)"], [gf[1, 1], gf[2, 1]])
    Label(layout[1, 1, TopLeft()], label, 
    padding = (0, -25, -30, 0), 
    halign = :right
    )
end

#axislegend(ax1, position=:rb, framevisible=false)
leg = Legend(gl[1,1], ax1, orientation=:horizontal, framecolor=:lightgrey, labelsize=15)
leg.nbanks = 2

@show fig
Makie.save("fig-benchmark-riemannian-c$(c)_N$(N)_L$(L)_chi$(χ).pdf", fig)