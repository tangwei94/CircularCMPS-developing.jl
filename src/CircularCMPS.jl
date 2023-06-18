module CircularCMPS

__precompile__(true)

using LinearAlgebra
using TensorKit, TensorOperations, KrylovKit, TensorKitAD, TensorKitManifolds
using ChainRules, ChainRulesCore, Zygote
using OptimKit
using LoopVectorization 
using Tullio
using JLD2
using Printf
#using FastGaussQuadrature

# utils.jl
export MPSBondTensor, GenericMPSTensor, MPSTensor, fill_data!, randomize!, K_permute, K_permute_back, K_otimes, herm_reg_inv 

# cmps.jl
export AbstractCMPSData, CMPSData, get_χ, get_d, get_matrices, transfer_matrix, transfer_matrix_dagger, left_canonical, right_canonical, expand, K_mat, finite_env, rescale

# operators.jl
export kinetic, particle_density, point_interaction, pairing

# ground_state.jl 
export lieb_liniger_ground_state

# excited_state.jl 
export θ2, θ3, AbstractCoeffs, Coeff2, Coeff3, ExcitationData, gauge_fixing_map, effective_N, effective_H, Kac_Moody_gen

# optim_alg.jl 
export CircularCMPSRiemannian, minimize, leading_boundary_cmps

# cmpo.jl 
export AbstractCMPO, CMPO, ln_ovlp, compress, direct_sum, W_mul, variance, free_energy, energy, klein, inner

# cmpo_zoo.jl
export ising_cmpo, xxz_af_cmpo, xxz_fm_cmpo, heisenberg_j1j2_cmpo

# power_iteration.jl 
export power_iteration

# cMPS code for continuous Hamiltonians
include("utils.jl");
include("cmps.jl");
include("cmpsAD.jl");
include("operators.jl");
include("ground_state.jl")
include("excited_state.jl");
include("entanglement.jl")

# cMPO code
include("optim_alg.jl");
include("cmpo.jl");
include("cmpoAD.jl");
include("cmpo_zoo.jl");
include("power_iteration.jl")
include("power_iteration_1.jl")

end
