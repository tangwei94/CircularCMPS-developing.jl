module CircularCMPS

__precompile__(true)

using LinearAlgebra
using TensorKit, TensorOperations, KrylovKit, TensorKitAD
using ChainRules, ChainRulesCore, Zygote
using OptimKit
using LoopVectorization 
using Tullio
using JLD2
using Printf
using Polyester
#using FastGaussQuadrature
# Write your package code here.

# utils.jl
export MPSBondTensor,
       GenericMPSTensor,
       MPSTensor,
       fill_data!,
       randomize!,
       K_permute,
       K_permute_back,
       K_otimes,
       herm_reg_inv 

# cmps.jl
export AbstractCMPSData, 
       CMPSData, 
       get_χ,
       get_d, 
       get_matrices, 
       transfer_matrix, 
       transfer_matrix_dagger, 
       left_canonical, 
       right_canonical, 
       expand, 
       K_mat, 
       finite_env, 
       rescale

# operators.jl
export kinetic, 
       particle_density, 
       point_interaction, 
       pairing

# ground_state.jl 
export lieb_liniger_ground_state

# excited_state.jl 
export θ2, 
       θ3, 
       AbstractCoeffs,
       Coeff2,
       Coeff3, 
       ExcitationData,
       gauge_fixing_map,
       effective_N, 
       effective_H, 
       Kac_Moody_gen

include("utils.jl");
include("cmps.jl");
include("cmpsAD.jl");
include("operators.jl");
include("ground_state.jl")
include("excited_state.jl");

end
