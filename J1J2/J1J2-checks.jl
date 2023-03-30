using LinearAlgebra, TensorKit
using CairoMakie
using JLD2 
using Revise
using CircularCMPS

T, W = ising_cmpo(1.0)

CircularCMPS.cmpo_checker(T)

σz = CircularCMPS.pauli.σz
σx = CircularCMPS.pauli.σx
σy = CircularCMPS.pauli.σy
Id = CircularCMPS.pauli.Id

J1, J2 = 1, 0.5
T, W = heisenberg_j1j2_cmpo(J1, J2)

H3_from_cmpo = CircularCMPS.cmpo_checker(T)

S_halves = [σx / 2, σy / 2, σz / 2]
H3 = J1 * sum(S_halves .⊗ S_halves .⊗ Ref(Id) + Ref(Id) .⊗ S_halves .⊗ S_halves) + J2 * sum(S_halves .⊗ Ref(Id) .⊗ S_halves)

H3 - H3_from_cmpo |> norm