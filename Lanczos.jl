# TODO: - extend to complex numbers 
#       - struct for Krylov space 
# Later: 
#       - entanglement entropy

using LinearAlgebra

abstract type SparseSymmetricMatrix end

struct SparseHamiltonian{iT, fT} <: SparseSymmetricMatrix 
    n::iT       # dim of Hilbert space 
    B::Vector{iT}  # B stores the positions of non-zero matrix elements in the respective rows of H 
    E::Vector{iT}  # E stores for each state s the number of non-zero matrix elements <s'|H|s>
    H::Vector{fT}  # Matrix elements in the same succession as stored in B 
end 

SparseHamiltonian(N::Int) = SparseHamiltonian(Int64, Float64, N)
function SparseHamiltonian(iT::Type, fT::Type, N::Int)
    n = 2^N
    num_matrix_el = n+Int(n//2)*N
    B = Vector{iT}(undef, num_matrix_el) 
    E = Vector{iT}(undef, n)  
    H = Vector{fT}(undef, num_matrix_el)
    SparseHamiltonian{iT, fT}(n, B, E, H)
end 


abstract type AbstractKrylovSubspace end 

struct KrylovSubspace{rT, cT} <: AbstractKrylovSubspace 
    m::Int 
    Q::Array{cT, 2}
    α::Vector{rT}
    β::Vector{rT}
    L::Array{rT, 2}
    ψ0::Vector{cT}
end 

function KrylovSubspace(rT::Type, n::Int, m::Int, ψ0::AbstractVector) 
    cT = rT  # IMPROVE: cT = complex(rT)
    Q = Array{rT, 2}(undef, (n, m+1))
    α = Vector{rT}(undef, m)
    β = Vector{rT}(undef, m+1)
    L = Array{rT, 2}(undef, (m, m))
    KrylovSubspace{rT, cT}(m, Q, α, β, L, ψ0)
end 


function hinit(J::Array{T, 2}; hx=1.0, hz=1.0) where T<:Float64
    (N1, N2) = size(J)
    N1 == N2 || error("J must be square")
    N = N1
    n::Int = 2^N

    h = SparseHamiltonian(N)

    i = 1
    for s in 0:n-1 # binary representation
        e = 0

        # 1. diagonal matrix elements 
        spins = [(s & (1<<site_k) == 0) ? -1 : 1 for site_k in 0:N-1]
        intsum = sum([(J[k,l]*spins[k]*spins[l]) for k in 1:N for l in 1:k-1])
        h.B[i] = s + 1  # one-indexing
        h.H[i] = (intsum - sum(spins) * hz) / 2.0  # later on diagonal matrix elements are double counted, therefore we divide by 2 here.
        i += 1
        e += 1

        # 2. off-diagonal matrix elements 
        for site in 0:N-1
            sprime = xor(s, (1<<site))  # xor corresponds to spin flip at `site`
            if sprime > s  # only upper triangular matrix since the Hamiltonian is hermitian 
                h.B[i] = sprime + 1  # one-indexing
                h.H[i] = -hx
                i += 1 
                e += 1
            end 
        end 
        h.E[s+1] = e  # one-indexing
    end 
    return h
end 

"This function returns the *unnormalized* vector |gamma> = H |phi>, obtained by applying the Hamiltonian H
to a normalize!d state |phi>. H contains the Hamiltonian matrix elements and B their locations
in the sparse notation."
function hoperation!(h::SparseHamiltonian{Int64, T}, phi::Vector{T}, gamma::Vector{T}) where T<:Float64
    n::Int = length(phi)
    fill!(gamma, 0.0)
    i=1
    for s in 1:n
        for _ in 1:h.E[s]
            gamma[h.B[i]] += h.H[i]*phi[s]
            gamma[s] += h.H[i]*phi[h.B[i]]
            i += 1 
        end
    end 
    return gamma
end 


"normalize a vector in place"
function normalize!(v::Vector{T}) where T<:Float64
    norm = 0.0
    for i in 1:length(v)
        norm += v[i]*v[i]
    end 
    norm = sqrt(norm)
    v[:] = v[:] / norm 
    return norm 
end 

"normalize column `col` of a 2-dim array in place"
function normalize!(v::AbstractArray{T, 2}, col::Int) where T<:Float64
    norm = 0.0
    for i in 1:length(v[:, col])
        norm += v[i, col]*v[i, col]
    end 
    norm = sqrt(norm)
    v[:, col] = v[:, col] / norm 
    return norm 
end 

mutable struct LanczosInfo 
    qmin::Int  # minimum number of Lanczos vectors; on output: number of Lanczos states actually needed for convergence 
    qmax::Int  # maximum number of Lanczos vectors 
    converged::Bool # if error < eps
    error::Float64  # error, if error > eps 
end 

""""return ground state vector and Krylov subspace object 
 - v: random input vector on which the Krylov space is built
 - g_out: allocated vector into which the normalized ground state is written"""
function lanczos_sparse(h::SparseHamiltonian{Int64, T}, v::Vector{T}, g_out::Vector{T}, 
    info::LanczosInfo; eps=1e-13) where T<:Float64

    length(g_out) == length(v) || error("size mismatch")
    n = length(v)
    if info.qmax > n
        info.qmax = convert(Int, n)
    end 

    Ks = KrylovSubspace(Float64, n, info.qmax, v)

    energies = Vector{Float64}(undef, info.qmax)

    # generate the first two Lanczos states
    normalize!(v)
    Ks.Q[:, 1] = v    
    Ks.Q[:, 2] = hoperation!(h, Ks.Q[:,1], Ks.Q[:,2])
    Ks.α[1] = dot(Ks.Q[:,1], Ks.Q[:,2])
    for k = 1:n 
        Ks.Q[k,2] -= Ks.α[1] * Ks.Q[k,1]
    end 
    Ks.β[2] = normalize!(Ks.Q, 2)

    # generate the rest of the Lanczos basis
    for m = 2:info.qmin
        Ks.Q[:,m+1] = hoperation!(h, Ks.Q[:,m], Ks.Q[:,m+1])
        Ks.α[m] = dot(Ks.Q[:,m], Ks.Q[:,m+1])

        tri = SymTridiagonal(Ks.α[1:m], Ks.β[2:m]) 
        F = eigen(tri)
        Ks.L[1:m, 1:m] = F.vectors
        energies[1:m] = F.values
        perm = sortperm(energies[1:m])
        energies[1:m] = energies[perm[1:m]]
        Ks.L[1:m, 1:m] = Ks.L[1:m, perm[1:m]]

        Ks.Q[:, m+1] -= Ks.α[m] .* Ks.Q[:, m] .+ Ks.β[m]*Ks.Q[:, m-1]
        Ks.β[m+1] = normalize!(Ks.Q, m+1)            

        # explicit reorthogonalization of each newly generated state w.r.t. all 
        # other Lanczos vectors 
        for i = 1:m
            Konst = dot(Ks.Q[:, m+1], Ks.Q[:, i]) 
            Ks.Q[:, m+1] = (Ks.Q[:, m+1] .- Konst .* Ks.Q[:, i]) ./ (1 - Konst * Konst)
        end     

    end 

    j=info.qmin
    ene = energies[2]   # init with 1st excited state. Later it will be the g.s. energy of previous Lanczos basis.
    e = energies[1]     # ground state energy 
    error = abs((ene - e)/e)
    # Generate more Lanczos vectors 
    if error > eps 
        ene = e 
        for m = info.qmin+1:info.qmax

            Ks.Q[:, m+1] = hoperation!(h, Ks.Q[:, m], Ks.Q[:, m+1])
            Ks.α[m] = dot(Ks.Q[:,m], Ks.Q[:,m+1])
        
            tri = SymTridiagonal(Ks.α[1:m], Ks.β[2:m]) # ??? indices right ????
            F = eigen(tri)
            Ks.L[1:m, 1:m] = F.vectors
            energies[1:m] = F.values
            perm = sortperm(energies[1:m])
            energies[1:m] = energies[perm[1:m]]
            Ks.L[1:m, 1:m] = Ks.L[1:m, perm[1:m]]
    
            e = energies[1]
            error = abs((ene-e)/e)
            info.error = error
            if error <= eps
                info.converged = true
                break
            else 
                ene = e 
            end 

            j += 1 
            # one Lanczos vector is added to the basis 
            Ks.Q[:, m+1] -= Ks.α[m] .* Ks.Q[:, m] .+ Ks.β[m]*Ks.Q[:, m-1]

            Ks.β[m+1] = normalize!(Ks.Q, m+1)            
            # explicit reorthogonalization of each newly generated state w.r.t. all 
            # other Lanczos vectors 
            for i = 1:m
                Konst = dot(Ks.Q[:, m+1], Ks.Q[:, i]) 
                Ks.Q[:, m+1] = (Ks.Q[:, m+1] .- Konst .* Ks.Q[:, i]) ./ (1 - Konst * Konst)
            end     

        end 
    end 
    info.qmin = j # number of Lanczos states actually needed 

    # construction of the ground state from Lanczos vectors 
    fill!(g_out, 0.0)
    for k = 1:n 
        for l = 1:j
            g_out[k] = g_out[k] + Ks.L[l, 1] * Ks.Q[k, l]
        end 
    end 
    normalize!(g_out)
    return Ks, g_out

end