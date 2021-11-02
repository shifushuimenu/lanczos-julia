# TODO: - extend to complex numbers 
#       - 
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

""""return ground state vector
 - v: random input vector on which the Krylov space is built
 - g_out: allocated vector into which the normalized ground state is written"""
function lanczos_sparse(h::SparseHamiltonian{Int64, T}, v::Vector{T}, g_out::Vector{T}, 
    info::LanczosInfo; eps=1e-13) where T<:Float64

    length(g_out) == length(v) || error("size mismatch")
    n = length(v)
    if info.qmax > n
        info.qmax = convert(Int, n)
    end 

    p = Array{Float64, 2}(undef, (n, info.qmax+1))
    a = Vector{Float64}(undef, info.qmax)
    b = Vector{Float64}(undef, info.qmax+1)
    energies = Vector{Float64}(undef, info.qmax)
    fill!(a, 0.0); fill!(b, 0.0); fill!(energies, 0.0); fill!(p, 0.0)
    L = Array{Float64, 2}(undef, (info.qmax, info.qmax))
    fill!(L, 0.0)
    
    # generate the first two Lanczos states
    normalize!(v)
    p[:, 1] = v    
    p[:, 2] = hoperation!(h, p[:,1], p[:,2])
    a[1] = dot(p[:,1], p[:,2])
    for k = 1:n 
        p[k,2] -= a[1] * p[k,1]
    end 
    b[2] = normalize!(p, 2)

    # generate the rest of the Lanczos basis
    for m = 2:info.qmin
        p[:,m+1] = hoperation!(h, p[:,m], p[:,m+1])
        a[m] = dot(p[:,m], p[:,m+1])

        tri = SymTridiagonal(a[1:m], b[2:m]) 
        F = eigen(tri)
        L[1:m, 1:m] = F.vectors
        energies[1:m] = F.values
        perm = sortperm(energies[1:m])
        energies[1:m] = energies[perm[1:m]]
        L[1:m, 1:m] = L[1:m, perm[1:m]]

        p[:, m+1] -= a[m] .* p[:, m] .+ b[m]*p[:, m-1]
        b[m+1] = normalize!(p, m+1)            

        # explicit reorthogonalization of each newly generated state w.r.t. all 
        # other Lanczos vectors 
        for i = 1:m
            Konst = dot(p[:, m+1], p[:, i]) 
            p[:, m+1] = (p[:, m+1] .- Konst .* p[:, i]) ./ (1 - Konst * Konst)
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

            p[:, m+1] = hoperation!(h, p[:, m], p[:, m+1])
            a[m] = dot(p[:,m], p[:,m+1])
        
            tri = SymTridiagonal(a[1:m], b[2:m]) # ??? indices right ????
            F = eigen(tri)
            L[1:m, 1:m] = F.vectors
            energies[1:m] = F.values
            perm = sortperm(energies[1:m])
            energies[1:m] = energies[perm[1:m]]
            L[1:m, 1:m] = L[1:m, perm[1:m]]
    
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
            p[:, m+1] -= a[m] .* p[:, m] .+ b[m]*p[:, m-1]

            b[m+1] = normalize!(p, m+1)            
            # explicit reorthogonalization of each newly generated state w.r.t. all 
            # other Lanczos vectors 
            for i = 1:m
                Konst = dot(p[:, m+1], p[:, i]) 
                p[:, m+1] = (p[:, m+1] .- Konst .* p[:, i]) ./ (1 - Konst * Konst)
            end     

        end 
    end 
    info.qmin = j # number of Lanczos states actually needed 

    # construction of the ground state from Lanczos vectors 
    fill!(g_out, 0.0)
    for k = 1:n 
        for l = 1:j
            g_out[k] = g_out[k] + L[l, 1] * p[k, l]
        end 
    end 
    normalize!(g_out)
    return g_out

end