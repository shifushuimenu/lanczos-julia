using Test  

include("Lanczos.jl")

@testset "6-site TIsing" begin 
    N = 6
    J = zeros(N, N)
    for i = 1:N-1
        J[i,i+1] = 1.0
        J[i+1, i] = 1.0
    end 
    J[1, N] = J[N, 1] = 1.0

    h = hinit(J, hx=0.1, hz=0.0)
    info = LanczosInfo(4, 20, false, 1.0)
    gamma = hoperation!(h, ones(2^N), ones(2^N))
    g_out = lanczos_sparse(h, ones(2^N), gamma, info)
    # measure ground state energy
    g2 = zeros(2^N)
    g2 = hoperation!(h, g_out, g2)
    GS_energy = sum(g2 .* g_out) / N

    @test isapprox(GS_energy, -1.0025016072757338, atol=1e-8 )
end 