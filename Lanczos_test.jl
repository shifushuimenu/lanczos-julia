include("Lanczos.jl")

N = 6
J = zeros(N, N)
for i = 1:N-1
    J[i,i+1] = 1.0
    J[i+1, i] = 1.0
end 
J[1, N] = J[N, 1] = 1.0

display(J)

for ipar in 1:40

    hx = 0.0 + ipar * 0.1
    (B, E, H) = hinit(J, hx=hx, hz=0.0)

    # println("matrix elements")
    # display(H)
    # println("locations of matrix elements")
    # display(B)
    # println("number of connecting states")
    # display(E)
    # println(" ")
    
    info = LanczosInfo(4, 20, false, 1.0)
    gamma = hoperation!(B, E, H, ones(2^N), ones(2^N))
    g_out = lanczos_sparse(B, E, H, ones(2^N), gamma, info)
    # measure ground state energy
    #display(g_out)
    g2 = zeros(2^N)
    g2 = hoperation!(B, E, H, g_out, g2)
    GS_energy = sum(g2 .* g_out) / N
    println(hx, " ", GS_energy)

end 
