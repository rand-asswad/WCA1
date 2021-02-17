struct Lift{T}
    lift::Array{T, 3}
    freq::Frequencies
    time::FloatRange{Float64}
    slopes::FloatRange{Float64}
    width::Int
    sig_length::Int
    window::Union{Function,AbstractVector,Nothing}
end

Base.eltype(m::Lift{T}) where T = eltype(lift(m))
Base.size(m::Lift{T}, etc...) where T = size(lift(m), etc...)
Base.getindex(m::Lift{T}, etc...) where T = getindex(lift(m), etc...)

width(L::Lift) = L.width
freq(L::Lift) = L.freq
lift(L::Lift) = L.lift
Libc.time(L::Lift) = L.time
slopes(L::Lift) = L.slopes

or_length(L::Lift) = L.sig_length
window(L::Lift) = L.window

grad(f::Matrix{T}) where {T<:Real} = imgradients(f, KernelFactors.ando3)

function compute_chirpiness(X::STFT; threshold = 1e-3)
    dw, dt = grad(abs.(data(X)))
    dw *= length(freq(X))
    dt /= step(time(X))
    @. ifelse(abs(dw) > threshold, -dt / dw, 0)
end

get_ν(dw, dt; ϵ=1e-3) = abs(dw) > ϵ ? -dt/dw : 0.
function brdcast_chirpiness(X::STFT; threshold = 1e-3)
    dw, dt = grad(abs.(data(X)))
    dw *= length(freq(X))
    dt /= step(time(X))
    get_ν.(dw, dt; ϵ=threshold)
end

function old_school(X::STFT; threshold = 1e-3)
    dw, dt = grad(abs.(X.stft))
    F = length(X.freq) * step(X.time)
    ν = similar(dw)
    for i in eachindex(ν)
        ν[i] = abs(dw[i]) > threshold ? -dt[i]/(dw[i]*F) : 0.
    end
    ν
end

function compute_slopes(SS; threshold = 1e-3, args...)
    M = abs.(data(SS))
    gx, gy = grad(M)
    
    gx = gx * length(SS.freq)
    gy = gy / step(SS.time)
    
    G = similar(M)
    for i in 1:length(M)
        G[i] = abs(gx[i]) > threshold  ? -gy[i]/gx[i] : 0.
    end
    G
end

function auto_cut(data; p = 0.95, mode=:median)
    if mode == :median
        dist = Cauchy(median(data), median(abs.(data)))
    elseif mode == :interquartile
        dist = Cauchy(median(data), quantile(data, 0.75) - quantile(data, 0.5)/2)
    elseif mode == :mle
        dist = fit_mle(Cauchy, data)
    else
        @error("Unknown mode '$(mode)'")
        return minimum(data), maximum(data)
    end
    return quantile(dist, 0.5 - p/2), quantile(dist, 0.5 + p/2)
end


#normalize(x) = (first(x)/last(x)):(step(x)/last(x)):1
normalize(f) = range(first(f)/last(f), 1.0; step=step(f)/last(f))

rng(t) = last(t) - first(t)

zs(M, νMin, νMax, N) = range(νMin,νMax, length = N)

function compute_slope_matrix(M, νMin, νMax, N = 100; args...)
    Z = zs(M, νMin, νMax, N)
    slopeMatrix = similar(M, Union{Int,Nothing})
    A = round.(Int,(M .- first(Z))*(N-1)/(rng(Z)) .+ 1)
    for i in 1:length(M)
        if first(Z) <= M[i] <= last(Z)
            slopeMatrix[i] = A[i]
        elseif M[i] > last(Z)
                slopeMatrix[i] = nothing#N
        elseif M[i] < first(Z)
                    slopeMatrix[i] = nothing#1
        else
            slopeMatrix[i] = nothing
        end
    end
    slopeMatrix, Z
end


function slopes(S::STFT, N = 100; p=0.95, mode=:median, args...)
    G = compute_slopes(S; args...)
    a, b = auto_cut(vec(G); p=p, mode=mode)
    compute_slope_matrix(G, a, b, N)
end

# should remove vmin and vmax
function lift(m::STFT; νMin=-1, νMax=1, N::Int = 100,  args...)
    slopeMatrix, Z = slopes(m, N; args...)

    imgLift = zeros(eltype(m),(size(m,1),size(m,2),N))
    for i=1:size(m,1), j=1:size(m,2)
        if slopeMatrix[i,j] !== nothing
            imgLift[i,j,slopeMatrix[i,j]] = m[i,j]
        else
            imgLift[i,j,:] = ones(N)*m[i,j] / N
        end
    end

    Lift(imgLift, freq(m), time(m), Z, width(m), or_length(m), window(m))
end

function project(Φ::Lift)
    f = sum(Φ[:,:,:], dims = 3) |> x->dropdims(x, dims = 3)
    STFT(f, freq(Φ), time(Φ), width(Φ), or_length(Φ), window(Φ))
end

Plots.plot(L::Lift, args...; kwargs...) = plot(project(L), args...; kwargs...)