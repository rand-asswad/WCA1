# Type definition -----------------------------------------------------------------------

"""
    Lift{T<:Complex} <: AbstractArray{T,3}

Lift object type.

### Fields
- `data::Array{Complex,3}`: lift matrix I[ω,τ,ν].
- `freq::Union{Frequencies, AbstractRange}`: frequency bins vector.
- `time::FloatRange{Float64}`: time samples vector.
- `slopes::FloatRange{Float64}`: slope samples vector.
- `width::Int`: the number of samples per window.
- `sig_length::Int`: the original signal length.
- `window::Window`: callable window function or window values vector.
"""
struct Lift{T<:Complex} <: AbstractArray{T,3}
    data::Array{T,3}
    freq::Frequencies
    time::FloatRange{Float64}
    slopes::FloatRange{Float64}
    width::Int
    sig_length::Int
    window::Window
end

data(L::Lift) = L.data
freq(L::Lift) = L.freq
Libc.time(L::Lift) = L.time
slopes(L::Lift) = L.slopes
width(L::Lift) = L.width
sig_length(L::Lift) = L.sig_length
window(L::Lift) = L.window

# Base functions for interface inheritance
Base.size(L::Lift, args...) = size(data(L), args...)
Base.getindex(L::Lift, args...) = getindex(data(L), args...)
Base.setindex!(L::Lift, v, args...) = setindex!(data(L), v, args...)
Base.similar(L::Lift, ::Type{T}) where {T} = Lift(similar(data(L)), similar(freq(L)), similar(time(L)), similar(slopes(L)), width(L), sig_length(L), window(L)) 
#Base.showarg(io::IO, L::Lift, toplevel) = toplevel && print(io, "Lift{$(eltype(L))}")

# Lift implementation -------------------------------------------------------------------

grad(f::Matrix{T}) where {T<:Real} = imgradients(f, KernelFactors.ando3)

get_ν(dw, dt; ε=1e-3) = abs(dw) > ε ? -dt/dw : 0.0
function compute_chirpiness(X::STFT; threshold = 1e-3, args...)
    dw, dt = grad(abs.(data(X)))
    dw *= length(freq(X))
    dt /= step(time(X))
    get_ν.(dw, dt; ε=threshold)
end

function auto_cut(data; p = 0.95, mode=:interquartile)
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


function slopes(S::STFT, νsamples=100; νlims::Union{Nothing,Tuple{T,T}}=nothing, args...) where {T<:Number}
    G = compute_chirpiness(S; args...)
    a, b = νlims === nothing ? auto_cut(vec(G); args...) : νlims
    compute_slope_matrix(G, a, b, νsamples)
end

function lift(m::STFT; νsamples::Int=100, args...)
    slopeMatrix, Z = slopes(m, νsamples; args...)

    imgLift = zeros(eltype(m), (size(m,1), size(m,2), νsamples))
    for i=1:size(m,1), j=1:size(m,2)
        if slopeMatrix[i,j] !== nothing
            imgLift[i,j,slopeMatrix[i,j]] = m[i,j]
        else
            imgLift[i,j,:] = ones(νsamples) * (m[i,j] / νsamples)
        end
    end

    Lift(imgLift, freq(m), time(m), Z, width(m), sig_length(m), window(m))
end

function project(Φ::Lift)
    f = sum(Φ[:,:,:], dims = 3) |> x->dropdims(x, dims = 3)
    STFT(f, freq(Φ), time(Φ), width(Φ), sig_length(Φ), window(Φ))
end

# Plotting methods ----------------------------------------------------------------------

Plots.plot(L::Lift, args...; kwargs...) = plot(project(L), args...; kwargs...)
Plots.plot!(L::Lift, args...; kwargs...) = plot!(project(L), args...; kwargs...)

"""
    plot(L::Lift; mode::Symbol=:spectrogram, yscale::Symbol=:identity, axislabels::Bool=true, kwargs...)

Plot recipe for Lift object. Plots spectrogram or phase heatmap of projected Lift (STFT object).

`mode` is either `:spectrogram` (default) or `:phase`
If `axislabels` is true, the default axis labels are displayed.
Can be overriden using the appropriate keyword arguments.
The `yscale` parameter allows plotting logarithmic frequency scale.
"""
plot(::Lift)