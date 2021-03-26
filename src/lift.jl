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
struct Lift{T<:Real} <: AbstractArray{Complex{T},3}
    data::AbstractArray{Complex{T},3}
    freq::FloatRange
    time::FloatRange
    slopes::FloatRange
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
Base.similar(L::Lift{T}) where {T} = similar(L, T)
Base.similar(L::Lift, ::Type{T}) where {T<:Real} = Lift(similar(data(L), Complex{T}), freq(L), time(L), slopes(L), width(L), sig_length(L), window(L)) 
function Base.show(io::IO, ::MIME"text/plain", L::Lift{T}) where T
    n, m, l = size(L)
    if issparse(L) 
        xnnz = nnz(data(L))
        print(io, n, "×", m, "×", l, " Lift{", T, "} with ", xnnz, " stored ", xnnz == 1 ? "entry" : "entries")
        if xnnz != 0
            print(io, ":")
            show(IOContext(io, :typeinfo => eltype(L)), L)
        end
    else
        print(io, n, "×", m, "×", l, " Lift{", T, "}")
    end
end
Base.show(io::IO, L::Lift) = show(convert(IOContext, io), L::Lift)
Base.show(io::IOContext, L::Lift) = show(io, data(L))

# SparseArrays functions
SparseArrays.issparse(L::Lift) = issparse(data(L))
SparseArrays.nnz(L::Lift) = nnz(data(L))
SparseArrays.nonzeros(L::Lift) = nonzeros(data(L))
SparseArrays.findnz(L::Lift) = findnz(data(L))
SparseArrays.sparse(L::Lift) = Lift(sparse(data(L)), freq(L), time(L), slopes(L), width(L), sig_length(L), window(L)) 

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
        @error("Unknown mode '$(mode)'. Available modes are [:interquartile, :median, :mle]")
        return minimum(data), maximum(data)
    end
    return quantile(dist, 0.5 - p/2), quantile(dist, 0.5 + p/2)
end


#normalize(x) = (first(x)/last(x)):(step(x)/last(x)):1
normalize(f::FloatRange) = range(first(f)/last(f), 1.0; step=step(f)/last(f))

@inline function discretize_chirpiness(ν, νMin, νMax, N = 100)
    ν_vals = range(νMin, νMax, length = N)
    #index = round(1 + (N-1)(ν - νMin)/(νMax - νMin)) = round(a⋅ν + b)
    a = (N-1) / (νMax - νMin)
    b = 1.0 - ((N-1) * νMin / (νMax - νMin))
    index = [νMin<=x<=νMax ? round(Int, a*x + b) : nothing for x in ν]
    return index, ν_vals
end

function lift(S::STFT; νsamples::Int=100, νlims::Union{Nothing,Tuple{T,T}}=nothing, sparse=false, args...) where {T<:Real}
    ν = compute_chirpiness(S; args...)
    νMin, νMax = νlims === nothing ? auto_cut(vec(ν); args...) : νlims
    νindex, νrange = discretize_chirpiness(ν, νMin, νMax, νsamples)

    init_zeros = sparse ? spzeros : zeros
    L = init_zeros(eltype(S), size(S,1), size(S,2), νsamples)
    for j = 1:size(S,2), i = 1:size(S,1)
        if νindex[i,j] !== nothing
            L[i,j,νindex[i,j]] = S[i,j]
        else
            L[i,j,:] = ones(νsamples) * (S[i,j] / νsamples)
        end
    end

    Lift(L, freq(S), time(S), νrange, width(S), sig_length(S), window(S))
end

function project(L::Lift)
    f = sum(L, dims = 3) |> x->dropdims(x, dims = 3)
    STFT(f, freq(L), time(L), width(L), sig_length(L), window(L))
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