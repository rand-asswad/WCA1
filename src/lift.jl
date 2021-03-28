# Type definition -----------------------------------------------------------------------

"""
    Lift{T<:Complex} <: AbstractArray{T,3}

Lift object type.

### Fields
- `data::AbstractArray{Complex,3}`: lift matrix L[ω,τ,ν].
- `freq::FloatRange`: frequency bins vector.
- `time::FloatRange`: time samples vector.
- `slopes::FloatRange`: slope samples vector.
- `width::Int`: the number of samples per window.
- `sig_length::Int`: the original signal length.
- `window::Window`: callable window function or window values vector.
"""
struct Lift{T<:Complex} <: AbstractArray{T,3}
    data::AbstractArray{T,3}
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
Base.similar(L::Lift, ::Type{T}) where {T<:Complex} = Lift(similar(data(L), T), freq(L), time(L), slopes(L), width(L), sig_length(L), window(L)) 
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

grad(f::Matrix) = imgradients(f, KernelFactors.ando3)
get_ν(dSw, dSt, ε=1e-3) = abs(dSw) > ε ? -dSt/dSw : 0.0

"""
    compute_chirpiness(X::STFT; threshold = 1e-3)

Computes chirpiness `ν[ω,τ]` as a function of frequency `ω` and time `τ` from an `STFT` object.

Chirpiness is calculated `∀(τ,ω,ν) ∈ ℝ³` such that

      ∂|S|        ∂|S|
    ν ────(τ,ω) + ────(τ,ω) = 0
       ∂ω          ∂τ

"""
function compute_chirpiness(X::STFT; threshold = 1e-3)
    dSw, dSt = grad(abs.(data(X)))
    dSw *= length(freq(X))
    dSt /= step(time(X))
    get_ν.(dSw, dSt, threshold)
end

"""
    get_cauchy_quantile(data::AbstractVector; p=0.95, mode=:half_iqr)

Fits a distribution `Cauchy(x₀,γ)` to given `data` vector
and returns interval limits for the `p=95%` quantile centered around `x₀`.

The Cauchy scale parameter `γ` can be estimated as the median of the absolute
values of the data (`mode=:median`) or as half the interquartile range (`mode=:half_iqr`),
the parameters `(x₀,γ)` can also be estimated using the maximum likely method (`mode=:mle`).
"""
function get_cauchy_quantile(data::AbstractVector; p::Real=0.95, mode::Symbol=:half_iqr)
    if mode == :median
        dist = Cauchy(median(data), median(abs.(data)))
    elseif mode == :half_iqr
        dist = Cauchy(median(data), quantile(data, 0.75) - quantile(data, 0.5)/2)
    elseif mode == :mle
        dist = fit_mle(Cauchy, data)
    else
        @error("Unknown mode '$(mode)'. Available modes are [:half_iqr, :median, :mle]")
        return minimum(data), maximum(data)
    end
    return quantile(dist, 0.5 - p/2), quantile(dist, 0.5 + p/2)
end

normalize(f::FloatRange) = range(first(f)/last(f), 1.0; step=step(f)/last(f))

"""
    discretize_chirpiness(ν::AbstractMatrix, νMin::Real, νMax::Real, N::Integer=100)

Discretizes chirpiness matrix `ν` in `[νMin,νMax]` with `N` points.

Calculates the index for each value `ν[ω,τ]` in the discrete range.

                 ⎛  ν - νMin               ⎞
    index = round⎜ ─────────── (N - 1) + 1 ⎟
                 ⎝ νMax - νMin             ⎠

Returns the index matrix and the discrete range in a tuple.
"""
@inline function discretize_chirpiness(ν::AbstractMatrix, νMin::Real, νMax::Real, N::Integer=100)
    a = (N-1) / (νMax - νMin)
    b = 1.0 - ((N-1) * νMin / (νMax - νMin))
    #index = round(1 + (N-1)(ν - νMin)/(νMax - νMin)) = round(a⋅ν + b)
    index = [νMin<=x<=νMax ? round(Int, a*x + b) : nothing for x in ν]
    return index, range(νMin, νMax, length = N)
end

"""
    lift(S::STFT; νsamples=100, νlims=nothing, sparse=false, args...)

Calculates the `Lift` of a given `STFT` object for a given number of samples
in discrete range of chirpiness values.
If `νlims` is set to `nothing` the range is set to contain 95% of the chirpiness values
following a Cauchy distribution estimated using `WCA1.get_cauchy_quantile`
whose parameters can be controlled using `args`.

# Extended help

### Keyword Arguments
- `νsamples::Int=100`: number of chirpiness samples.
- `νlims::Union{Nothing,Tuple{Real,Real}}=nothing`: limits of chirpiness range.
- `sparse::Bool=false`: calculate lift as sparse 3D array.
"""
function lift(S::STFT; νsamples::Int=100, νlims::Union{Nothing,Tuple{T,T}}=nothing, sparse::Bool=false, args...) where {T<:Real}
    ν = compute_chirpiness(S; args...)
    νMin, νMax = νlims === nothing ? get_cauchy_quantile(vec(ν); args...) : νlims
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

"""
    project(L::Lift)

Projects a Lift into STFT.

              ∞
    S(τ,ω) = ∫ L(t,ω,ν) dν
             -∞

"""
function project(L::Lift)
    S = sum(L, dims = 3) |> x->dropdims(x, dims = 3)
    STFT(S, freq(L), time(L), width(L), sig_length(L), window(L))
end

# Plotting methods ----------------------------------------------------------------------

Plots.plot(L::Lift, args...; kwargs...) = plot(project(L), args...; kwargs...)
Plots.plot!(L::Lift, args...; kwargs...) = plot!(project(L), args...; kwargs...)

"""
    plot(L::Lift; mode=:spectrogram, yscale=:identity, axislabels=true, kwargs...)

Plot recipe for Lift object. Plots spectrogram or phase heatmap of projected Lift (STFT object).

`mode` is either `:spectrogram` (default) or `:phase`
If `axislabels` is true, the default axis labels are displayed.
Can be overriden using the appropriate keyword arguments.
The `yscale` parameter allows plotting logarithmic frequency scale.
"""
plot(::Lift)