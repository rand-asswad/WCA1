# Type definition -----------------------------------------------------------------------

const FloatRange{T<:AbstractFloat} = Union{AbstractRange{T}, AbstractVector{T}}

"""
    Signal{T<:Real} <: AbstractVector{T}

Signal object type

### Fields
- `data::Vector{T}`: signal data vector signal[t].
- `fs::Float64`: signal sampling rate.
- `time::FloatRange{Float64}`: time samples range.
"""
struct Signal{T<:Real} <: AbstractVector{T}
    data::Vector{T}
    fs::Real
    time::FloatRange
end

Signal(data::AbstractVector{T}, fs::Real) where {T<:Real} = Signal(data, fs, (0:length(data) - 1) / fs)

data(s::Signal) = s.data
fs(s::Signal) = s.fs
Libc.time(s::Signal) = s.time
duration(s::Signal) = length(s) / fs(s)

# Base functions for interface inheritance
Base.size(s::Signal, args...) = size(data(s), args...)
Base.getindex(s::Signal, args...) = getindex(data(s), args...)
Base.setindex!(s::Signal, v, args...) = setindex!(data(s), v, args...)
Base.similar(s::Signal, ::Type{T}) where {T} = Signal(similar(data(s)), fs(s), time(s)) 
Base.showarg(io::IO, s::Signal, toplevel) = toplevel && print(io, "Signal{$(eltype(s))} with sample rate $(Int(fs(s))) Hz")

# Signal Arithmatics

function Base.:+(x::Signal, y::Signal)
    # raise error if fs(x) != fs(y)
    Signal(data(x) + data(y), fs(x))
end

Base.:*(x::Signal, y::Number) = Signal(data(x) * y, fs(x))
Base.:*(x::Number, y::Signal) = y * x

# Signal functions
energy(s::AbstractVector) = sum(s .* s)
power(s::AbstractVector) = energy(s) / length(s)

hopsamples(s::Signal, hop::Integer) = Signal(s[1:hop:length(s)], fs(s) / hop)

# Input/Output --------------------------------------------------------------------------

function stereo2mono(data::AbstractArray)
    if ndims(data) <= 2
        return size(data, 2) == 1 ? vec(data) : sum(data, dims=2) / size(x, 2)
    elseif size(data, 1) * size(data, 2) == length(data)
        return stereo2mono(dropdims(data, dims=tuple(collect(3:ndims(data))...)))
    else
        @error("Cannot convert stereo signal of size $(size(data))")
    end
end

"""
    wavread(filename; subrange=:)

Wrapper around `WAV.wavread` function that reads a WAV file and returns a `WCA1.Signal` object.
The samples are converted to floating point values between -1.0 and 1.0.
Multi-channel (stereo) files are converted to mono-channel signals.
The `subrange` parameter allos to return a subset of the audio samples.

See `?WAV.wavread` for more details.
"""
function wavread(filename::AbstractString; subrange=(:))
    data, fs = WAV.wavread(filename; format="double", subrange=subrange)
    Signal(stereo2mono(data), fs)
end

"""
    wavwrite(s::Signal, filename)

Wrapper around `WAV.wavwrite` function that writes a `WCA1.Signal` object into a WAV file.
"""
wavwrite(s::Signal, filename::AbstractString) = WAV.wavwrite(data(s), filename, Fs = fs(s))

# Plotting methods ----------------------------------------------------------------------

@recipe function plot_signal(s::Signal; axislabels=true, normalize=false)
    legend --> false
    xguide --> (axislabels ? "Time (s)" : "")
    yguide --> (axislabels ? "Amplitude" : "")
    if normalize; ylims --> (-1, 1) end
    return time(s), data(s)
end

"""
    plot(s::Signal; axislabels=true, normalize=false, kwargs...)

Plot recipe for Signal object.
If `axislabels` is true, the default axis labels are displayed.
If `normalize` is true, the signal amplitude is normalized in [-1,+1].
Both parameters can be overriden using the appropriate keyword arguments.
"""
plot(::Signal)