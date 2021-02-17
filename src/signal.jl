# Type definition -----------------------------------------------------------------------

const FloatRange{T} = StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T}}

"""
    Signal

Signal object type

### Fields
- `data::Vector{<:Real}`: signal data vector signal[t].
- `fs::Float64`: signal sampling rate.
- `time::FloatRange{Float64}`: time samples range.
"""
struct Signal{T<:Real}
    data::Vector{T}
    fs::Float64
    time::FloatRange{Float64}
end

data(s::Signal) = s.data
fs(s::Signal) = s.fs
Libc.time(s::Signal) = s.time

Base.eltype(s::Signal) = eltype(data(s))
Base.size(s::Signal, args...) = size(data(s), args...)
Base.length(s::Signal, args...) = length(data(s), args...)
Base.getindex(s::Signal, args...) = getindex(data(s), args...)
Base.extrema(s::Signal) = extrema(data(s))

duration(s::Signal) = length(s) / fs(s)

Signal(data::Vector{T}, fs::Number) where {T<:Real} = Signal(data, Float64(fs), (0:length(data) - 1) / Float64(fs))

# Input/Output --------------------------------------------------------------------------

function stereo2mono(data::AbstractArray{T}) where {T<:Number}
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

@recipe function plot_signal(s::Signal; axes=true, normalize=false)
    legend --> false
    xguide --> (axes ? "Time (s)" : "")
    yguide --> (axes ? "Amplitude" : "")
    if normalize; ylims --> (-1, 1) end
    return time(s), data(s)
end

"""
    plot(s::Signal; axes::Bool=true, normalize::Bool=false, kwargs...)

Plot recipe for Signal object.
If `axes` is true, the default axis labels are displayed.
If `normalize` is true, the signal amplitude is normalized in [-1,+1].
Both parameters can be overriden using the appropriate keyword arguments.
"""
plot