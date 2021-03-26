module WCA1

using FFTW, OffsetArrays, Plots, ProgressMeter
using Distributions
using SparseArrays

import WAV, DSP, Statistics
using DSP.Windows, ImageFiltering
import FFTW: Frequencies, fftfreq, rfftfreq
import LinearAlgebra: norm

export Signal, STFT, Lift, fs, time, freq, data, width, slopes, duration, window,
    energy, power, hopsamples,
    stft, istft, lift, project, wc_delay, normalize, resync,
    norm, distance, relative_distance, reconstruct,
    wavread, wavwrite

include("sparse.jl")

include("signal.jl")
include("stft.jl")
include("lift.jl")
include("kernel.jl")
include("wc.jl")
include("util.jl")

end # module
