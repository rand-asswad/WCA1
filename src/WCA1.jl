module WCA1

using FFTW, OffsetArrays, Plots, ProgressMeter
using Distributions

import WAV, DSP, Statistics
using DSP.Windows, ImageFiltering
import FFTW: Frequencies, fftfreq, rfftfreq
import LinearAlgebra: norm

export Signal, STFT, Lift, fs, time, freq, data, width, slopes, duration, window,
    energy, power, hopsamples,
    stft, istft, lift, project, wc_delay, normalize,
    wavread, wavwrite


include("signal.jl")
include("stft.jl")
include("lift.jl")
include("kernel.jl")
include("wc.jl")
include("util.jl")

end # module
