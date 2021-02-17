module WCA1

using FFTW, OffsetArrays, Plots, ProgressMeter
using Distributions

import WAV, DSP, Statistics
using DSP.Windows, ImageFiltering
import FFTW: Frequencies, fftfreq, rfftfreq

export Signal, STFT, Lift, time, freq, data, width, slopes,
    stft, istft, lift, project, wc_delay, normalize,
    wavread, wavwrite, plot, plot!


include("signal.jl")
include("stft.jl")
include("lift.jl")
include("kernel.jl")
include("wc.jl")

#include("util.jl)
#include("filters.jl")

end # module
