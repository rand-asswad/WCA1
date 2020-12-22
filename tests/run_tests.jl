import Pkg
Pkg.activate("..")
using WCA1

using Test

include("signals.jl")

include("test_kernel.jl")
include("test_stft.jl")
include("test_chi2.jl")