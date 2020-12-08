import Pkg
Pkg.activate("..")
using WCA1

using Test

include("test_kernel.jl")
include("test_stft.jl")