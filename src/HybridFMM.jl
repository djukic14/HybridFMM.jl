module HybridFMM

using LinearAlgebra
using SumTypes
using StaticArrays
using Dictionaries

using NURBS
using H2Factory
using BEAST
using MLFMA
using iFMM

const iFMMNURBS = Base.get_extension(iFMM, :iFMMNURBS)

include("HybridFMM/utilsmaps.jl")
include("HybridFMM/GalerkinHybridFMM.jl")
include("moments/hybridmoment.jl")
include("translations/i2otranslationcollection.jl")
include("translations/i2itranslations/hybridi2itranslation.jl")
include("translations/i2itranslations/MLFMAinterpolation.jl")
include("translations/i2itranslations/i2itranslations.jl")
include("translations/o2otranslations/hybrido2otranslation.jl")
include("translations/o2otranslations/o2otranslations.jl")

export HybridMoment

end
