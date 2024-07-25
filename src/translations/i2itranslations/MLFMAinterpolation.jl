"""
    For each component:

    f(θ,φ) = f(θ,φ+2π),     for φ < 0
    f(θ,φ) = f(θ,φ-2π),     for φ > 2π

    f(θ,φ) = f(-θ,φ+pi),    for θ< 0, 0 < φ < π
    f(θ,φ) = f(-θ,φ-pi),    for θ < 0, π < φ < 2π
    f(θ,φ) = f(2π-θ,φ+pi),  for θ > π, 0 < φ < π
    f(θ,φ) = f(2π-θ,φ-pi)   for θ > π, π < φ < 2π
"""
struct HybridInterpolationTrait{D} <: MLFMA.InterpolationExpansionTrait end

function MLFMA.converttrait(
    ::HybridInterpolationTrait{4}, dictvalue, ::MLFMA.DefaultInterpolation
)
    return (dictvalue.virtual, dictvalue.virtual, dictvalue.virtual, dictvalue.virtual)
end

function MLFMA.converttrait(
    ::HybridInterpolationTrait{4}, dictvalue, ::MLFMA.SphericalInterpolation
)
    return SVector(dictvalue[1], dictvalue[1], dictvalue[1], dictvalue[1])
end

function MLFMA.converttrait(
    ::HybridInterpolationTrait{3}, dictvalue, ::MLFMA.DefaultInterpolation
)
    return (dictvalue.virtual, dictvalue.virtual, dictvalue.virtual)
end

function MLFMA.converttrait(
    ::HybridInterpolationTrait{3}, dictvalue, ::MLFMA.SphericalInterpolation
)
    return SVector(dictvalue[1], dictvalue[1], dictvalue[1])
end

function MLFMA.converttrait(
    ::HybridInterpolationTrait{1}, dictvalue, ::MLFMA.DefaultInterpolation
)
    return (dictvalue.virtual,)
end

function MLFMA.converttrait(
    ::HybridInterpolationTrait{1}, dictvalue, ::MLFMA.SphericalInterpolation
)
    return SVector(dictvalue[1])
end

function defaultinterpolationtrait(::F) where {D,F<:MLFMA.FarField{D}}
    return HybridInterpolationTrait{D}()
end

function MLFMA.interpolationtrait(::Union{iFMM.MWHyperSingular3D,iFMM.MWWeaklySingular3D})
    return MLFMA.SphericalInterpolation()  #TODO: change this to HybridInterpolationTrait
end
