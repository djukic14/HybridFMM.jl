"""
    For each of the 4 components:

    f(θ,φ) = f(θ,φ+2π),     for φ < 0
    f(θ,φ) = f(θ,φ-2π),     for φ > 2π

    f(θ,φ) = f(-θ,φ+pi),    for θ< 0, 0 < φ < π
    f(θ,φ) = f(-θ,φ-pi),    for θ < 0, π < φ < 2π
    f(θ,φ) = f(2π-θ,φ+pi),  for θ > π, 0 < φ < π
    f(θ,φ) = f(2π-θ,φ-pi)   for θ > π, π < φ < 2π
"""
struct Hybrid4InterpolationTrait <: MLFMA.InterpolationExpansionTrait end

function MLFMA.converttrait(
    ::Hybrid4InterpolationTrait, dictvalue, ::MLFMA.DefaultInterpolation
)
    return (dictvalue.virtual, dictvalue.virtual, dictvalue.virtual, dictvalue.virtual)
end

function MLFMA.converttrait(
    ::Hybrid4InterpolationTrait, dictvalue, ::MLFMA.SphericalInterpolation
)
    return SVector(dictvalue[1], dictvalue[1], dictvalue[1], dictvalue[1])
end
