struct HybridO2OTranslator{C,U,V}
    convertmatrices::C
    Jus::U
    Jvs::V
end

function HybridO2OTranslator(::Nothing)
    return nothing
end

function HybridO2OTranslator(hybridi2i)
    return HybridO2OTranslator(hybridi2i.convertmatrices, hybridi2i.Jus, hybridi2i.Jvs)
end

function (hybridtranslator::HybridO2OTranslator)(
    parentmoment::P, childmoment::F, tree, parent::Int, child::Int
) where {P<:H2Factory.AbstractMoment{3},F<:H2Factory.AbstractMoment{4}}
    for i in eachindex(parentmoment)
        parentmoment[i] .= zero(eltype(parentmoment[i]))
    end

    LinearAlgebra.mul!(
        reshape(parentmoment[3], :),
        transpose(hybridtranslator.convertmatrices[child]),
        reshape(childmoment[4], :),
    )

    Jus = hybridtranslator.Jus[child]
    Jvs = hybridtranslator.Jvs[child]

    convertmatrices = [
        view(hybridtranslator.convertmatrices[child], :, i) for i in eachindex(Jus)
    ]

    for i in eachindex(Jus)
        convertmatrix = convertmatrices[i]
        childmomentx = view(childmoment[1], :)
        childmomenty = view(childmoment[2], :)
        childmomentz = view(childmoment[3], :)

        x = _elementwise_mul_sum_conj(childmomentx, convertmatrix)
        y = _elementwise_mul_sum_conj(childmomenty, convertmatrix)
        z = _elementwise_mul_sum_conj(childmomentz, convertmatrix)

        u = Jus[i][1] * x + Jus[i][2] * y + Jus[i][3] * z
        v = Jvs[i][1] * x + Jvs[i][2] * y + Jvs[i][3] * z

        parentmoment[1][i] = conj(u)
        parentmoment[2][i] = conj(v)
    end

    return parentmoment
end

function (hybridtranslator::HybridO2OTranslator)(
    parentmoment::P, childmoment::F, tree, parent::Int, child::Int
) where {P<:H2Factory.AbstractMoment{1},F<:H2Factory.AbstractMoment{1}}
    LinearAlgebra.mul!(
        reshape(parentmoment[1], :),
        adjoint(hybridtranslator.convertmatrices[child]),
        reshape(childmoment[1], :),
    )
    return parentmoment
end

function (hybridtranslator::HybridO2OTranslator)(
    parentmoment::P, childmoment::F, tree, parent::Int, child::Int
) where {P<:H2Factory.AbstractMoment{2},F<:H2Factory.AbstractMoment{3}}
    for i in eachindex(parentmoment)
        parentmoment[i] .= zero(eltype(parentmoment[i]))
    end

    Jus = hybridtranslator.Jus[child]
    Jvs = hybridtranslator.Jvs[child]

    convertmatrices = [
        view(hybridtranslator.convertmatrices[child], :, i) for i in eachindex(Jus)
    ]

    for i in eachindex(Jus)
        convertmatrix = convertmatrices[i]
        childmomentx = view(childmoment[1], :)
        childmomenty = view(childmoment[2], :)
        childmomentz = view(childmoment[3], :)

        x = _elementwise_mul_sum_conj(childmomentx, convertmatrix)
        y = _elementwise_mul_sum_conj(childmomenty, convertmatrix)
        z = _elementwise_mul_sum_conj(childmomentz, convertmatrix)

        u = Jus[i][1] * x + Jus[i][2] * y + Jus[i][3] * z
        v = Jvs[i][1] * x + Jvs[i][2] * y + Jvs[i][3] * z

        parentmoment[1][i] = conj(u)
        parentmoment[2][i] = conj(v)
    end

    return parentmoment
end

function _elementwise_mul_sum_conj(a, b)
    result = zero(eltype(a))
    for i in eachindex(a)
        result += conj(a[i]) * b[i]
    end
    return result
end
