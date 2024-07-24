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
    parentmoment, childmoment, tree, parent::Int, child::Int
)
    for i in eachindex(childmoment)
        conj!(childmoment[i])
    end

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

        x = _elementwise_mul_sum(childmomentx, convertmatrix)
        y = _elementwise_mul_sum(childmomenty, convertmatrix)
        z = _elementwise_mul_sum(childmomentz, convertmatrix)

        u = Jus[i][1] * x + Jus[i][2] * y + Jus[i][3] * z
        v = Jvs[i][1] * x + Jvs[i][2] * y + Jvs[i][3] * z

        parentmoment[1][i] = u
        parentmoment[2][i] = v
    end

    for i in eachindex(parentmoment)
        conj!(parentmoment[i])
    end

    for i in eachindex(childmoment)
        conj!(childmoment[i])
    end

    return parentmoment
end

function _elementwise_mul_sum(a, b)
    result = zero(eltype(a))
    for i in eachindex(a)
        result += a[i] * b[i]
    end
    return result
end
