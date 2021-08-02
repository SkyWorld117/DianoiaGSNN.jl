using Polyester

function oneHot(input::Array{Int64}, depth::Int64, dict::Dict)
    output_data = zeros(Int8, depth, size(input, 1))
    @batch for i in 1:length(input)
        output_data[dict[input[i]],i] = 1
    end
    return output_data
end
