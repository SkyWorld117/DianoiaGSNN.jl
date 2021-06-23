using Polyester

function flatten(dataset::AbstractArray, batch_dim::Int64)
    dims = ndims(dataset)
    len = 1
    for i in 1:dims
        if i!=batch_dim
            len *= size(dataset, i)
        end
    end
    input_data = zeros(Float32, len, size(dataset, batch_dim))
    @batch for i in axes(dataset, batch_dim)
        input_data[:,i] = Array{Float32}(reshape(selectdim(dataset, batch_dim, i), (len,1)))
    end
    return input_data
end