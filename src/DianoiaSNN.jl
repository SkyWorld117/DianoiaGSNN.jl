module DianoiaSNN
    using LinearAlgebra
    BLAS.set_num_threads(1)
    
    include("./network.jl")
    include("./flatten.jl")
    include("./oneHot.jl")

    export Network, oneHot, flatten
end