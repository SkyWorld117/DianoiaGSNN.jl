module DianoiaRNNS
    include("./network.jl")
    include("./flatten.jl")
    include("./oneHot.jl")

    export Network, oneHot, flatten
end