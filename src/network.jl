using LoopVectorization, Polyester

mutable struct Network
    input_size::Int64
    output_size::Int64
    hidden_neurons::Int64

    threshold::Float64
    droprate::Float64
    max_runtime::Int64
    regression::Float32

    weights::Array{Float32}

    value::Array{Float32}
    output::Array{Int8}

    loss::Int64

    activate::Any
    train::Any
    ga_train::Any

    function Network(;input_size::Int64, output_size::Int64, hidden_neurons::Int64, threshold::Float64, droprate::Float64, regression::Float32=1.0f0/(input_size+output_size+hidden_neurons), max_runtime::Int64)
        s = input_size + output_size + hidden_neurons
        weights = fill(regression, s,s)
        @avxt for i in 1:s
            weights[i,i] = 0.0f0
        end
        value = zeros(Float32, s,1)
        output = zeros(Int8, s,max_runtime)
        new(input_size, output_size, hidden_neurons, threshold, droprate, max_runtime, regression, weights, value, output, 0, activate, train, ga_train)
    end

    function activate(nn::Network, input_data::Array{Float32}, output_data::Array{Int8}, α::Float64, γ::Float64, θ, label::Int64, monitor::String)
        @avxt nn.output .= 0
        @avxt nn.value .= 0.0f0
        @avxt for i in eachindex(input_data)
            nn.value[i] = input_data[i]
        end

        t = 0
        active = true
        w⁺ = nn.threshold/20 * (10 - log(1/θ-1))
        while active && t<nn.max_runtime-1
            t += 1
            active = false

            @avxt for i in axes(nn.value, 1)
                s = 0.0f0
                for j in axes(nn.weights, 2)
                    s += nn.weights[i,j] * nn.output[j,t]
                end
                nn.value[i,1] += s
                nn.value[i,1] = ifelse(nn.value[i,1]>3.0f38, 3.0f38, nn.value[i,1])
            end

            if t == label
                loss = 0
                @avxt for i in 1:length(output_data)
                    output = ifelse(nn.value[nn.input_size+nn.hidden_neurons+i]>=nn.threshold, 1, 0)
                    loss += ifelse(output==output_data[i], 0, 1)
                    nn.value[nn.input_size+nn.hidden_neurons+i] += nn.threshold*output_data[i]
                end
                if monitor == "absolute"
                    nn.loss += loss
                elseif monitor == "examples"
                    nn.loss += loss>0 ? 1 : 0
                end
            end

            @avxt for i in axes(nn.weights, 1), j in axes(nn.weights, 2)
                #nn.weights[i,j] += ifelse(i!=j, ifelse(nn.value[j]>θ, α*nn.value[i]*nn.output[j,t], -α*nn.value[i]*nn.output[j,t]), 0.0f0)
                nn.weights[i,j] += ifelse(i!=j, ifelse(1/(1+ℯ^(-20*nn.value[j]/nn.threshold+10))>θ, α*nn.value[i]*nn.output[j,t], -α*nn.value[i]*nn.output[j,t]), 0.0f0)
                #nn.weights[i,j] = ifelse(nn.weights[i,j]>nn.threshold, nn.threshold, nn.weights[i,j])
                nn.weights[i,j] = ifelse(nn.weights[i,j]>w⁺, w⁺, nn.weights[i,j])
                nn.weights[i,j] += ifelse(i!=j, (nn.regression-nn.weights[i,j])*ℯ^(-γ), 0.0f0)
                nn.weights[i,j] = ifelse(nn.weights[i,j]<0.0f0, 0.0f0, nn.weights[i,j])
            end

            @avxt for i in eachindex(nn.value)
                nn.output[i,t+1] = ifelse(nn.value[i]>nn.threshold, 1, 0)
                nn.value[i] = ifelse(nn.value[i]>=nn.threshold, 0.0f0, nn.value[i]*(1-nn.droprate))
            end

            for i in axes(nn.output, 1)
                if nn.output[i, t+1] == 1
                    active = true
                    break
                end
            end
            if !active && t<label
                nn.loss += length(output_data)
            end
        end
    end

    function train(nn::Network; input_data::Array{Float32}, output_data::Array{Int8}, epochs::Int64, α::Float64, γ::Float64, θ::Float64, label::Int64, batch::Real=32, monitor::String="absolute")
        batch_size = ceil(Int64, size(input_data,2)/batch)
        current_input_data = zeros(Float32, size(input_data,1))
        current_output_data = zeros(Int8, size(output_data,1))

        for e in 1:epochs
            print("Epoch ", e, "\n[")
            @time begin
                for t in 1:batch_size
                    index = rand(1:size(input_data)[end])
                    @avxt current_input_data .= input_data[:,index]
                    @avxt current_output_data .= output_data[:,index]

                    if t%ceil(batch_size/50)==0
                        print("=")
                    end
                    nn.activate(nn, current_input_data, current_output_data, α, γ, θ, label, monitor)
                end
                print("] with loss ", nn.loss, "\nTime usage: ")
                nn.loss = 0
            end
        end
    end

    function ga_train(nn::Network; input_data::Array{Float32}, output_data::Array{Int8}, epochs::Int64, α::Float64, γ::Float64, θ::Float64, label::Int64, batch::Real=32, monitor::String="absolute")
        batch_size = ceil(Int64, size(input_data,2)/batch)
        current_input_data = zeros(Float32, size(input_data,1))
        current_output_data = zeros(Int8, size(output_data,1))

        for e in 1:epochs
            for t in 1:batch_size
                index = rand(1:size(input_data)[end])
                @avxt current_input_data .= input_data[:,index]
                @avxt current_output_data .= output_data[:,index]

                nn.activate(nn, current_input_data, current_output_data, α, γ, θ, label, monitor)
            end
        end
    end
end