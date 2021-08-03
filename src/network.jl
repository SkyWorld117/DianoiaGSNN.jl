using LoopVectorization, Polyester

mutable struct Network
    input_size::Int64
    output_size::Int64
    hidden_neurons::Int64

    max_runtime::Int64

    weights::Array{Float32}
    u::Array{Float32}
    r::Array{Float32}

    voltage::Array{Float32}
    M::Array{Float32}
    output::Array{Float32}

    loss::Int64

    activate::Any
    train::Any
    ga_train::Any

    function Network(;input_size::Int64, output_size::Int64, hidden_neurons::Int64, max_runtime::Int64)
        s = input_size + output_size + hidden_neurons
        weights = rand(-sqrt(3/s):1.0e-5:sqrt(3/s), s,s)
        @avxt for i in 1:s
            weights[i,i] = 0.0f0
        end
        u = zeros(Float32, s,1)
        r = ones(Float32, s,1)

        voltage = zeros(Float32, s,1)
        M = zeros(Float32, s,1)
        output = zeros(Float32, s,max_runtime)
        new(input_size, output_size, hidden_neurons, max_runtime, weights, u, r, voltage, M, output, 0, activate, train, ga_train)
    end

    function activate(nn::Network, input_data::Array{Float32}, output_data::Array{Int8}, label::Int64, monitor::String; Cₘ, gₗ, Eₗ, U, τᵣ, τᵤ, f, θ, A₊, A₋)
        @avxt nn.u .= U
        @avxt nn.r .= 1.0f0
        @avxt nn.output .= 0.0f0
        @avxt nn.voltage .= 0.0f0
        @avxt nn.M .= 0.0f0
        
        @avxt for i in eachindex(input_data)
            nn.output[i,1] = input_data[i]
        end

        t = 0
        active = true
        while active && t<nn.max_runtime-1
            t += 1
            active = false

            @avxt for i in axes(nn.voltage, 1)
                s = 0.0f0
                for j in axes(nn.weights, 2)
                    s += nn.weights[i,j] * nn.output[j,t]
                    nn.weights[i,j] += ifelse(i!=j, min(A₋*nn.M[i]*nn.weights[i,j], 1.0f0), 0.0f0) * ifelse(nn.output[j,t]!=0, 1.0f0, 0.0f0)
                    nn.weights[i,j] = ifelse(nn.weights[i,j]>θ*Cₘ+gₗ*(θ-Eₗ), θ*Cₘ+gₗ*(θ-Eₗ), nn.weights[i,j])
                    nn.weights[i,j] = ifelse(nn.weights[i,j]<-θ*Cₘ-gₗ*(θ-Eₗ), -θ*Cₘ-gₗ*(θ-Eₗ), nn.weights[i,j])
                end
                nn.voltage[i,1] += 1/Cₘ * (s - gₗ*(nn.voltage[i,1] - Eₗ))
                nn.voltage[i,1] = ifelse(nn.voltage[i,1]>θ, θ, nn.voltage[i,1])
                nn.voltage[i,1] = ifelse(nn.voltage[i,1]<Eₗ, Eₗ, nn.voltage[i,1])
            end

            if t == label
                loss = 0
                #=@avxt for i in 1:length(output_data)
                    output = ifelse(nn.voltage[nn.input_size+nn.hidden_neurons+i, 1]>=θ, 1, 0)
                    loss += ifelse(output==output_data[i], 0, 1)
                    nn.voltage[nn.input_size+nn.hidden_neurons+i, 1] = (1-output_data[i])*Eₗ + output_data[i]*θ
                end=#
                for i in 1:length(output_data)
                    output = nn.voltage[nn.input_size+nn.hidden_neurons+i, 1]>=θ ? 1 : 0
                    loss += output==output_data[i] ? 0 : 1
                    nn.voltage[nn.input_size+nn.hidden_neurons+i, 1] = (1-output_data[i])*Eₗ + output_data[i]*θ
                end
                if monitor == "absolute"
                    nn.loss += loss
                elseif monitor == "examples"
                    nn.loss += loss>0 ? 1 : 0
                end
            end

            @avxt for i in eachindex(nn.voltage)
                nn.output[i,t+1] = ifelse(nn.voltage[i]>=θ, nn.u[i]*nn.r[i], 0.0f0)
                for j in axes(nn.weights, 2)
                    nn.weights[i,j] += ifelse(i!=j, max(A₊*(nn.voltage[i]-Eₗ)*nn.weights[i,j], 0.0f0), 0.0f0) * ifelse(nn.output[j,t]!=0, 1.0f0, 0.0f0)
                    nn.weights[i,j] = ifelse(nn.weights[i,j]>θ*Cₘ+gₗ*(θ-Eₗ), θ*Cₘ+gₗ*(θ-Eₗ), nn.weights[i,j])
                    nn.weights[i,j] = ifelse(nn.weights[i,j]<-θ*Cₘ-gₗ*(θ-Eₗ), -θ*Cₘ-gₗ*(θ-Eₗ), nn.weights[i,j])
                    #nn.weights[i,j] += ifelse(i!=j, (nn.regression-nn.weights[i,j])*ℯ^(-γ), 0.0f0)
                    #nn.weights[i,j] += ifelse(i!=j, -nn.weights[i,j]*ℯ^(-2.5), 0.0f0)
                end
                nn.M[i] -= 1/Cₘ * (nn.output[i,t+1] + gₗ*nn.M[i])
                nn.M[i] = ifelse(nn.M[i]>0.0f0, 0.0f0, nn.M[i])
                nn.M[i] = ifelse(nn.M[i]<-θ, -θ, nn.M[i])

                nn.r[i] = ifelse(nn.voltage[i]>=θ, nn.u[i]*nn.r[i], nn.r[i])
                nn.u[i] = ifelse(nn.voltage[i]>=θ, f*(1-nn.u[i]), nn.u[i])
                nn.r[i] += (1-nn.r[i])/τᵣ
                nn.u[i] += (U-nn.u[i])/τᵤ
                #nn.r[i] = ifelse(nn.r[i]>1, 1.0f0, nn.r[i])
                nn.u[i] = ifelse(nn.u[i]>U, U, nn.u[i])

                nn.voltage[i] = ifelse(nn.voltage[i]>=θ, Eₗ, nn.voltage[i])
            end

            for i in axes(nn.output, 1)
                if nn.output[i, t+1] != 0.0f0
                    active = true
                    break
                end
            end
            if !active && t<label
                if monitor == "absolute"
                    nn.loss += length(output_data)
                elseif monitor == "examples"
                    nn.loss += 1
                end
            end
        end
    end

    function train(nn::Network; input_data::Array{Float32}, output_data::Array{Int8}, epochs::Int64, label::Int64, batch::Real=32, monitor::String="absolute", parameters...)
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
                    nn.activate(nn, current_input_data, current_output_data, label, monitor; parameters...)
                end
                print("] with loss ", nn.loss, "\nTime usage: ")
                nn.loss = 0
            end
        end
    end

    function ga_train(nn::Network; input_data::Array{Float32}, output_data::Array{Int8}, epochs::Int64, label::Int64, batch::Real=32, monitor::String="absolute", parameters...)
        batch_size = ceil(Int64, size(input_data,2)/batch)
        current_input_data = zeros(Float32, size(input_data,1))
        current_output_data = zeros(Int8, size(output_data,1))

        lossₜ = 0
        for e in 1:epochs
            nn.loss = 0
            for t in 1:batch_size
                index = rand(1:size(input_data)[end])
                @avxt current_input_data .= input_data[:,index]
                @avxt current_output_data .= output_data[:,index]

                nn.activate(nn, current_input_data, current_output_data, label, monitor; parameters...)
            end
            lossₜ += nn.loss
        end

        return lossₜ*0.4/epochs + nn.loss*0.6
    end
end