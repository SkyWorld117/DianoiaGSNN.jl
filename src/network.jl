include("./layer.jl")
include("./spiking.jl")
using LoopVectorization, Polyester

mutable struct Network
    layers::Array

    runtime::Int64
    spiking_trains::Array{Float32}

    default_input_size::Int64

    add_layer
    activate
    train

    function Network(;runtime::Int64, input_size::Int64)
        spiking_trains = zeros(Float32, runtime, input_size)
        new(Any[], runtime, spiking_trains, input_size, add_layer, activate, train)
    end

    function add_layer(self::Network; args...)
        push!(self.layers, Layer(;input_size=self.default_input_size, args...))
        self.default_input_size = self.layers[end].layer_size
    end

    function activate(self::Network, input_data::Array{Float32}, output_data::Array{Float32}, table::Array{Int64}; η, A₊, A₋, γᵥ, γₚ, γₘ, γₜ, w₊, w₋, r, scales, guide, record)
        for layer in self.layers
            layer.reset(layer, r)
        end
        @avxt self.spiking_trains .= 0.0f0

        θ = Spiking!(input_data, self.spiking_trains, scales, r, self.runtime)*γₜ
        γᵥ = 1-γᵥ/θ

        self.layers[guide].guide = true
        self.layers[guide].sample_output = output_data
        self.layers[record].record = true

        first_spike = false
        for t in 1:self.runtime
            first_spike = self.layers[1].activate(self.layers[1], self.spiking_trains[t,:], η, A₊, A₋, γᵥ, γₚ, γₘ, w₊, w₋, θ, t, r, first_spike, table)
            for l in 2:length(self.layers)
                first_spike = self.layers[l].activate(self.layers[l], self.layers[l-1].output, η, A₊, A₋, γᵥ, γₚ, γₘ, w₊, w₋, θ, t, r, first_spike, table)
            end
        end
    end

    function train(self::Network; input_data::Array{Float32}, output_data::Array{Float32}, epochs::Int64=20, batch_size::Int64=500, parameters...)
        current_input_data = zeros(Float32, size(input_data,1))
        current_output_data = zeros(Float32, size(output_data,1))
        table = zeros(Int64, self.layers[end].layer_size,self.layers[end].layer_size)
        accuracy = 0.0

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
                    self.activate(self, current_input_data, current_output_data, table; parameters...)
                end
                @avxt for i in 1:10
                    accuracy += table[i,i]
                end
                accuracy = accuracy/batch_size*100
                print("] with accuracy $accuracy%\nTime usage: ")
                accuracy = 0
            end
            display(table)
            println("\n")
            @avxt table .= 0
        end
    end
end