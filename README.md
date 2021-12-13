# DianoiaSNN

**DianoiaGSNN** is a framework for Spiking Neural Network. 

Please notice it is still in basic developing process and is not fully functioning. 

It has not been registered yet. You may want to try it like below:

```julia
include("DianoiaSNN.jl")
using .DianoiaSNN
using MLDatasets, LoopVectorization

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

dict = Dict{Int64, Int64}(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 6=>6, 7=>7, 8=>8, 9=>9, 0=>10)

nn = Network(;runtime=200, input_size=784)
nn.add_layer(nn; layer_size=10)

nn.train(nn; input_data=flatten(train_x,3), output_data=oneHot(train_y,10,dict), epochs=20, batch_size=500, 
         η=0.1, A₊=0.8, A₋=0.3, γᵥ=0.15, γₚ=1/1.1, γₘ=0.8, γₜ=0.34, w₊=1.5, w₋=-1.2, r=30, scales=3, guide=1, record=1)
```
