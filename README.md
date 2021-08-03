# DianoiaGSNN

**DianoiaGSNN** is a framework for Graph Spiking Neural Network. 

It has not been registered yet. You may want to try it like below:

```julia
include("./DianoiaGSNN.jl")
using .DianoiaGSNN
using MLDatasets

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

dict = Dict{Int64, Int64}(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 6=>6, 7=>7, 8=>8, 9=>9, 0=>10)

nn = Network(;input_size=784, output_size=10, hidden_neurons=206, max_runtime=5)
nn.train(nn; input_data=flatten(train_x,3), output_data=oneHot(train_y,10,dict), 
         epochs=100, label=2, batch=128, monitor="examples", 
         Cₘ=1.85, gₗ=1.0, Eₗ=0.0, U=1.0, τᵣ=3.0, τᵤ=3.0, f=1.0, θ=0.4, A₊=0.001, A₋=0.002)
```
