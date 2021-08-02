include("./DianoiaGSNN.jl")
using .DianoiaGSNN
using MLDatasets

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

dict = Dict{Int64, Int64}(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 6=>6, 7=>7, 8=>8, 9=>9, 0=>10)

G = GA(;pool_size=50, run_time=200, pass=5, α=0.3, 
        input_size=784, output_size=10, hidden_neurons=206, max_runtime=4, label=2, epochs=15, batch=128, monitor="absolute", 
        input_data=flatten(train_x,3), output_data=oneHot(train_y,10,dict))

#G.reload(G, "./gen_temp.h5")

G.construct(G)