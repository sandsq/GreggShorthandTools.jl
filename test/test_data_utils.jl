using GreggShorthandTools
using GreggShorthandTools.Alphabet
using MLDatasets, Flux
using Test

args = Dict(
    :bsz => 64, # batch size
    :img_size => (100, 100), # mnist image size
    :n_epochs => 5, # no. epochs to train
)

@testset "dims" begin
    dev = cpu

    train_digits, train_labels = MNIST(split=:train)[:]
    test_digits, test_labels = MNIST(split=:test)[:]

    println("training features $(size(train_digits))")
    println("training labels $(size(train_labels))")
    println("test features $(size(test_digits))")
    println("test labels $(size(test_labels))")

    train_labels_onehot = Flux.onehotbatch(train_labels, 0:9)
    test_labels_onehot = Flux.onehotbatch(test_labels, 0:9)

    train_loader = Flux.DataLoader((train_digits |> dev, train_labels_onehot |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
    test_loader = Flux.DataLoader((test_digits |> dev, test_labels_onehot |> dev), batchsize=args[:bsz], shuffle=true, partial=false)


    letters_to_predict = [_K, _G]
    train_data, test_data = GreggShorthandTools.load_from_directory(0.8, letters_to_predict)

    train_digits, train_labels = train_data.features, train_data.targets
    test_digits, test_labels = test_data.features, test_data.targets

    println("training features $(size(train_digits))")
    println("training labels $(size(train_labels))")
    println("test features $(size(test_digits))")
    println("test labels $(size(test_labels))")

    train_labels_onehot = Flux.onehotbatch(train_labels, letters_to_predict)
    test_labels_onehot = Flux.onehotbatch(test_labels, letters_to_predict)

    train_loader = Flux.DataLoader((train_digits |> dev, train_labels_onehot |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
    test_loader = Flux.DataLoader((test_digits |> dev, test_labels_onehot |> dev), batchsize=args[:bsz], shuffle=true, partial=false)

end
