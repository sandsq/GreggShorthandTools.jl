struct Data
    features
    targets
    split
end
function Base.length(d::Data)
    return length(d.targets)
end

function ready_picture!(picture_path::String, label::Letter, all_features::Vector{Array{Float32,2}}, all_labels::Vector{Letter})
    img = FileIO.load(picture_path)
    img_array = Gray.(img)

    push!(all_features, img_array)
    push!(all_labels, label)
end

function load_from_directory(proportion::Float64, letters_to_predict::Vector{Letter})
    base_dir = joinpath(@__DIR__, "..", "data")
    # "data"

    # hard coded image size for now
    training_features = Vector{Array{Float32,2}}()
    training_labels = Vector{Letter}()

    testing_features = Vector{Array{Float32,2}}()
    testing_labels = Vector{Letter}()

    for subdir_enum in letters_to_predict
        subdir = to_string(subdir_enum)
        full_dir = "$(base_dir)/$(subdir)"
        num_instances = length(readdir(full_dir))
        train_range = 1:floor(Int, num_instances * proportion)
        test_range = ceil(Int, num_instances * proportion):num_instances
        println("train range $(train_range) and test range $(test_range)")
        for picture_number in train_range
            # println("picture number $picture_number")
            ready_picture!("$(full_dir)/$(picture_number).png", subdir_enum, training_features, training_labels)
        end
        for picture_number in test_range
            # println("picture number $picture_number")
            ready_picture!("$(full_dir)/$(picture_number).png", subdir_enum, testing_features, testing_labels)
        end
    end
    training_features = reshape(reduce(hcat, training_features), IMAGE_SIZE_X, IMAGE_SIZE_Y, :)
    testing_features = reshape(reduce(hcat, testing_features), IMAGE_SIZE_X, IMAGE_SIZE_Y, :)
    # training_data = Dict("features" => training_features, "targets" => training_labels)
    # testing_data = Dict("features" => testing_features, "targets" => testing_labels)
    return Data(training_features, training_labels, :train), Data(testing_features, testing_labels, :test)
end
