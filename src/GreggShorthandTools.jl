module GreggShorthandTools

# using DrWatson
# @quickactivate "spatial_transformer"

# using AMDGPU
# AMDGPU.allowscalar(false)
using MLDatasets, Flux, JLD2, MLUtils  # this will install everything if necc.
using Zygote
using Flux: batch, onehotbatch, unsqueeze
using Flux: DataLoader
using Statistics: mean  # standard library
using ImageCore, ImageInTerminal, Images
using FileIO
using LinearAlgebra, Statistics
using Base.Iterators: partition
using ProgressMeter
using ProgressMeter: Progress
using ProgressBars


include("Alphabet.jl")
using .Alphabet
include("Drawer/Drawer.jl")
using .Drawer

include("data_utils.jl")



export run_conv, run_spatial_transformer



const IMAGE_SIZE_X = 50
const IMAGE_SIZE_Y = 50

include("models/spatial_transformer.jl")
include("models/conv.jl")

end
