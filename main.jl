using Random
include("src/GreggShorthandTools.jl")
using .GreggShorthandTools
using .GreggShorthandTools.Alphabet


println(_A)

# GreggShorthandTools.run()

rng = Xoshiro(0)
GreggShorthandTools.Drawer.draw_stroke(_G, "/home/sand/.julia/dev/GreggShorthandTools/data/g_path.png")
GreggShorthandTools.Drawer.draw_stroke(_R, "/home/sand/.julia/dev/GreggShorthandTools/data/r_path.png")


# for letter in [_K, _G, _R, _L]
#     println("letter is $letter")
#     for i in 1:1000
#         path = "/home/sand/.julia/dev/GreggShorthandTools/data/$(to_string(letter))/$i.png"
#         mkpath(dirname(path))
#         # println("path is $path")
#         GreggShorthandTools.Drawer.draw_stroke(Random.default_rng(), letter, path)
#     end
# end
