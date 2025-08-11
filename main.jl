using Random
include("src/shorthand_detector.jl")
using shorthand_detector.GreggAlphabet


println(_A)

# shorthand_detector.run()

rng = Xoshiro(0)
# shorthand_detector.Drawer.draw_stroke("/home/sand/.julia/dev/shorthand_detector/data/g_path.png")

for i in 1:1000
    for letter in [_K, _G]

        path = "/home/sand/.julia/dev/shorthand_detector/data/$(to_string(letter))/$i.png"
        # println("path is $path")
        shorthand_detector.Drawer.draw_stroke(Random.default_rng(), letter, path)
    end
end
