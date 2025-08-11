module Drawer

export test_func,
    # draw_ellipse,
    draw_stroke

# include("../GreggAlphabet.jl")
using ..GreggAlphabet
using Luxor
using Match
import Random: AbstractRNG, default_rng

const BASE_DIR = joinpath(@__DIR__, "..", "..", "data")
const SQUARE_CANVAS_SIZE = 50
const CANVAS_HEIGHT = SQUARE_CANVAS_SIZE
const CANVAS_WIDTH = SQUARE_CANVAS_SIZE
const CENTER_X = CANVAS_WIDTH / 2
const CENTER_Y = CANVAS_HEIGHT / 2

# function draw_ellipse()
#     Drawing(CANVAS_HEIGHT, CANVAS_WIDTH, "$(BASE_DIR)/test.png")
#     text("Hello world", Point(5, 5))
#     circle(CENTER, 15, action = :stroke)
#     finish()
# end

"""
  draw_stroke()

Draw `k` or `g` stroke. Rotate for `r, l, p, b, f, v`
"""
function draw_stroke(rng::AbstractRNG, letter::Letter, path::String; rotation=0)
    # println("saving $(BASE_DIR)/k_path.png")
    println("saving to $path")

    Drawing(CANVAS_HEIGHT, CANVAS_WIDTH, path)
    origin(CENTER_X, CENTER_Y)
    background("white")

    function randomize_between(inner_rng::AbstractRNG, min_thickness::Number, max_thickness::Number)
        return min_thickness + rand(inner_rng, Float64) * (max_thickness - min_thickness)
    end
    setline(randomize_between(rng, 0.9, 1.5))

    # scale(0.25)

    (tiny_range, small_range, medium_range, large_range) =
        if letter == _K
            ((14, 20), (7, 9), (4.5, 6.5), (2, 3))
        elseif letter == _G
            ((14, 20), (7, 9), (2.1, 3), (2, 3))
        else
            error("$letter is not a valid Gregg letter")
        end

    println("med range $medium_range")
    exit()
    tiny_offset_base() = randomize_between(rng, tiny_range...)
    small_offset_base() = randomize_between(rng, small_range...)
    # medium_offset_base() = randomize_between(rng, 4.5, 6.5)
    medium_offset_base() = randomize_between(rng, medium_range...)
    large_offset_base() = randomize_between(rng, large_range...)

    tiny_offset_x() = CANVAS_WIDTH / tiny_offset_base()
    tiny_offset_y() = CANVAS_HEIGHT / tiny_offset_base()
    small_offset_x() = CANVAS_WIDTH / small_offset_base()
    small_offset_y() = CANVAS_HEIGHT / small_offset_base()
    medium_offset_x() = CANVAS_WIDTH / medium_offset_base()
    medium_offset_y() = CANVAS_HEIGHT / medium_offset_base()
    large_offset_x() = CANVAS_WIDTH / large_offset_base()
    large_offset_y() = CANVAS_HEIGHT / large_offset_base()

    # sethue(0.1, 0.6, 0.8)
    # line(Point(CANVAS_WIDTH / 2, 0), Point(CANVAS_WIDTH / 2, CANVAS_HEIGHT))
    # line(Point(0, CANVAS_HEIGHT / 2), Point(CANVAS_WIDTH, CANVAS_HEIGHT / 2))
    # strokepath()

    rotate(rotation)

    sethue("black")
    p1 = Point(-medium_offset_x(), tiny_offset_y())
    p2 = Point(medium_offset_x(), -small_offset_y())

    should_hook_back = true
    if should_hook_back
        p3_randomized_offset = medium_offset_x()
        p3 = Point(p3_randomized_offset, 0)
        curve(p1, p2, p3)
        strokepath()
        setline(randomize_between(rng, 1.5, 2.5))

        p4_randomized_offset = tiny_offset_y()
        p4 = Point(p3_randomized_offset, 0.5 * p4_randomized_offset)
        p5 = Point(p3_randomized_offset * 0.9, p4_randomized_offset)
        curve(p3, p4, p5)
    else
        p3 = Point(medium_offset_x(), tiny_offset_y())
        curve(p1, p2, p3)
    end

    strokepath()

    finish()
end

function draw_stroke(args...; kwargs...)
    draw_stroke(default_rng(), args...; kwargs...)
end


function test_func()
    println("test function")
    return "hi"
end

end
