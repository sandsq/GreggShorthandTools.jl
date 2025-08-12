module Drawer

using Luxor: Random
export test_func,
    # draw_ellipse,
    draw_stroke,
    draw_line

# include("../GreggAlphabet.jl")
using ..GreggShorthandTools.Alphabet
using Luxor
import Bezier: bezier as bz
using Match
import Random: AbstractRNG, default_rng

const BASE_DIR = joinpath(@__DIR__, "..", "..", "data")
const SQUARE_CANVAS_SIZE = 100
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

# multiply elementwise
function Base.:*(p1::Point, p2::Point)
    return Point(p1.x * p2.x, p1.y * p2.y)
end


function randomize_between(inner_rng::AbstractRNG, min_val::Number, max_val::Number)
    return min_val + rand(inner_rng, Float64) * (max_val - min_val)
end

function draw_stroke_bezier(rng::AbstractRNG, letter::Letter, path::String)
    println("saving to $path")

    Drawing(CANVAS_HEIGHT, CANVAS_WIDTH, path)
    origin(CENTER_X, CENTER_Y)

    # @@@@@@@@@@@@@@ background
    bg_grey = randomize_between(rng, 0.7, 1.0)
    background(bg_grey, bg_grey, bg_grey)

    D = 100
    mat = [Luxor.ARGB32(
        noise(0.01r, 0.01c),
        noise(0.01r, 0.01c),
        noise(0.01r, 0.01c),
        noise(0.01r, 0.01c)) for r in 1:D, c in 1:D]
    placeimage(mat, boxtopleft())

    setline(randomize_between(rng, 1, 2))

    # # @@@@@@@@@@@@@@ axes
    # sethue(0.1, 0.6, 0.8)
    # line(Point(-CANVAS_WIDTH, 0), Point(CANVAS_WIDTH, 0))
    # line(Point(0, CANVAS_HEIGHT), Point(0, -CANVAS_HEIGHT))
    # strokepath()

    # @@@@@@@@@@@@@@ stroke
    (rotation, transformation) =
        if letter == _K || letter == _G
            (0, [1 0 0 1 0 0])
        elseif letter == _L || letter == _R
            (pi, [1 0 0 1 0 0])
        elseif letter == _F || letter == _V
            (-pi / 2.5, [1 0 0 -1 0 0])
        elseif letter == _P || letter == _B
            (pi / 1.75, [1 0 0 -1 0 0])
        else
            error("$letter is not a valid Gregg letter for stroke")
        end

    rotate(rotation)# * randomize_between(rng, 0.9, 1.1))
    translation_amount_x = CANVAS_WIDTH / 5
    translation_amount_y = CANVAS_HEIGHT / 5
    translation_noise_x = randomize_between(rng, -translation_amount_x, translation_amount_x)
    translation_noise_y = randomize_between(rng, -translation_amount_y, translation_amount_y)
    translate(translation_noise_x, translation_noise_y)
    transform(transformation)

    sethue("black")

    # these values are manually tuned
    (scaling_offset_x, scaling_offset_y) =
        if letter == _K || letter == _R || letter == _P || letter == _F
            (6, 6)
        elseif letter == _G || letter == _L || letter == _B || letter == _V
            (3, 6)
        else
            error("$letter is not a valid Gregg letter for stroke")
        end
    base_xs = [
        -CANVAS_WIDTH / scaling_offset_x, # left point
        -CANVAS_WIDTH / scaling_offset_x / 4, # slight rising
        CANVAS_WIDTH / scaling_offset_x / 2, # main curve
        CANVAS_WIDTH / scaling_offset_x, # right point
        0.9 * CANVAS_WIDTH / scaling_offset_x,
        0.8 * CANVAS_WIDTH / scaling_offset_x
    ] # hook end
    base_ys = [
        0.2 * CANVAS_HEIGHT / scaling_offset_y,
        -CANVAS_HEIGHT / scaling_offset_y / 4,
        -CANVAS_HEIGHT / scaling_offset_y,
        -0.1 * CANVAS_HEIGHT / scaling_offset_y,
        0.1 * CANVAS_HEIGHT / scaling_offset_y,
        0.2 * CANVAS_HEIGHT / scaling_offset_y
    ]

    base_jitter_x = CANVAS_WIDTH / 15
    base_jitter_y = CANVAS_HEIGHT / 15

    base_small_jitter_x = CANVAS_WIDTH / 25
    base_small_jitter_y = CANVAS_HEIGHT / 25

    jitter_x() = randomize_between(rng, -base_jitter_x, base_jitter_x)
    jitter_y() = randomize_between(rng, -base_jitter_y, base_jitter_y)
    jitter(x, y) = randomize_between(rng, x, y)
    hook_jitter_x = jitter(-base_small_jitter_x, base_small_jitter_x)
    jitter_xs = [
        jitter(-base_small_jitter_x, base_small_jitter_x),
        0,
        jitter_x(),
        jitter(-base_small_jitter_x, base_small_jitter_x),
        hook_jitter_x,
        hook_jitter_x
    ]
    hook_jitter_y = jitter(-base_small_jitter_y, base_small_jitter_y)
    jitter_ys = [
        jitter(-base_small_jitter_y, base_small_jitter_y),
        0,
        jitter_y(),
        jitter(-base_small_jitter_y, base_small_jitter_y),
        hook_jitter_y,
        hook_jitter_y
    ]

    (xs, ys) = bz(base_xs + jitter_xs, base_ys + jitter_ys)
    for i in 1:length(xs)-1
        line(Point(xs[i], ys[i]), Point(xs[i+1], ys[i+1]))
    end
    strokepath()

    finish()

end

function draw_stroke_bezier(args...; kwargs...)
    draw_stroke_bezier(default_rng(), args...; kwargs...)
end

"""
  draw_stroke()

Draw `k` or `g` stroke. Rotate for `r, l, p, b, f, v`
"""
function draw_stroke(rng::AbstractRNG, letter::Letter, path::String)
    # println("saving $(BASE_DIR)/k_path.png")
    println("saving to $path")

    Drawing(CANVAS_HEIGHT, CANVAS_WIDTH, path)
    origin(CENTER_X, CENTER_Y)

    bg_grey = randomize_between(rng, 0.7, 1.0)
    background(bg_grey, bg_grey, bg_grey)

    D = 100
    mat = [Luxor.ARGB32(
        noise(0.01r, 0.01c),
        noise(0.01r, 0.01c),
        noise(0.01r, 0.01c),
        noise(0.01r, 0.01c)) for r in 1:D, c in 1:D]
    placeimage(mat, boxtopleft())

    setline(randomize_between(rng, 1, 2))

    # scale(0.25)

    # these values are manually tuned
    (scaling_offset_x, scaling_offset_y) =
        if letter == _K || letter == _R || letter == _P || letter == _F
            (6, 6)
        elseif letter == _G || letter == _L || letter == _B || letter == _V
            (3, 4)
        else
            error("$letter is not a valid Gregg letter for stroke")
        end
    jitter_x() = randomize_between(rng, 0.75, 1.25)
    jitter_y() = randomize_between(rng, 0.75, 1.25)
    jitter_x(x, y) = randomize_between(rng, x, y)
    jitter_y(x, y) = randomize_between(rng, x, y)


    base_left_bezier_point = Point(-CANVAS_WIDTH / scaling_offset_x, 0.3 * CANVAS_HEIGHT / scaling_offset_y) * (jitter_x(), jitter_y())
    base_middle_bezier_point = Point(CANVAS_WIDTH / scaling_offset_x, -CANVAS_HEIGHT / scaling_offset_y) * (jitter_x(0.5, 1.1), jitter_y())
    base_right_bezier_point = Point(CANVAS_WIDTH / scaling_offset_x, 0.3 * CANVAS_HEIGHT / scaling_offset_y) * (jitter_x(), jitter_y())

    base_middle_bezier_point_hook_variant = Point(0.5 * CANVAS_WIDTH / scaling_offset_x, -0.5 * CANVAS_HEIGHT / scaling_offset_y) * (jitter_x(0.5, 1.1), jitter_y(0.9, 1.5))
    # the hook randomization needs to be the same to ensure the concavity doesn't break
    hook_jitter = (jitter_x(), jitter_y(0.9, 1.25))
    base_right_bezier_point_hook_start = Point(0.9 * CANVAS_WIDTH / scaling_offset_x, -0.2 * CANVAS_HEIGHT / scaling_offset_y) * hook_jitter
    base_right_bezier_point_hook_middle = Point(CANVAS_WIDTH / scaling_offset_x, 0.1 * CANVAS_HEIGHT / scaling_offset_y) * hook_jitter
    base_right_bezier_point_hook_end = Point(0.85 * CANVAS_WIDTH / scaling_offset_x, 0.3 * CANVAS_HEIGHT / scaling_offset_y) * hook_jitter


    (rotation, transformation) =
        if letter == _K || letter == _G
            (0, [1 0 0 1 0 0])
        elseif letter == _L || letter == _R
            (pi, [1 0 0 1 0 0])
        elseif letter == _F || letter == _V
            (-pi / 2, [1 0 0 -1 0 0])
        elseif letter == _P || letter == _B
            (pi / 2, [1 0 0 -1 0 0])
        else
            error("$letter is not a valid Gregg letter for stroke")
        end

    # tiny_offset_base() = randomize_between(rng, tiny_range...)
    # small_offset_base() = randomize_between(rng, small_range...)
    # # medium_offset_base() = randomize_between(rng, 4.5, 6.5)
    # medium_offset_base() = randomize_between(rng, medium_range...)
    # large_offset_base() = randomize_between(rng, large_range...)

    # tiny_offset_x() = CANVAS_WIDTH / tiny_offset_base()
    # tiny_offset_y() = CANVAS_HEIGHT / tiny_offset_base()
    # small_offset_x() = CANVAS_WIDTH / small_offset_base()
    # small_offset_y() = CANVAS_HEIGHT / small_offset_base()
    # medium_offset_x() = CANVAS_WIDTH / medium_offset_base()
    # medium_offset_y() = CANVAS_HEIGHT / medium_offset_base()
    # large_offset_x() = CANVAS_WIDTH / large_offset_base()
    # large_offset_y() = CANVAS_HEIGHT / large_offset_base()

    # sethue(0.1, 0.6, 0.8)
    # line(Point(-CANVAS_WIDTH, 0), Point(CANVAS_WIDTH, 0))
    # line(Point(0, CANVAS_HEIGHT), Point(0, -CANVAS_HEIGHT))
    # strokepath()

    rotate(rotation * randomize_between(rng, 0.8, 1.2))
    translation_amount_x = CANVAS_WIDTH / scaling_offset_x / 3
    translation_amount_y = CANVAS_HEIGHT / scaling_offset_y / 3
    translation_noise_x = randomize_between(rng, -translation_amount_x, translation_amount_x)
    translation_noise_y = randomize_between(rng, -translation_amount_y, translation_amount_y)
    translate(translation_noise_x, translation_noise_y)
    transform(transformation)

    sethue("black")
    p1 = base_left_bezier_point

    should_hook_back = true
    if should_hook_back
        p2 = base_middle_bezier_point_hook_variant
        p3 = base_right_bezier_point_hook_start
        curve(p1, p2, p3)
        strokepath()
        setline(randomize_between(rng, 1, 3))

        p4 = base_right_bezier_point_hook_middle
        p5 = base_right_bezier_point_hook_end
        curve(p3, p4, p5)
    else
        p2 = base_middle_bezier_point
        p3 = base_right_bezier_point
        curve(p1, p2, p3)
    end

    strokepath()

    finish()
end

function draw_stroke(args...; kwargs...)
    draw_stroke(default_rng(), args...; kwargs...)
end

function draw_line(rng::AbstractRNG, letter::Letter, path::String)
    println("saving to $path")

    Drawing(CANVAS_HEIGHT, CANVAS_WIDTH, path)
    origin(CENTER_X, CENTER_Y)

    bg_grey = randomize_between(rng, 0.7, 1.0)
    background(bg_grey, bg_grey, bg_grey)

    D = 100
    mat = [Luxor.ARGB32(
        noise(0.01r, 0.01c),
        noise(0.01r, 0.01c),
        noise(0.01r, 0.01c),
        noise(0.01r, 0.01c)) for r in 1:D, c in 1:D]
    placeimage(mat, boxtopleft())

    setline(randomize_between(rng, 1, 2))

    # these values are manually tuned
    (scaling_offset_x, scaling_offset_y) =
        if letter == _N || letter == _T
            (6, 6)
        elseif letter == _M || letter == _D
            (3, 4)
        else
            error("$letter is not a valid Gregg letter for stroke")
        end
    jitter_x() = randomize_between(rng, 0.75, 1.25)
    jitter_y() = randomize_between(rng, 0.75, 1.25)
    jitter_x(x, y) = randomize_between(rng, x, y)
    jitter_y(x, y) = randomize_between(rng, x, y)


    base_left_point = Point(-CANVAS_WIDTH / scaling_offset_x, randomize_between(rng, -CANVAS_HEIGHT / scaling_offset_y / 4, CANVAS_HEIGHT / scaling_offset_y / 4)) * (jitter_x(), 1.0)
    base_right_point = Point(CANVAS_WIDTH / scaling_offset_x, randomize_between(rng, -CANVAS_HEIGHT / scaling_offset_y / 4, CANVAS_HEIGHT / scaling_offset_y / 4)) * (jitter_x(), 1.0)


    rotation =
        if letter == _N || letter == _M
            randomize_between(rng, -deg2rad(3), deg2rad(3))
        elseif letter == _T || letter == _D
            randomize_between(rng, -deg2rad(40), -deg2rad(20))
        else
            error("$letter is not a valid Gregg letter for stroke")
        end
    rotate(rotation)
    translation_amount_x = CANVAS_WIDTH / scaling_offset_x / 3
    translation_amount_y = CANVAS_HEIGHT / scaling_offset_y / 3
    translation_noise_x = randomize_between(rng, -translation_amount_x, translation_amount_x)
    translation_noise_y = randomize_between(rng, -translation_amount_y, translation_amount_y)
    translate(translation_noise_x, translation_noise_y)

    sethue("black")
    line(base_left_point, base_right_point)
    strokepath()

    finish()
end

function draw_line(args...; kwargs...)
    draw_line(default_rng(), args...; kwargs...)
end


function test_func()
    println("test function")
    return "hi"
end

end
