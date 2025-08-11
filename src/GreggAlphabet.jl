module GreggAlphabet

using Match

export Letter, _K, _G, _R, _L, _P, _B, _F, _V, _N, _M, _T, _D, _H, _TH_RIGHT, _TH_LEFT, _S_RIGHT, _S_LEFT, _A, _O, _E, _U, _SH, _CH, _J, _NG, _NK,
    to_string

@enum Letter begin
    _K
    _G
    _R
    _L
    _P
    _B
    _F
    _V
    _N
    _M
    _T
    _D
    _H
    _TH_RIGHT
    _TH_LEFT
    _S_RIGHT
    _S_LEFT
    _A
    _O
    _E
    _U
    _SH
    _CH
    _J
    _NG
    _NK
end

function to_string(letter::Letter)
    val = @match letter begin
        $_K => "k"
        $_G => "g"
        $_R => "r"
        $_L => "l"
        $_P => "p"
        $_B => "b"
        $_F => "f"
        $_V => "v"
        $_N => "n"
        $_M => "m"
        $_T => "t"
        $_D => "d"
        $_H => "h"
        $_TH_RIGHT => "th(right)"
        $_TH_LEFT => "th(left)"
        $_S_RIGHT => "s(right)"
        $_S_LEFT => "s(left)"
        $_A => "a"
        $_O => "o"
        $_E => "e"
        $_U => "u"
        $_SH => "sh"
        $_CH => "ch"
        $_J => "j"
        $_NG => "ng"
        $_NK => "nk"
        _ => error("$letter is not a valid Gregg shorthand letter")
    end
    return val
end

function Base.show(io::IO, letter::Letter)
    print(io, to_string(letter))
end

# function Base.show(io::IO, letters::Vector{Letter})
#     print(io, """[$(join(print.(letters), ", "))]""")
# end

end
