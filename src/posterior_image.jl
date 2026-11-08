# Posterior image utilities

"""
    _resample_mass(post, x_min, x_max, y_min, y_max, W, H) -> Matrix{Float64}

Internal: the posterior image `post` (NamedTuple with `image[ix, iy]`, `edges_x`,
`edges_y`, `pixel_size`, μm) put on a W×H grid spanning [x_min, x_max]×[y_min, y_max]
μm, `[ix, iy]` layout. Each output pixel gets the image mass inside its footprint:
every source pixel's mass is split among the output pixels it overlaps in proportion
to the overlap area. Mass is conserved at any ratio of the two pixel sizes, so an
emitter keeps its brightness whatever the render pixel (picking the source pixel
under each output pixel's center made a sharp emitter's brightness depend on where it
sat within a pixel).
"""
function _resample_mass(post::NamedTuple, x_min, x_max, y_min, y_max, W::Int, H::Int)
    src = post.image
    nx, ny = size(src)
    sp = Float64(post.pixel_size)
    wx = _overlap_weights(Float64(post.edges_x[1]), sp, nx, Float64(x_min), (x_max - x_min) / W, W)
    wy = _overlap_weights(Float64(post.edges_y[1]), sp, ny, Float64(y_min), (y_max - y_min) / H, H)
    out = zeros(W, H)
    @inbounds for iy in 1:ny
        isempty(wy[iy]) && continue
        for ix in 1:nx
            v = Float64(src[ix, iy])
            (v == 0 || isempty(wx[ix])) && continue
            for (oy, fy) in wy[iy], (ox, fx) in wx[ix]
                out[ox, oy] += v * fx * fy
            end
        end
    end
    return out
end

# Internal (1D): for source pixel i, edges s0 + (i-1)·sp .. s0 + i·sp, the output pixels j,
# edges o0 + (j-1)·op .. o0 + j·op, that it overlaps and the fraction of it in each.
function _overlap_weights(s0::Float64, sp::Float64, n::Int, o0::Float64, op::Float64, m::Int)
    w = [Tuple{Int, Float64}[] for _ in 1:n]
    for i in 1:n
        a = s0 + (i - 1) * sp
        b = a + sp
        jlo = max(1, floor(Int, (a - o0) / op) + 1)
        jhi = min(m, floor(Int, (b - o0) / op) + 1)
        for j in jlo:jhi
            lo = max(a, o0 + (j - 1) * op)
            hi = min(b, o0 + j * op)
            hi > lo && push!(w[i], (j, (hi - lo) / sp))
        end
    end
    return w
end

# ============================================================================
# PNG saving with percentile scaling and colormaps
# ============================================================================

"""
    save_posterior_png(filename, post; percentile=0.99, colormap=:inferno)

Save posterior image as a color PNG with percentile scaling.

Scales non-zero pixels so the `percentile` value maps to 255.
Values above the percentile are clipped to 255.

# Arguments
- `filename`: Output path (should end in .png)
- `post`: a posterior image NamedTuple (`image`, `edges_x`, `edges_y`, `pixel_size`), e.g.
  `diagnostics.posterior_image`
- `percentile`: Percentile of non-zero pixels for max scaling (default 0.99)
- `colormap`: `:inferno` (default), `:hot`, or `:gray`
"""
function save_posterior_png(filename::String, post::NamedTuple;
                            percentile::Float64=0.99,
                            colormap::Symbol=:inferno)
    img = post.image
    nx, ny = size(img)

    lut = _get_colormap(colormap)

    # Percentile scaling on non-zero pixels
    nonzero = filter(>(0), vec(img))
    if isempty(nonzero)
        rgb = zeros(UInt8, ny, nx, 3)
    else
        sorted_nz = sort(nonzero)
        idx = clamp(round(Int, percentile * length(sorted_nz)), 1, length(sorted_nz))
        vmax = Float64(sorted_nz[idx])
        # Build PNG matrix: rows=y, cols=x, 3 channels
        # Camera convention: y_min at top (row 1), y increases downward
        rgb = Array{UInt8, 3}(undef, ny, nx, 3)
        for iy in 1:ny
            for ix in 1:nx
                v = clamp(round(Int, 255.0 * min(Float64(img[ix, iy]) / vmax, 1.0)), 0, 255)
                ci = v + 1  # 1-indexed into LUT
                rgb[iy, ix, 1] = lut[ci, 1]
                rgb[iy, ix, 2] = lut[ci, 2]
                rgb[iy, ix, 3] = lut[ci, 3]
            end
        end
    end

    _save_png_rgb(filename, rgb)
end

# ============================================================================
# Built-in colormaps (no external dependencies)
# ============================================================================

function _get_colormap(name::Symbol)
    if name === :inferno
        return _INFERNO_LUT
    elseif name === :hot
        return _HOT_LUT
    elseif name === :gray
        return _GRAY_LUT
    else
        error("Unknown colormap :$name. Use :inferno, :hot, or :gray.")
    end
end

# Build a 256×3 LUT by linearly interpolating control points
function _build_lut(points::Vector{Tuple{Float64, UInt8, UInt8, UInt8}})
    lut = Matrix{UInt8}(undef, 256, 3)
    np = length(points)
    for i in 0:255
        t = i / 255.0
        # Find segment
        j = 1
        while j < np && points[j + 1][1] <= t
            j += 1
        end
        if j >= np
            lut[i + 1, 1] = points[np][2]
            lut[i + 1, 2] = points[np][3]
            lut[i + 1, 3] = points[np][4]
        else
            t0, r0, g0, b0 = points[j]
            t1, r1, g1, b1 = points[j + 1]
            f = (t - t0) / (t1 - t0)
            lut[i + 1, 1] = round(UInt8, clamp(r0 + f * (Float64(r1) - Float64(r0)), 0, 255))
            lut[i + 1, 2] = round(UInt8, clamp(g0 + f * (Float64(g1) - Float64(g0)), 0, 255))
            lut[i + 1, 3] = round(UInt8, clamp(b0 + f * (Float64(b1) - Float64(b0)), 0, 255))
        end
    end
    return lut
end

# Inferno colormap (17 control points from matplotlib)
const _INFERNO_LUT = _build_lut([
    (0.000, UInt8(0),   UInt8(0),   UInt8(4)),
    (0.063, UInt8(11),  UInt8(7),   UInt8(52)),
    (0.125, UInt8(40),  UInt8(11),  UInt8(84)),
    (0.188, UInt8(66),  UInt8(10),  UInt8(104)),
    (0.250, UInt8(89),  UInt8(13),  UInt8(107)),
    (0.313, UInt8(111), UInt8(20),  UInt8(101)),
    (0.375, UInt8(133), UInt8(33),  UInt8(88)),
    (0.438, UInt8(152), UInt8(48),  UInt8(72)),
    (0.500, UInt8(169), UInt8(65),  UInt8(56)),
    (0.563, UInt8(184), UInt8(83),  UInt8(42)),
    (0.625, UInt8(197), UInt8(102), UInt8(29)),
    (0.688, UInt8(208), UInt8(123), UInt8(17)),
    (0.750, UInt8(218), UInt8(145), UInt8(10)),
    (0.813, UInt8(226), UInt8(168), UInt8(13)),
    (0.875, UInt8(231), UInt8(191), UInt8(27)),
    (0.938, UInt8(233), UInt8(215), UInt8(57)),
    (1.000, UInt8(252), UInt8(255), UInt8(164)),
])

# Hot colormap (black → red → yellow → white)
const _HOT_LUT = _build_lut([
    (0.000, UInt8(0),   UInt8(0),   UInt8(0)),
    (0.333, UInt8(255), UInt8(0),   UInt8(0)),
    (0.667, UInt8(255), UInt8(255), UInt8(0)),
    (1.000, UInt8(255), UInt8(255), UInt8(255)),
])

# Gray colormap
const _GRAY_LUT = _build_lut([
    (0.0, UInt8(0),   UInt8(0),   UInt8(0)),
    (1.0, UInt8(255), UInt8(255), UInt8(255)),
])

# ============================================================================
# Pure-Julia PNG writer (zero dependencies)
# ============================================================================

const _CRC32_TABLE = let
    table = Vector{UInt32}(undef, 256)
    for i in 0:255
        crc = UInt32(i)
        for _ in 1:8
            if crc & 1 != 0
                crc = xor(crc >> 1, 0xEDB88320)
            else
                crc >>= 1
            end
        end
        table[i + 1] = crc
    end
    table
end

function _crc32(data::AbstractVector{UInt8}, crc::UInt32=0xFFFFFFFF)
    for b in data
        crc = xor(_CRC32_TABLE[(xor(crc, UInt32(b)) & 0xFF) + 1], crc >> 8)
    end
    return xor(crc, 0xFFFFFFFF)
end

function _adler32(data::AbstractVector{UInt8})
    a = UInt32(1)
    b = UInt32(0)
    for byte in data
        a = (a + UInt32(byte)) % UInt32(65521)
        b = (b + a) % UInt32(65521)
    end
    return (b << 16) | a
end

function _write_png_chunk(io::IO, chunk_type::Vector{UInt8}, data::Vector{UInt8})
    write(io, hton(UInt32(length(data))))
    write(io, chunk_type)
    write(io, data)
    crc = _crc32(vcat(chunk_type, data))
    write(io, hton(crc))
end

function _zlib_store(data::Vector{UInt8})
    result = UInt8[0x78, 0x01]  # zlib header: deflate, no dict
    max_block = 65535
    offset = 1
    remaining = length(data)

    while remaining > 0
        block_size = min(remaining, max_block)
        is_final = remaining <= max_block ? UInt8(0x01) : UInt8(0x00)
        push!(result, is_final)
        len = UInt16(block_size)
        nlen = ~len
        append!(result, reinterpret(UInt8, [len]))
        append!(result, reinterpret(UInt8, [nlen]))
        append!(result, @view data[offset:offset + block_size - 1])
        offset += block_size
        remaining -= block_size
    end

    checksum = _adler32(data)
    append!(result, reinterpret(UInt8, [hton(checksum)]))
    return result
end

function _save_png_rgb(filename::String, img::Array{UInt8, 3})
    height, width, _ = size(img)

    sig = UInt8[137, 80, 78, 71, 13, 10, 26, 10]

    # IHDR
    ihdr = UInt8[]
    append!(ihdr, reinterpret(UInt8, [hton(UInt32(width))]))
    append!(ihdr, reinterpret(UInt8, [hton(UInt32(height))]))
    push!(ihdr, 0x08)  # bit depth 8
    push!(ihdr, 0x02)  # color type: truecolor RGB
    push!(ihdr, 0x00)  # compression
    push!(ihdr, 0x00)  # filter
    push!(ihdr, 0x00)  # interlace

    # Raw scanlines: filter_byte + R G B per pixel
    raw = Vector{UInt8}(undef, height * (1 + 3 * width))
    idx = 1
    for r in 1:height
        raw[idx] = 0x00  # filter: none
        idx += 1
        for c in 1:width
            raw[idx]     = img[r, c, 1]
            raw[idx + 1] = img[r, c, 2]
            raw[idx + 2] = img[r, c, 3]
            idx += 3
        end
    end

    zlib = _zlib_store(raw)

    open(filename, "w") do io
        write(io, sig)
        _write_png_chunk(io, collect(UInt8, "IHDR"), ihdr)
        _write_png_chunk(io, collect(UInt8, "IDAT"), zlib)
        _write_png_chunk(io, collect(UInt8, "IEND"), UInt8[])
    end
end
