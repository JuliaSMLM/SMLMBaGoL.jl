# Posterior image generation from RJMCMC chain samples

"""
    posterior_image(chain::RJMCMCChain; pixel_size, xlim=nothing, ylim=nothing)

Generate a 2D posterior image by histogramming emitter positions from all
post-burn-in chain samples.

Returns `NamedTuple{(:image, :edges_x, :edges_y, :pixel_size)}` where `image`
is a matrix of raw counts. Plot with `heatmap(post.edges_x, post.edges_y, post.image')`.

# Arguments
- `pixel_size`: Histogram bin size in μm (required)
- `xlim`: `(xmin, xmax)` bounds; auto-computed from data if `nothing`
- `ylim`: `(ymin, ymax)` bounds; auto-computed from data if `nothing`
"""
function posterior_image(chain::RJMCMCChain;
                         pixel_size::Float64,
                         xlim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
                         ylim::Union{Nothing, Tuple{Float64, Float64}} = nothing)
    xs = Float64[]
    ys = Float64[]
    for sample in chain.samples
        for e in sample.emitters
            push!(xs, Float64(e.x))
            push!(ys, Float64(e.y))
        end
    end
    return _histogram_positions(xs, ys, pixel_size, xlim, ylim)
end

"""
    posterior_image(chains::Vector{<:RJMCMCChain}; pixel_size, xlim=nothing, ylim=nothing)

Generate a 2D posterior image accumulated from multiple chains (partitioned BaGoL).
"""
function posterior_image(chains::Vector{<:RJMCMCChain};
                         pixel_size::Float64,
                         xlim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
                         ylim::Union{Nothing, Tuple{Float64, Float64}} = nothing)
    xs = Float64[]
    ys = Float64[]
    for chain in chains
        for sample in chain.samples
            for e in sample.emitters
                push!(xs, Float64(e.x))
                push!(ys, Float64(e.y))
            end
        end
    end
    return _histogram_positions(xs, ys, pixel_size, xlim, ylim)
end

"""
    _histogram_positions(xs, ys, pixel_size, xlim, ylim)

Internal: bin positions into a 2D integer histogram.
"""
function _histogram_positions(xs::Vector{Float64}, ys::Vector{Float64},
                              pixel_size::Float64,
                              xlim::Union{Nothing, Tuple{Float64, Float64}},
                              ylim::Union{Nothing, Tuple{Float64, Float64}})
    if isempty(xs)
        edges_x = Float64[0.0, pixel_size]
        edges_y = Float64[0.0, pixel_size]
        image = zeros(Int, 1, 1)
        return (image=image, edges_x=edges_x, edges_y=edges_y, pixel_size=pixel_size)
    end

    xmin, xmax = xlim !== nothing ? xlim : (minimum(xs), maximum(xs))
    ymin, ymax = ylim !== nothing ? ylim : (minimum(ys), maximum(ys))

    # Build edge vectors
    edges_x = collect(xmin:pixel_size:(xmax + pixel_size))
    edges_y = collect(ymin:pixel_size:(ymax + pixel_size))

    nx = length(edges_x) - 1
    ny = length(edges_y) - 1
    image = zeros(Int, nx, ny)

    for k in eachindex(xs)
        ix = floor(Int, (xs[k] - xmin) / pixel_size) + 1
        iy = floor(Int, (ys[k] - ymin) / pixel_size) + 1
        if 1 <= ix <= nx && 1 <= iy <= ny
            image[ix, iy] += 1
        end
    end

    return (image=image, edges_x=edges_x, edges_y=edges_y, pixel_size=pixel_size)
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
- `post`: NamedTuple from `posterior_image()`
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
