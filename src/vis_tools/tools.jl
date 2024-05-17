function normalize!(im::BGLImage2D)
    min_val = minimum(im.data).val
    max_val = maximum(im.data).val
    for i in eachindex(im.data)
        im.data[i] = (im.data[i] - min_val) / (max_val - min_val)
    end
end
