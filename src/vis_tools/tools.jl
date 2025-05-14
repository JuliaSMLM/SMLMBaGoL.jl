function normalize!(im::BGLImage2D)
    min_val = minimum(im.data).val
    max_val = maximum(im.data).val
    for i in eachindex(im.data)
        im.data[i] = (im.data[i] - min_val) / (max_val - min_val)
    end
end

# Make percentile normalization function
function quantile_stretch!(arr::AbstractArray; max_quantile::Float64=100.0)
    max_val = quantile(arr[arr.>0], max_quantile)
    min_val = minimum(arr)
    for i in eachindex(arr)
        if arr[i] > max_val
            arr[i] = max_val
        end
    end

    for i in eachindex(arr)
        arr[i] = (arr[i] - min_val) / (max_val - min_val)
    end
end

function quantile_stretch!(arr::AbstractArray{Gray}; max_quantile::Float64=100.0)
    max_val = quantile(arr[arr.>0], max_quantile).val
    min_val = minimum(arr)
    for i in eachindex(arr)
        if arr[i] > max_val
            arr[i] = max_val
        end
    end

    for i in eachindex(arr)
        arr[i] = (arr[i].val - min_val) / (max_val - min_val)
    end
end



# Make percentile normalization function
function quantile_stretch!(im::BGLImage2D; max_quantile::Float64=100.0)
    quantile_stretch!(im.data, max_quantile=max_quantile)
end

