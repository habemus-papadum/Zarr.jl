function pipeline_encode(p::V2Pipeline, data::AbstractArray, fill_value)
    if fill_value !== nothing && all(isequal(fill_value), data)
        return nothing
    end
    dtemp = UInt8[]
    zcompress!(dtemp, data, p.compressor, p.filters)
    return dtemp
end

function pipeline_decode!(p::V2Pipeline, output::AbstractArray, compressed::Vector{UInt8}; fill_value=nothing)
    zuncompress!(output, compressed, p.compressor, p.filters)
    return output
end

function pipeline_encode(p::V3Pipeline, data::AbstractArray, fill_value)
    if fill_value !== nothing && all(isequal(fill_value), data)
        return nothing
    end
    # Phase 1: array->array codecs (forward order)
    result = data
    for codec in p.array_array
        result = Codecs.V3Codecs.codec_encode(codec, result)
    end
    # Phase 2: array->bytes codec
    bytes = Codecs.V3Codecs.codec_encode(p.array_bytes, result)
    # Phase 3: bytes->bytes codecs (forward order)
    for codec in p.bytes_bytes
        bytes = Codecs.V3Codecs.codec_encode(codec, bytes)
    end
    return bytes
end

function pipeline_decode!(p::V3Pipeline, output::AbstractArray, compressed::Vector{UInt8}; fill_value=nothing)
    intermediate_shape = foldl(
        (sz, codec) -> Codecs.V3Codecs.encoded_shape(codec, sz),
        p.array_array; init=size(output)
    )

    n_bb = length(p.bytes_bytes)
    ab   = p.array_bytes

    # Hot path: pipeline is exactly `[BytesCodec, one bytes_bytes codec]`
    # with matching endian. The bytes-bytes step's output IS the byte
    # view of `output`'s underlying memory, so we decompress straight
    # into it — no intermediate Vector{UInt8} allocation, no second
    # copy from a scratch buffer into the typed array. This is what
    # this archive (and most well-configured sharded archives) exercise
    # on every inner-chunk decode.
    if isempty(p.array_array) && n_bb == 1 && ab isa Codecs.V3Codecs.BytesCodec &&
       !Codecs.V3Codecs._needs_bswap(ab.endian)
        bytes_view = reinterpret(UInt8, vec(output))
        Codecs.V3Codecs.codec_decode!(only(p.bytes_bytes), bytes_view, compressed)
        return output
    end

    # Endian-mismatch variant: still avoid the extra buffer by
    # decoding into the byte view, then byte-swapping in place via the
    # array_bytes codec's in-place dispatch.
    if isempty(p.array_array) && n_bb == 1 && ab isa Codecs.V3Codecs.BytesCodec
        bytes_view = reinterpret(UInt8, vec(output))
        Codecs.V3Codecs.codec_decode!(only(p.bytes_bytes), bytes_view, compressed)
        # codec_decode!(::BytesCodec) handles bswap when needed.
        Codecs.V3Codecs.codec_decode!(ab, output, bytes_view; fill_value)
        return output
    end

    # Multi bytes-bytes step (rare): need one chunk-sized scratch buffer
    # to chain through. Final step writes into the buffer; array_bytes
    # then writes into output. Only valid when the array_bytes codec has
    # a fixed encoded byte size — otherwise `decoded_byte_size` (=
    # prod(shape) × sizeof(eltype)) is meaningless (e.g. for variable-
    # length string codecs) and we fall through to the allocating
    # fallback below.
    if isempty(p.array_array) && n_bb >= 1 && Codecs.V3Codecs.is_fixed_size(ab)
        # `sizeof(eltype(output))` is only valid for fixed-size codecs;
        # variable-size codecs (e.g. vlen-utf8) bypass this branch via
        # the `is_fixed_size(ab)` guard above.
        decoded_byte_size = prod(intermediate_shape) * sizeof(eltype(output))
        bytes_buf = Vector{UInt8}(undef, decoded_byte_size)
        bytes = compressed
        bb = collect(p.bytes_bytes)
        for i in length(bb):-1:1
            codec = bb[i]
            if i == 1
                Codecs.V3Codecs.codec_decode!(codec, bytes_buf, bytes)
            else
                bytes = Codecs.V3Codecs.codec_decode(codec, bytes)
            end
        end
        Codecs.V3Codecs.codec_decode!(ab, output, bytes_buf; fill_value)
        return output
    end

    # No bytes-bytes step (uncompressed): array_bytes from the encoded
    # input directly into `output`. Requires `n_bb == 0` — the original
    # bug here matched any pipeline with no array_array codecs and
    # silently skipped the bytes-bytes chain (catastrophic for e.g.
    # vlen-utf8 + zstd where the input still needs decompressing).
    if isempty(p.array_array) && n_bb == 0
        Codecs.V3Codecs.codec_decode!(ab, output, compressed; fill_value)
        return output
    end

    # Fallback for pipelines with array_array codecs (e.g. transpose).
    # Allocate as before — these are uncommon enough that further
    # tuning isn't worth the case-analysis.
    bytes = compressed
    for codec in reverse(collect(p.bytes_bytes))
        bytes = Codecs.V3Codecs.codec_decode(codec, bytes)
    end
    arr = Codecs.V3Codecs.codec_decode(ab, bytes, eltype(output), intermediate_shape; fill_value)
    for codec in reverse(collect(p.array_array))
        arr = Codecs.V3Codecs.codec_decode(codec, arr)
    end
    copyto!(output, arr)
    return output
end

# Convenience: extract pipeline from metadata
get_pipeline(m::MetadataV2) = V2Pipeline(m.compressor, m.filters)
get_pipeline(m::MetadataV3) = m.pipeline
