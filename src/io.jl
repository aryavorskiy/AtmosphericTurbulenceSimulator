using HDF5, ProgressMeter

const DEFAULT_BATCH = 128

struct BufferedDataset{Dt, Bt}
    dataset::Dt
    buffer::Bt
end
function _make_buffer(ds::HDF5.Dataset, batch::Int)
    props = HDF5.get_create_properties(ds)
    if batch == 0
        if props.layout === :chunked
            batch = props.chunk[3]
        else
            batch = DEFAULT_BATCH
        end
    end
    return zeros(eltype(ds)::Type, size(ds, 1)::Int, size(ds, 2)::Int, batch)
end
_make_buffer(::Union{<:AbstractArray, Nothing}, _) = nothing
BufferedDataset(ds1, batch::Int=0) = BufferedDataset(ds1, _make_buffer(ds1, batch))

function write_batch!(bd::BufferedDataset{<:HDF5.Dataset}, j, batch)
    copy!(bd.buffer, batch)
    HDF5.do_write_chunk(bd.dataset, (1, 1, (j - 1) * size(batch, 3) + 1), bd.buffer)
end
function write_batch!(bd::BufferedDataset{<:AbstractArray}, j, batch)
    batch_len = size(batch, 3)
    dset_len = size(bd.dataset, 3)::Int
    j1 = (j - 1) * batch_len + 1
    if dset_len > j * batch_len
        bd.dataset[:, :, j1:j1 + batch_len - 1] .= batch
    else
        bd.dataset[:, :, j1:end] .= @view batch[:, :, 1:dset_len - j1 + 1]
    end
end
write_batch!(::BufferedDataset{Nothing}, _, _) = nothing

_copyto!(dest, ds::HDF5.Dataset, ix, iy, iz) = copyto!(dest, ds, ix, iy, iz)
_copyto!(dest, ds::AbstractArray, ix, iy, iz) = copy!(dest, @view ds[ix, iy, iz])
function read_batch!(dest, bd::BufferedDataset, j, ix=Colon(), iy=Colon())
    batch_len = size(dest, 3)
    nframes = size(bd.dataset, 3)
    j1 = (j - 1) * batch_len + 1
    j1 > nframes && throw(BoundsError(bd.dataset, (ix, iy, j1)))
    n_avail = min(batch_len, nframes - j1 + 1)
    if n_avail == batch_len
        if bd.buffer === nothing || typeof(bd.buffer) === typeof(dest)
            _copyto!(dest, bd.dataset, ix, iy, j1:j1 + batch_len - 1)
        else
            _copyto!(bd.buffer, bd.dataset, :, :, j1:j1 + batch_len - 1)
            dest .= @view bd.buffer[ix, iy, :]
        end
    else
        dest[:, :, 1:n_avail] .= bd.dataset[ix, iy, j1:nframes]
        dest[:, :, n_avail + 1:end] .= NaN
    end
end

struct HDF5File
    filename::String
    group::String
    overwrite::Bool
end

"""
    HDF5File(filename[, group=""][; overwrite=false])

A convenience struct for specifying HDF5 output options.

# Arguments
- `filename`: name of the HDF5 file to write to.
- `group`: optional group within the HDF5 file to write datasets to (default: root group).
- `overwrite`: if `true`, overwrite the group if it already exists (or the entire file if `group=""`),
    otherwise throw an error if datasets with the same name already exist (default: `false`).
"""
HDF5File(filename::String, group::String; overwrite::Bool=false) = HDF5File(filename, group, overwrite)
function HDF5File(filename::String; group="", kw...) # deprecated
    group != "" && Base.depwarn("`HDF5File(filename; group=gr)` is deprecated and will be removed in v0.5; use `HDF5File(filename, gr)` instead.", :HDF5File)
    HDF5File(filename, group; kw...)
end

function open_file(f::Function, h5file::HDF5File)
    h5open(h5file.filename, h5file.overwrite && h5file.group == "" ? "w" : "cw") do fid
        if h5file.group != ""
            h5file.overwrite && haskey(fid, h5file.group) && HDF5.delete_object(fid[h5file.group])
            f(create_group(fid, h5file.group))
        else
            f(fid)
        end
    end
end
open_file(f::Function, filename::String) = if endswith(lowercase(filename), r".h(df)?5")
    open_file(f, HDF5File(filename))
else
    throw(ArgumentError("Unsupported file extension: $filename. HDF5 expected."))
end
open_file(f::Function, ::Nothing) = f(nothing)

prepare_dataset(fid::Union{HDF5.File,HDF5.Group}, name::String, type, sz, n, batch) =
    BufferedDataset(create_dataset(fid, name, type, (sz..., n), chunk=(sz..., batch)), batch)
prepare_dataset(::Nothing, ::String, ::Type{T}, sz, n, batch::Int) where T =
    BufferedDataset(Array{T}(undef, (sz..., n)), batch)
