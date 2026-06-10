import AtmosphericTurbulenceSimulator: BufferedDataset, read_batch!, write_batch!, HDF5File, open_file, prepare_dataset

@testset "IO" begin
    @testset "BufferedDataset" begin
        data = Float64.(LinearIndices((4, 5, 12)))
        batch = 5

        @testset "write_batch!" begin
            batch3 = cat(data[:, :, 11:12], zeros(4, 5, 3), dims=3)
            dest = zeros(4, 5, 12)
            bd = BufferedDataset(dest, batch)
            @test bd.buffer === nothing
            write_batch!(bd, 1, data[:, :, 1:5])
            write_batch!(bd, 2, data[:, :, 6:10])
            write_batch!(bd, 3, batch3)
            @test dest == data
            # Should error if trying to write wrong batch length
            # @test_throws BoundsError write_batch!(bd, 3, data[:, :, 11:12])

            dest2 = zeros(4, 5, 10)
            bd2 = BufferedDataset(dest2, batch)
            write_batch!(bd2, 1, data[:, :, 1:5])
            write_batch!(bd2, 2, data[:, :, 6:10])
            @test dest2 == data[:, :, 1:10]
            @test_throws BoundsError write_batch!(bd2, 3, batch3)
        end

        @testset "read_batch!" begin
            bd = BufferedDataset(data, batch)
            out = zeros(4, 5, 5)
            read_batch!(out, bd, 1)
            @test out == data[:, :, 1:5]
            read_batch!(out, bd, 2)
            @test out == data[:, :, 6:10]

            # last is not written
            fill!(out, NaN)
            read_batch!(out, bd, 3)
            @test out[:, :, 1:2] == data[:, :, 11:12]
            @test all(isnan, out[:, :, 3:5])    # no writes beyond available frames

            # non-existent batch
            @test_throws BoundsError read_batch!(out, bd, 4)

            # crop indices
            bd = BufferedDataset(data, batch)
            out = zeros(3, 4, 5)
            read_batch!(out, bd, 1, 1:3, 1:4)
            @test out == data[1:3, 1:4, 1:5]
        end
    end

    @testset "Nothing backend" begin
        bd = BufferedDataset(nothing)
        @test bd.dataset === nothing
        @test bd.buffer === nothing
        @test try
            write_batch!(bd, 1, rand(3, 3, 2))
            true # should not error
        catch e
            false
        end
    end

    @testset "HDF5File" begin
        tmpfile = tempname() * ".h5"

        @testset "Constructors" begin
            h = HDF5File(tmpfile)
            @test h.filename == tmpfile
            @test h.group == ""
            @test h.overwrite == false

            h2 = HDF5File(tmpfile, "mygroup")
            @test h2.group == "mygroup"

            h3 = HDF5File(tmpfile; overwrite=true)
            @test h3.overwrite == true
        end

        @testset "File operations" begin
            open_file(HDF5File(tmpfile, "g1")) do fid
                create_dataset(fid, "x", Float32, (3, 3, 2))
            end
            h5open(tmpfile, "r") do fid
                @test haskey(fid, "g1")
                @test haskey(fid["g1"], "x")
            end

            # overwrite should replace the group
            open_file(HDF5File(tmpfile, "g1"; overwrite=true)) do fid
                create_dataset(fid, "y", Float32, (2, 2, 2))
            end
            h5open(tmpfile, "r") do fid
                @test !haskey(fid["g1"], "x")
                @test haskey(fid["g1"], "y")
            end

            # bad extension should error
            @test_throws ArgumentError open_file(_ -> nothing, "output.txt")
        end

        @testset "Nothing passthrough" begin
            called = Ref(false)
            open_file(nothing) do fid
                called[] = true
                @test fid === nothing
            end
            @test called[]
        end

        rm(tmpfile, force=true)
    end
end
