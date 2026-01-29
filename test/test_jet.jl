@testset "JET Precompilation Report" begin
    ignored_modules = [HDF5, AnyFrameModule(ProgressMeter)]
    jet_report = JET.report_package(AtmosphericTurbulenceSimulator; toplevel_logger=nothing,
        ignored_modules=[HDF5, AnyFrameModule(ProgressMeter)])
    print(jet_report)
    @test length(JET.get_reports(jet_report)) <= 22
    @test_broken length(JET.get_reports(jet_report)) == 0
end
