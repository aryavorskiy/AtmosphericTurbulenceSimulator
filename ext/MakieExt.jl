module MakieExt

using AtmosphericTurbulenceSimulator, Makie

"""
    speckle_viewer(; kwargs...)

Open an interactive Makie window showing a simulated phase screen and the speckle image it
produces. Controls are split into two columns:

- **Atmosphere**: sliders for the Fried parameter ``r_0`` and the wind speed (direction fixed
  horizontal), plus a *New phase* button that draws a fresh random phase screen.
- **Imaging**: sliders for the central wavelength, filter bandwidth and exposure time.

The phase screen is recomputed only when an atmosphere control changes (or the button is pressed);
the speckle image is recomputed on any control change. A fixed circular aperture is used.

# Keyword Arguments
- `wavelength_range`: central-wavelength slider values, in nm.
- `bw_range`: filter-bandwidth slider values, in nm (0 ⇒ monochromatic).
- `exptime_range`: exposure-time slider values, in s (0 ⇒ short exposure).
- `r0_range`: Fried-parameter slider values, in cm.
- `wind_range`: wind-speed slider values (same length units as `d`, per second).
- `apsize`: aperture grid size (default `(64, 64)`).
- `d`: aperture diameter (default `200`).
- `nphotons`: photon budget per image (default `1e6`).
"""
function AtmosphericTurbulenceSimulator.speckle_viewer(;
        wavelength_range=550:10:750, # nm
        bw_range=0:10:100,           # nm
        exptime_range=0:0.1:3,       # s
        r0_range=10:5:30,            # cm
        wind_range=0:5:100,          # length units / s
        aperture = CircularAperture((64, 64)),
        d=200
    )

    fig = Figure(size=(1200, 800))

    Label(fig[1, 1:2], "Atmosphere", font=:bold, fontsize=18, tellwidth=false)
    atm_sg = SliderGrid(fig[2, 1:2], valign=:top, alignmode=Inside(),
        (label="r₀", range=r0_range, startvalue=first(r0_range), format="{:d} cm"),
        (label="Wind", range=wind_range, startvalue=first(wind_range), format="{:d}/s"))
    r0_s, wind_s = (s.value for s in atm_sg.sliders)
    buttons_grid = GridLayout(atm_sg.layout[3, 2], tellwidth=false, halign=:left, colgap=5)
    newphase_btn = Button(buttons_grid[1, 1], label="New phase", halign=:left)
    remove_tiptilt_btn = Button(buttons_grid[1, 2], label="Remove tip/tilt", halign=:left)

    Label(fig[1, 3:4], "Imaging", font=:bold, fontsize=18, tellwidth=false)
    img_sg = SliderGrid(fig[2, 3:4], valign=:top, alignmode=Inside(),
        (label="Wavelength", range=wavelength_range, startvalue=first(wavelength_range), format="{:d} nm"),
        (label="Bandwidth", range=bw_range, startvalue=first(bw_range), format="{:d} nm"),
        (label="Exposure", range=exptime_range, startvalue=first(exptime_range), format="{:.1f} s"))
    wl_s, bw_s, exp_s = (s.value for s in img_sg.sliders)

    phase_screen = lift(r0_s, newphase_btn.clicks) do r0, _
        atm = SingleLayer(r0; interpolate=:auto)
        simulate_phases(atm, ceil.(Int, size(aperture) .* (1 + maximum(wind_range) / d * maximum(exptime_range), 1));
            n=1, verbose=false, grid_step=d / size(aperture, 2))
    end

    on(remove_tiptilt_btn.clicks) do _
        phs_tot = phase_screen[]
        phs = phs_tot[axes(aperture)..., 1]
        ox, oy = size(phs) .÷ 2
        coords = hcat((axes(phs, 1) .- ox) .* ones(size(phs, 2))' |> vec,
            ones(size(phs, 1)) .* (axes(phs, 2) .- oy)' |> vec) .* vec(aperture)
        txy = coords' * vec(phs)
        gxy = coords' * coords
        cx, cy = gxy \ txy
        phase_screen[] = phs_tot .- cx .* (axes(phs_tot, 1) .- ox) .- cy .* (axes(phs_tot, 2) .- oy)'
    end

    speckle_pattern = lift(wind_s, wl_s, bw_s, exp_s, phase_screen) do wind, wl, bw, exptime, phs
        atm = SavedPhases(phs, wind_velocity=(wind, 0))
        img_spec = ImagingSpec(aperture, d, PhotonCount(Inf);
            filter=FilterSpec(wl; bandwidth=bw, npts=iszero(bw) ? 0 : ceil(Int, bw / wl * 20 + 3)),
            exposure=Exposure(exptime))
        out = simulate_images(atm, img_spec; n=1, verbose=false, savephases=false)
        return Float32.(out.images[:, :, 1])
    end

    axis_kw = (aspect=DataAspect(), xticklabelspace=24.0, yticklabelspace=30.0)
    cbar_height(ax) = lift(vp -> vp.widths[2], ax.scene.viewport)

    phase_screen_2d = lift(phs -> phs[axes(aperture)..., 1], phase_screen)
    ax_p, hm_p = heatmap(fig[3, 2], phase_screen_2d, colormap=:viridis, axis=axis_kw)
    contour!(ax_p, aperture, levels=[0.5], color=:white, linewidth=2)
    Colorbar(fig[3, 1], hm_p, label="Phase Screen (rad)", ticks=MultiplesTicks(5, pi, "π"),
        flipaxis=false, height=cbar_height(ax_p), tellheight=false, valign=:center, ticklabelspace=48.0)

    ax_i, hm_i = heatmap(fig[3, 3], speckle_pattern, colormap=:inferno, axis=axis_kw)
    Colorbar(fig[3, 4], hm_i, label="PSF (normalized)", tellheight=false,
        height=cbar_height(ax_i), valign=:center, ticklabelspace=48.0)

    Label(fig[4, :], """
    Hint: drag to zoom, left-click to pan, ctrl+click to reset view.
    """, fontsize=12)
    fig
end

end # module MakieExt
