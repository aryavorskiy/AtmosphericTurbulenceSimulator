using AtmosphericTurbulenceSimulator, CairoMakie, Random
Random.seed!(20260716)

const NGRID = 64

atm = SingleLayer(0.2; interpolate=:auto)
aperture = CircularAperture((NGRID, NGRID))
img_spec = ImagingSpec(aperture, 2.0, PhotonCount(Inf); filter=FilterSpec(500nm))
wf, img = dropdims.(Tuple(simulate_images(atm, img_spec; n=1, verbose=false)), dims=3)

# Crop to the brightest speckle grains
const CROP = 32
tot = sum(img)
cx = clamp(round(Int, sum(img .* (1:2NGRID)) / tot), CROP + 1, 2NGRID - CROP)
cy = clamp(round(Int, sum(img .* (1:2NGRID)') / tot), CROP + 1, 2NGRID - CROP)
speckle = img[cx-CROP+1:cx+CROP, cy-CROP+1:cy+CROP]
speckle = log.(0.02maximum(speckle) .+ speckle)
speckle .-= minimum(speckle)
speckle ./= maximum(speckle)

gridcoords(n) = range(-1, 1, length=n)
function disk_mask(n, inner=0)
    c = gridcoords(n)
    @. inner^2 <= c^2 + c'^2 <= 1.0
end
apply_mask(data, mask) = map((v, k) -> k ? v : NaN, data, mask)

const OBSTRUCTION = 0.28
wavefront_m = apply_mask(wf, disk_mask(NGRID, OBSTRUCTION))
mi, ma = extrema(speckle)
speckle_m = apply_mask(speckle .- 0.08, disk_mask(2CROP) .& (speckle .>= 0.08))

function draw_disk!(ax; radius=1, fill=:transparent, outline=:white, width=LINE_WIDTH)
    stroke = outline === nothing ? :transparent : outline
    poly!(ax, Circle(Point2f(0, 0), radius);
        color=fill, strokecolor=stroke, strokewidth=(outline === nothing ? 0 : width))
end

function draw_content!(ax, data; colormap, colorrange=Makie.automatic, interpolate=false)
    xs = range(-1, 1, length=size(data, 1))
    ys = range(-1, 1, length=size(data, 2))
    heatmap!(ax, xs, ys, data; colormap=colormap, colorrange=colorrange,
        nan_color=:transparent, interpolate=interpolate)
end

# colormaps
const LINE_WIDTH = 10
const GREEN_LIGHT = colorant"#5fbf4a"
const GREEN_DARK = colorant"#123a0d"
const RED_CMAP = cgrad([colorant"#9a1b1b", colorant"#d62828", colorant"#ff8fa3", colorant"#ffffff"])
const PURPLE_CMAP = cgrad([colorant"#3d1a52", colorant"#9558b2", colorant"#c9a7dd"])

fig = Figure(size=(760, 680), backgroundcolor=:transparent)

const LIM = 1.05
ax_star, ax_speckle, ax_wavefront = map([(1, 1:2), (2, 1), (2, 2)]) do pos
    ax = Axis(fig[pos...]; aspect=DataAspect(), backgroundcolor=:transparent)
    hidedecorations!(ax)
    hidespines!(ax)
    limits!(ax, -LIM, LIM, -LIM, LIM)
    ax
end

draw_disk!(ax_star; fill=GREEN_DARK, outline=:white)
draw_disk!(ax_speckle; fill=RED_CMAP[0.0], outline=:white)

scatter!(ax_star, [(0, 0)]; marker=:star4, markersize=2 * 0.83,
    markerspace=:data, color=GREEN_LIGHT)
draw_content!(ax_speckle, speckle_m; colormap=RED_CMAP, interpolate=false)
draw_content!(ax_wavefront, wavefront_m; colormap=PURPLE_CMAP)

nan = Point2f(NaN, NaN)
spider = Point2f[(OBSTRUCTION, 0), (1, 0), nan, (-OBSTRUCTION, 0), (-1, 0), nan,
                    (0, OBSTRUCTION), (0, 1), nan, (0, -OBSTRUCTION), (0, -1)]
lines!(ax_wavefront, spider; color=:white, linewidth=LINE_WIDTH)
draw_disk!(ax_wavefront; fill=:transparent, outline=:white)
draw_disk!(ax_wavefront; radius=OBSTRUCTION)

rowgap!(fig.layout, 0)
colgap!(fig.layout, 0)
save("logo.svg", fig, pt_per_unit=0.4)
