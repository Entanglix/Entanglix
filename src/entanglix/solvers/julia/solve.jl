# Julia entry for the pinned Piccolo ProblemSpec runner. No new process group.
using Piccolo, TOML

problem, output, options_path = ARGS
options = TOML.parsefile(options_path)
spec = Piccolo.Specs.parse_spec(read(problem, String))
kwargs = Dict{Symbol,Any}(Symbol(key) => value for (key, value) in options)
result = Piccolo.Specs.solve_spec(problem; run_dir=output, kwargs...)
pulse_path = only(filter(path -> endswith(path, ".jld2"), readdir(output; join=true)))
pulse = load_pulse(pulse_path)
times = collect(pulse.controls.t)
# Keep all N stored knots, including the terminal value used in differences.
amplitudes = hcat([pulse(t) for t in times]...)
ignored = String[]
if spec.pulse.init != :random && haskey(TOML.parsefile(problem)["pulse"], "seed")
    push!(ignored, "pulse.seed")
end
initialization = Dict{String,Any}("kind" => string(spec.pulse.init))
if spec.pulse.init == :random
    initialization["effective_seed"] = spec.pulse.seed
end
open(joinpath(output, "neutral-pulse.toml"), "w") do io
    TOML.print(io, Dict(
        "times_ns" => times,
        "amplitudes_ghz" => [collect(row) for row in eachrow(amplitudes)],
        "effective_options" => Dict(
            "max_iter" => get(options, "max_iter", spec.solver.max_iter),
            "tol" => get(options, "tol", IpoptOptions().tol),
        ),
        "initialization" => initialization,
        "ignored_guidance_fields" => ignored,
    ))
end
