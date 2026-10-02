# Contributing

Each `packages/<framework>` directory builds its own wheel and declares only its
own framework plus agent-hooks dependencies. Do not add empty adapter packages
or an umbrella install that pulls in every framework.

Run the commands in the root README. Before widening a dependency range, run the
real-runner integration suite against every supported version. Publish packages
independently; future release tags should identify the adapter, for example
`crewai/v0.1.0a1`. No automatic publication workflow is enabled here.

Keep public Python APIs typed, run strict pyright, and prefer small explicit
boundary mappings over private framework patches. Unit tests may stub model
I/O. Integration tests must use the real framework runner and SDK emitter.
Model-behavior examples must use a real local model and retain honest receipts.

Every new adapter must document its enforcement coverage, transforms, failure
behavior, concurrency and lifecycle ownership. Observation hooks are not
automatically enforcement hooks. Do not weaken upstream conformance assertions
to make an adapter pass.

Never commit credentials, model weights, virtual environments or private
prompts. Obtain explicit authorization before publishing packages, creating
remote repositories, or changing upstream framework repositories.
