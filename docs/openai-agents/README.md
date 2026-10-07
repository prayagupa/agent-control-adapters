# OpenAI Agents SDK Python controls

`agent-hooks-openai-agents` installs `agent_hooks_openai_agents` independently.
The supported version is **openai-agents 0.23.1**, using OpenAI Python **3.24.0**
for the examples. Do not include this project in the root CrewAI/OpenAI-2
dependency resolution.

- [Installation and copying complete folders](installation.md)
- [Supported native profile and executable transform matrix](coverage.md)
- [Local Qwen examples and mandatory evidence gates](examples.md)

The application creates an ordinary native `agents.Agent` with an explicit
public `Model`, then selects `Backend("openai-agents")` through
`with_agent_controls`. Execute with `await adapter.arun(RunRequest({"prompt": prompt}))`
on the application's existing loop. The constructor keywords are the common
portable keywords; there is no evaluator, executor bridge, implicit model
resolution, or plugin discovery.

The selected backend owns one bounded policy thread and one native run task.
It does not close the original Agent, Model, clients, tools, application loop
or Ollama service. The original native objects remain unchanged. `run` refuses
**before admission** and directs you to `arun`; that refusal does not consume the
instance. Every admitted `arun`, including invalid input, denial, SDK failure
and cancellation, consumes the single-use instance.

This is a **single-agent** profile. No multi-agent coverage is claimed. An
allowed run returns actual SDK records and an exact record eviction count.
Governance failure never returns a successful `RunResult`.
