package openai_controls

import rego.v1

default verdict := {"decision": "allow"}

verdict := {"decision": "deny", "reason": "The local prompt rule rejected this request"} if {
    input.snapshot.interception_point == "input"
    contains(lower(input.snapshot.target.content), "ignore all controls")
}

verdict := {"decision": "deny", "reason": "This local tool argument is not authorized"} if {
    input.snapshot.interception_point == "pre_tool_call"
    contains(lower(input.snapshot.target.text), "blocked-topic")
}
