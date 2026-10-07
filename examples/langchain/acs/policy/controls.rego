package controls
import rego.v1

default verdict := {"decision": "allow"}

verdict := {"decision": "deny", "reason": "tool-not-permitted"} if {
    input.snapshot.interception_point == "pre_tool_call"
    input.snapshot.tool_call.name == "blocked_echo"
}

verdict := {"decision": "deny", "reason": "output-too-long"} if {
    input.snapshot.interception_point == "post_model_call"
    is_string(input.snapshot.target.content)
    count(input.snapshot.target.content) > 800
}
