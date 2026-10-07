package adk_tools
import rego.v1

default verdict := {"decision": "allow"}

verdict := {"decision": "deny", "reason": "example:destructive_tool"} if {
    input.snapshot.interception_point == "pre_tool_call"
    input.snapshot.tool_call.name == "delete_file"
}
