package controls

import rego.v1

default text_limit := {"decision": "allow"}

text_limit := {"decision": "deny", "reason": "example:text_limit"} if {
    input.snapshot.interception_point == "post_model_call"
    is_string(input.snapshot.target.content)
    count(input.snapshot.target.content) > 800
}

default tool_guard := {"decision": "allow"}

tool_guard := {"decision": "deny", "reason": "example:destructive_tool"} if {
    input.snapshot.interception_point == "pre_tool_call"
    input.snapshot.tool_call.name == "delete_file"
}
