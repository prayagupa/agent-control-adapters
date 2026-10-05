package controls

import rego.v1

default tool_guard := {"decision": "allow"}

tool_guard := {"decision": "deny", "reason": "example:destructive_tool"} if {
    input.snapshot.interception_point == "pre_tool_call"
    not is_string(input.snapshot.tool_call.name)
}

tool_guard := {"decision": "deny", "reason": "example:destructive_tool"} if {
    input.snapshot.interception_point == "pre_tool_call"
    input.snapshot.tool_call.name in {"delete_file", "drop_table", "system_shutdown"}
}

default no_tools := {"decision": "allow"}

no_tools := {"decision": "deny", "reason": "example:no_tools"} if {
    input.snapshot.interception_point == "pre_tool_call"
}

within_limit(limit) if {
    input.snapshot.interception_point == "post_model_call"
    is_string(input.snapshot.target.content)
    count(input.snapshot.target.content) <= limit
}

exceeds_limit(limit) if {
    input.snapshot.interception_point == "post_model_call"
    not is_string(input.snapshot.target.content)
}

exceeds_limit(limit) if {
    input.snapshot.interception_point == "post_model_call"
    count(input.snapshot.target.content) > limit
}

default researcher_limit := {"decision": "allow"}

researcher_limit := {"decision": "allow", "reason": "example:researcher_text_limit"} if {
    within_limit(1600)
}

researcher_limit := {"decision": "deny", "reason": "example:text_limit"} if {
    exceeds_limit(1600)
}

default editor_limit := {"decision": "allow"}

editor_limit := {"decision": "allow", "reason": "example:editor_text_limit"} if {
    within_limit(800)
}

editor_limit := {"decision": "deny", "reason": "example:text_limit"} if {
    exceeds_limit(800)
}
