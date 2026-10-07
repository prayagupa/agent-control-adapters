package adk_models
import rego.v1

default verdict := {"decision": "allow"}

verdict := {
    "decision": "transform",
    "transform": {
        "path": "$target[1].content",
        "value": trim_space(input.snapshot.target[1].content),
    },
} if {
    input.snapshot.interception_point == "pre_model_call"
}

verdict := {
    "decision": "transform",
    "transform": {
        "path": "$target.content",
        "value": trim_space(input.snapshot.target.content),
    },
} if {
    input.snapshot.interception_point == "post_model_call"
}
