//! The MCP wire shapes: tool definitions for `tools/list` and result framing
//! for `tools/call`.
//!
//! Tool handlers in this crate return plain JSON. This module is the one place
//! that turns them into what an MCP client expects, so a handler never has to
//! know about the protocol and the protocol never has to know about a handler.

use serde_json::{json, Map, Value};

/// One argument of a tool.
struct Arg {
    name: &'static str,
    ty: &'static str,
    required: bool,
    doc: &'static str,
}

const fn req(name: &'static str, ty: &'static str, doc: &'static str) -> Arg {
    Arg { name, ty, required: true, doc }
}

const fn opt(name: &'static str, ty: &'static str, doc: &'static str) -> Arg {
    Arg { name, ty, required: false, doc }
}

/// One tool: its name, what it does, and its arguments.
struct Tool {
    name: &'static str,
    doc: &'static str,
    args: &'static [Arg],
}

const ID: Arg = req("id", "string", "Drawer id, as returned by add_drawer, search or list_drawers.");
const LIMIT: Arg = opt("limit", "integer", "Maximum number of results.");

const TOOLS: &[Tool] = &[
    Tool { name: "status", doc: "Counts of drawers, chunks and knowledge-graph facts, and whether vector search is fully indexed.", args: &[] },
    Tool { name: "membrane_status", doc: "Which tiers are open, private or locked. Never reveals contents.", args: &[] },
    Tool { name: "list_wings", doc: "List the wings (top-level areas) of the palace.", args: &[] },
    Tool { name: "list_rooms", doc: "List the rooms in a wing.", args: &[req("wing", "string", "Wing name.")] },
    Tool {
        name: "list_drawers",
        doc: "List drawers, optionally within one wing or room, oldest first. Returns a page and the total count.",
        args: &[
            opt("wing", "string", "Only this wing."),
            opt("room", "string", "Only this room."),
            opt("limit", "integer", "Page size (default 50)."),
            opt("offset", "integer", "Number of drawers to skip (default 0)."),
        ],
    },
    Tool { name: "get_drawer", doc: "Fetch one drawer by id.", args: &[ID] },
    Tool {
        name: "add_drawer",
        doc: "File a new memory. A drawer is a piece of text stored in a room of a wing; choose short, stable names for both.",
        args: &[
            req("wing", "string", "Top-level area, for example a project or a person."),
            req("room", "string", "Topic within the wing."),
            req("content", "string", "The text to remember."),
            opt("provenance", "string", "Where this came from (default \"mcp\")."),
            opt("name", "string", "Optional name, unique within the room, so the drawer can be addressed as wing/room:name."),
        ],
    },
    Tool { name: "update_drawer", doc: "Replace a drawer's text.", args: &[ID, req("content", "string", "The new text.")] },
    Tool { name: "delete_drawer", doc: "Delete a drawer. It is tombstoned, not erased, so the deletion can sync.", args: &[ID] },
    Tool {
        name: "check_duplicate",
        doc: "Find existing drawers similar to some text, to avoid filing the same memory twice.",
        args: &[req("content", "string", "Text to check."), opt("limit", "integer", "Maximum matches (default 5).")],
    },
    Tool {
        name: "search",
        doc: "Search the palace by meaning and by keyword. Returns the best-matching drawers.",
        args: &[req("query", "string", "What to look for, in plain words."), opt("limit", "integer", "Maximum results (default 10).")],
    },
    Tool {
        name: "diary_read",
        doc: "Read an agent's most recent diary entries, newest first.",
        args: &[req("agent", "string", "Agent name."), opt("last_n", "integer", "How many entries (default 10).")],
    },
    Tool {
        name: "diary_write",
        doc: "Write an entry in an agent's diary.",
        args: &[
            req("agent", "string", "Agent name."),
            req("entry", "string", "The entry text."),
            opt("topic", "string", "Optional name for the entry."),
        ],
    },
    Tool {
        name: "create_tunnel",
        doc: "Link two drawers with a named relation, for example \"supports\" or \"follows\".",
        args: &[
            req("from", "string", "Source drawer id."),
            req("to", "string", "Target drawer id."),
            req("relation", "string", "Relation name."),
            opt("confidence", "number", "0 to 1 (default 1)."),
        ],
    },
    Tool { name: "list_tunnels", doc: "List the links touching a drawer.", args: &[req("drawer", "string", "Drawer id.")] },
    Tool { name: "find_tunnels", doc: "List all links with a given relation.", args: &[req("relation", "string", "Relation name.")] },
    Tool {
        name: "follow_tunnels",
        doc: "Follow links of one relation outward from a drawer.",
        args: &[
            req("drawer", "string", "Starting drawer id."),
            req("relation", "string", "Relation name."),
            opt("depth", "integer", "How many hops (default 1)."),
        ],
    },
    Tool { name: "delete_tunnel", doc: "Remove a link.", args: &[req("id", "string", "Tunnel id.")] },
    Tool {
        name: "traverse",
        doc: "Walk all links outward from a drawer and return what is reachable.",
        args: &[req("drawer", "string", "Starting drawer id."), opt("depth", "integer", "How many hops (default 2).")],
    },
    Tool {
        name: "kg_add",
        doc: "Record a fact in the knowledge graph as subject, predicate, object, optionally with the period it holds for.",
        args: &[
            req("subject", "string", "Entity the fact is about."),
            req("predicate", "string", "Relation, for example \"works_at\"."),
            req("object", "string", "Value or other entity."),
            opt("valid_from", "string", "RFC 3339 time the fact became true."),
            opt("valid_to", "string", "RFC 3339 time the fact stopped being true."),
            opt("confidence", "number", "0 to 1 (default 1)."),
        ],
    },
    Tool {
        name: "kg_invalidate",
        doc: "Mark a fact as no longer true from a given time. The history is kept.",
        args: &[
            req("subject", "string", "Entity."),
            req("predicate", "string", "Relation."),
            req("object", "string", "Value."),
            req("ended", "string", "RFC 3339 time the fact stopped being true."),
        ],
    },
    Tool {
        name: "kg_query",
        doc: "Facts about an entity, optionally as they stood at a past time.",
        args: &[
            req("entity", "string", "Entity name."),
            opt("as_of", "string", "RFC 3339 time; facts valid then (default now)."),
            opt("direction", "string", "\"outgoing\", \"incoming\" or \"both\" (default)."),
        ],
    },
    Tool { name: "kg_stats", doc: "Counts of entities and facts in the knowledge graph.", args: &[] },
    Tool { name: "kg_timeline", doc: "Every fact about an entity in time order, including ended ones.", args: &[req("entity", "string", "Entity name.")] },
    Tool { name: "graph_stats", doc: "Counts and shape of the drawer link graph.", args: &[] },
    Tool { name: "get_taxonomy", doc: "Every wing with its rooms and drawer counts.", args: &[] },
    Tool { name: "get_aaak_spec", doc: "The AAAK compressed-note dialect, for agents that write in it (MemPalace compatibility).", args: &[] },
    Tool { name: "memories_filed_away", doc: "The most recently filed drawers.", args: &[LIMIT] },
    Tool { name: "hook_settings", doc: "MemPalace compatibility. Simonides has no hooks; returns an empty list.", args: &[] },
    Tool { name: "reconnect", doc: "Re-open the store, for example after it was restored from a backup.", args: &[] },
    Tool {
        name: "rebuild",
        doc: "Start indexing in the background: embeds drawers for vector search and refreshes derived indexes. Returns at once; poll rebuild_status.",
        args: &[opt("full_kg", "boolean", "Also rebuild the knowledge-graph projection (default false).")],
    },
    Tool { name: "rebuild_status", doc: "Progress of the background rebuild.", args: &[] },
];

/// The `tools` array for a `tools/list` response.
pub fn tool_definitions() -> Value {
    Value::Array(
        TOOLS
            .iter()
            .map(|t| {
                let mut props = Map::new();
                for a in t.args {
                    props.insert(a.name.to_string(), json!({"type": a.ty, "description": a.doc}));
                }
                let required: Vec<&str> = t.args.iter().filter(|a| a.required).map(|a| a.name).collect();
                json!({
                    "name": t.name,
                    "description": t.doc,
                    "inputSchema": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                        "additionalProperties": true
                    }
                })
            })
            .collect(),
    )
}

/// Frame a handler's outcome as a `tools/call` result.
///
/// A tool that fails returns a normal result with `isError: true`, as MCP
/// asks, so the model sees the message and can correct itself. Protocol-level
/// errors (bad JSON-RPC, unknown method) are reported elsewhere.
pub fn call_result(outcome: Result<Value, String>) -> Value {
    match outcome {
        Ok(v) => {
            let text = serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string());
            json!({"content": [{"type": "text", "text": text}], "isError": false})
        }
        Err(e) => json!({"content": [{"type": "text", "text": e}], "isError": true}),
    }
}

/// MCP protocol revisions this server speaks, newest first.
pub const SUPPORTED_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// The version to answer `initialize` with: the client's if supported, else our newest.
pub fn negotiate_version(requested: &str) -> &'static str {
    SUPPORTED_VERSIONS.iter().copied().find(|v| *v == requested).unwrap_or(SUPPORTED_VERSIONS[0])
}

/// Short guidance returned at `initialize`, shown to the model by clients that support it.
pub const INSTRUCTIONS: &str = "Simonides is your long-term memory. File what is worth keeping with add_drawer \
(choose a wing for the area and a room for the topic), find it again with search, and keep a running journal \
with diary_write and diary_read. Record facts that change over time with kg_add and kg_invalidate.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_definition_is_a_valid_mcp_tool() {
        let defs = tool_definitions();
        let defs = defs.as_array().unwrap();
        assert_eq!(defs.len(), TOOLS.len());
        let mut seen = std::collections::HashSet::new();
        for d in defs {
            let name = d["name"].as_str().unwrap();
            assert!(seen.insert(name.to_string()), "duplicate tool {name}");
            assert!(!d["description"].as_str().unwrap().is_empty(), "{name} needs a description");
            let schema = &d["inputSchema"];
            assert_eq!(schema["type"], "object", "{name}");
            for r in schema["required"].as_array().unwrap() {
                assert!(schema["properties"].get(r.as_str().unwrap()).is_some(), "{name}: required {r} not in properties");
            }
            for (_, p) in schema["properties"].as_object().unwrap() {
                let ty = p["type"].as_str().unwrap();
                assert!(["string", "integer", "number", "boolean"].contains(&ty), "{name}: odd type {ty}");
            }
        }
    }

    #[test]
    fn version_negotiation_never_echoes_an_unknown_version() {
        assert_eq!(negotiate_version("2025-03-26"), "2025-03-26");
        assert_eq!(negotiate_version("1999-01-01"), SUPPORTED_VERSIONS[0]);
        assert_eq!(negotiate_version(""), SUPPORTED_VERSIONS[0]);
    }

    #[test]
    fn results_are_framed_as_text_content_and_errors_are_flagged() {
        let ok = call_result(Ok(json!({"a": 1})));
        assert_eq!(ok["isError"], false);
        assert_eq!(ok["content"][0]["type"], "text");
        assert!(ok["content"][0]["text"].as_str().unwrap().contains("\"a\": 1"));
        let err = call_result(Err("missing required string arg: wing".into()));
        assert_eq!(err["isError"], true);
        assert_eq!(err["content"][0]["text"], "missing required string arg: wing");
    }
}