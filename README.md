# ternary-cadence

**Cadence detection for agent outputs.**

In music, a cadence is a harmonic progression that signals the end of a phrase — a moment of resolution, suspense, or surprise. This crate brings that concept to multi-agent systems, providing tools to detect when an agent's output stream is wrapping up, pivoting, or leaving you hanging.

## Why Cadences?

When orchestrating multiple AI agents, you need to know not just *what* they're saying, but *where they are* in their process. A cadence detector lets you:

- **Detect task completion** — know when an agent has finished strong
- **Identify pivots** — catch when an agent unexpectedly changes direction
- **Monitor suspension** — recognize when an agent is stalled or waiting
- **Profile agents** — learn which agents tend to produce which kinds of endings

## Cadence Types

### Perfect Authentic Cadence (V→I)
The strongest resolution. The agent's confidence rises to a high value on a terminal output — the task is definitively complete. Like ending a piece on a triumphant tonic chord.

```rust
use ternary_cadence::{CadenceDetector, AgentOutput, CadenceType};

let detector = CadenceDetector::default();

let outputs = vec![
    AgentOutput::new("Starting analysis...", 0.3, false),
    AgentOutput::new("Narrowing down options", 0.6, false),
    AgentOutput::new("The answer is 42", 0.95, true),
];

let cadence = detector.detect(&outputs).unwrap();
assert_eq!(cadence.cadence_type, CadenceType::PerfectAuthentic);
```

### Plagal Cadence (IV→I)
A gentle resolution. The agent finishes with stable or slightly declining confidence — the task is done, but without dramatic flair. The "Amen" cadence.

### Deceptive Cadence (V→vi)
A surprise. The output looked like it was heading for a clean finish, but then pivoted to new, non-terminal content. The agent thought it was done but discovered something new.

### Half Cadence (ending on V)
Suspension. The output ends on a non-terminal note with moderate confidence — there's more coming, the tension is unresolved. Like ending a phrase on the dominant chord.

## Core Types

### `CadenceDetector`

The main analysis engine. Maintains a sliding window of recent outputs and applies pattern-matching rules to identify cadences.

```rust
let detector = CadenceDetector::new(
    8,    // window size — how many recent outputs to consider
    0.5,  // confidence threshold — minimum confidence for detection
);

if let Some(cadence) = detector.detect(&outputs) {
    println!("Detected: {:?} (confidence: {:.2})", 
        cadence.cadence_type, cadence.confidence);
    println!("Reason: {}", cadence.reason);
}
```

### `CadenceMap`

Tracks which agents tend to produce which cadence types, building profiles over time.

```rust
let mut map = CadenceMap::new();

// Record observations
map.record("gpt-4", CadenceType::PerfectAuthentic);
map.record("gpt-4", CadenceType::PerfectAuthentic);
map.record("claude", CadenceType::Deceptive);

// Query patterns
assert_eq!(map.dominant_cadence("gpt-4"), Some(&CadenceType::PerfectAuthentic));
let similarity = map.similarity("gpt-4", "claude");
```

### `AgentOutput`

A single output from an agent, tagged with metadata for cadence analysis.

```rust
let output = AgentOutput::new("Final answer: yes", 0.95, true)
    .with_label("final_answer");
```

## Detection Algorithm

The detector works by maintaining a sliding window of the most recent N outputs and checking for four patterns in order of strength:

1. **Perfect Authentic**: Last output is terminal with rising, high confidence (≥0.8)
2. **Deceptive**: Penultimate output looked terminal, but last output pivoted to new content
3. **Plagal**: Last output is terminal with stable/slightly declining confidence (≥0.6)
4. **Half**: Last output is non-terminal with moderate confidence, or trails off with "..."

The strongest cadence above the confidence threshold is returned. If no pattern matches above threshold, `None` is returned.

## Feature Flags

No feature flags — everything is included by default.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-cadence = "0.1.0"
```

## License

MIT
