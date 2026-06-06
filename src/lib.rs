//! # ternary-cadence
//!
//! Cadence detection for agent outputs. Inspired by musical cadences — the harmonic
//! gestures that signal the end of a phrase — this crate provides tools for detecting
//! completion, transition, and suspension patterns in streams of agent outputs.

use std::collections::HashMap;

/// The type of cadence detected in an agent's output stream.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CadenceType {
    /// Strong resolution indicating a task is fully complete.
    /// Like a perfect authentic cadence in music (V→I with both chords in root position).
    PerfectAuthentic,
    /// Gentle resolution indicating a task is winding down gracefully.
    /// Like a plagal cadence (IV→I, the "Amen" cadence).
    Plagal,
    /// Unexpected resolution — the task pivoted to something new.
    /// Like a deceptive cadence (V→vi instead of V→I).
    Deceptive,
    /// Suspension — waiting for resolution, task is incomplete.
    /// Like a half cadence (ending on V, leaving tension unresolved).
    Half,
}

/// A single output from an agent, tagged with metadata for cadence analysis.
#[derive(Debug, Clone)]
pub struct AgentOutput {
    /// The text content of the output.
    pub content: String,
    /// Confidence score for the output (0.0–1.0).
    pub confidence: f64,
    /// Whether the output signals some kind of completion or continuation.
    pub is_terminal: bool,
    /// Optional label (e.g., "final_answer", "thinking", "tool_call").
    pub label: Option<String>,
}

impl AgentOutput {
    /// Create a new agent output.
    pub fn new(content: impl Into<String>, confidence: f64, is_terminal: bool) -> Self {
        Self {
            content: content.into(),
            confidence,
            is_terminal,
            label: None,
        }
    }

    /// Attach a label to this output.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self = self;
        self
    }
}

/// A detected cadence with context about what triggered it.
#[derive(Debug, Clone)]
pub struct Cadence {
    /// The type of cadence detected.
    pub cadence_type: CadenceType,
    /// Confidence of the detection (0.0–1.0).
    pub confidence: f64,
    /// Human-readable explanation of why this cadence was detected.
    pub reason: String,
}

/// A cadence pattern definition — a sequence of conditions on outputs.
pub type CadencePattern = fn(&[&AgentOutput]) -> Option<Cadence>;

/// Analyzes a stream of agent outputs for cadence patterns.
///
/// The detector maintains a sliding window of recent outputs and applies
/// pattern-matching rules to identify cadences as they emerge.
pub struct CadenceDetector {
    /// Maximum window size for cadence analysis.
    window_size: usize,
    /// Minimum confidence threshold for detection.
    threshold: f64,
}

impl Default for CadenceDetector {
    fn default() -> Self {
        Self {
            window_size: 8,
            threshold: 0.5,
        }
    }
}

impl CadenceDetector {
    /// Create a new detector with custom settings.
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            threshold,
        }
    }

    /// Analyze a sequence of agent outputs for cadence patterns.
    ///
    /// Returns the strongest cadence detected, if any.
    pub fn detect(&self, outputs: &[AgentOutput]) -> Option<Cadence> {
        if outputs.is_empty() {
            return None;
        }

        let window: Vec<&AgentOutput> = outputs
            .iter()
            .rev()
            .take(self.window_size)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        // Try each cadence pattern in order of strength
        let candidates = vec![
            self.detect_perfect_authentic(&window),
            self.detect_deceptive(&window),
            self.detect_plagal(&window),
            self.detect_half(&window),
        ];

        candidates
            .into_iter()
            .flatten()
            .filter(|c| c.confidence >= self.threshold)
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    }

    /// Perfect authentic cadence: strong finish with high confidence terminal output.
    ///
    /// Pattern: confidence rises to high values on the last 1-2 outputs,
    /// and the last output is terminal.
    fn detect_perfect_authentic(&self, window: &[&AgentOutput]) -> Option<Cadence> {
        if window.len() < 2 {
            return None;
        }

        let last = window.last()?;
        let second_last = window.get(window.len() - 2)?;

        // Strong terminal signal with high confidence
        let last_conf = last.confidence;
        let prev_conf = second_last.confidence;

        if last.is_terminal && last_conf > prev_conf && last_conf >= 0.8 {
            let conf = if last_conf >= 0.95 {
                0.98
            } else {
                last_conf
            };
            return Some(Cadence {
                cadence_type: CadenceType::PerfectAuthentic,
                confidence: conf,
                reason: format!(
                    "Strong terminal output with confidence {:.2} (rising from {:.2})",
                    last_conf, prev_conf
                ),
            });
        }

        // Also detect when both final outputs are high-confidence terminal
        if last.is_terminal
            && second_last.is_terminal
            && last_conf >= 0.85
            && prev_conf >= 0.85
        {
            return Some(Cadence {
                cadence_type: CadenceType::PerfectAuthentic,
                confidence: (last_conf + prev_conf) / 2.0,
                reason: format!(
                    "Two consecutive high-confidence terminal outputs ({:.2}, {:.2})",
                    prev_conf, last_conf
                ),
            });
        }

        None
    }

    /// Deceptive cadence: seemed like it was finishing but pivoted.
    ///
    /// Pattern: penultimate output looks terminal, but the last output
    /// is non-terminal and introduces new content.
    fn detect_deceptive(&self, window: &[&AgentOutput]) -> Option<Cadence> {
        if window.len() < 3 {
            return None;
        }

        let last = window.last()?;
        let second_last = window.get(window.len() - 2)?;
        let third_last = window.get(window.len() - 3)?;

        // Rising confidence then sudden pivot
        let rising = second_last.confidence > third_last.confidence;
        let pivot = !last.is_terminal && second_last.is_terminal;
        let new_content = last.content.len() > second_last.content.len() / 2;

        if rising && pivot && new_content {
            return Some(Cadence {
                cadence_type: CadenceType::Deceptive,
                confidence: 0.75,
                reason: String::from(
                    "Output appeared terminal but pivoted to new non-terminal content",
                ),
            });
        }

        None
    }

    /// Plagal cadence: gentle wind-down.
    ///
    /// Pattern: confidence is stable or slightly declining, output is terminal
    /// but without the strong rising pattern of a perfect cadence.
    fn detect_plagal(&self, window: &[&AgentOutput]) -> Option<Cadence> {
        if window.len() < 2 {
            return None;
        }

        let last = window.last()?;
        let second_last = window.get(window.len() - 2)?;

        if last.is_terminal && last.confidence >= 0.6 && last.confidence <= second_last.confidence + 0.1 {
            return Some(Cadence {
                cadence_type: CadenceType::Plagal,
                confidence: 0.65,
                reason: format!(
                    "Gentle terminal output with stable confidence ({:.2})",
                    last.confidence
                ),
            });
        }

        None
    }

    /// Half cadence: suspension, waiting for more.
    ///
    /// Pattern: non-terminal output with moderate confidence,
    /// or output that raises questions / ends mid-thought.
    fn detect_half(&self, window: &[&AgentOutput]) -> Option<Cadence> {
        if window.is_empty() {
            return None;
        }

        let last = window.last()?;

        if !last.is_terminal && last.confidence < 0.7 {
            // Check if confidence was rising (building tension)
            let tension = if window.len() >= 2 {
                let prev = window.get(window.len() - 2).unwrap();
                prev.confidence < last.confidence
            } else {
                true
            };

            return Some(Cadence {
                cadence_type: CadenceType::Half,
                confidence: if tension { 0.7 } else { 0.55 },
                reason: String::from(
                    "Non-terminal output with moderate confidence — awaiting resolution",
                ),
            });
        }

        // Also detect trailing off pattern
        if !last.is_terminal && last.content.ends_with("...") {
            return Some(Cadence {
                cadence_type: CadenceType::Half,
                confidence: 0.8,
                reason: String::from("Output trails off with ellipsis — suspended"),
            });
        }

        None
    }
}

/// Maps agents to their typical cadence patterns.
///
/// Tracks which cadence types each agent tends to produce, building a profile
/// over time that can be used for prediction and analysis.
pub struct CadenceMap {
    /// Agent ID → cadence type → count of occurrences.
    agent_cadences: HashMap<String, HashMap<CadenceType, usize>>,
    /// Agent ID → total observations.
    agent_total: HashMap<String, usize>,
}

impl Default for CadenceMap {
    fn default() -> Self {
        Self::new()
    }
}

impl CadenceMap {
    /// Create an empty cadence map.
    pub fn new() -> Self {
        Self {
            agent_cadences: HashMap::new(),
            agent_total: HashMap::new(),
        }
    }

    /// Record a cadence observation for an agent.
    pub fn record(&mut self, agent_id: impl Into<String>, cadence_type: CadenceType) {
        let id = agent_id.into();
        *self.agent_total.entry(id.clone()).or_insert(0) += 1;
        let counts = self.agent_cadences.entry(id).or_insert_with(HashMap::new);
        *counts.entry(cadence_type).or_insert(0) += 1;
    }

    /// Get the most common cadence type for an agent.
    pub fn dominant_cadence(&self, agent_id: &str) -> Option<&CadenceType> {
        let counts = self.agent_cadences.get(agent_id)?;
        counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(ct, _)| ct)
    }

    /// Get the frequency of a specific cadence type for an agent.
    pub fn frequency(&self, agent_id: &str, cadence_type: &CadenceType) -> f64 {
        let total = self.agent_total.get(agent_id).copied().unwrap_or(0);
        if total == 0 {
            return 0.0;
        }
        let count = self
            .agent_cadences
            .get(agent_id)
            .and_then(|m| m.get(cadence_type))
            .copied()
            .unwrap_or(0);
        count as f64 / total as f64
    }

    /// Get all agents tracked by this map.
    pub fn agents(&self) -> Vec<&str> {
        self.agent_total.keys().map(|s| s.as_str()).collect()
    }

    /// Get the count of observations for an agent.
    pub fn observation_count(&self, agent_id: &str) -> usize {
        self.agent_total.get(agent_id).copied().unwrap_or(0)
    }

    /// Compare two agents' cadence profiles using cosine similarity.
    pub fn similarity(&self, agent_a: &str, agent_b: &str) -> f64 {
        let all_types = [
            CadenceType::PerfectAuthentic,
            CadenceType::Plagal,
            CadenceType::Deceptive,
            CadenceType::Half,
        ];

        let vec_a: Vec<f64> = all_types
            .iter()
            .map(|ct| self.frequency(agent_a, ct))
            .collect();
        let vec_b: Vec<f64> = all_types
            .iter()
            .map(|ct| self.frequency(agent_b, ct))
            .collect();

        let dot: f64 = vec_a.iter().zip(&vec_b).map(|(a, b)| a * b).sum();
        let mag_a: f64 = vec_a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = vec_b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_perfect_cadence() {
        let detector = CadenceDetector::default();

        let outputs = vec![
            AgentOutput::new("Thinking about the problem...", 0.3, false),
            AgentOutput::new("Analyzing options", 0.5, false),
            AgentOutput::new("Almost there", 0.7, false),
            AgentOutput::new("The answer is 42", 0.95, true),
        ];

        let cadence = detector.detect(&outputs).expect("Should detect a cadence");
        assert_eq!(cadence.cadence_type, CadenceType::PerfectAuthentic);
        assert!(cadence.confidence >= 0.8);
    }

    #[test]
    fn test_detect_perfect_cadence_two_terminals() {
        let detector = CadenceDetector::default();

        let outputs = vec![
            AgentOutput::new("Working on it", 0.4, false),
            AgentOutput::new("Done with step 1", 0.9, true),
            AgentOutput::new("Done with step 2", 0.95, true),
        ];

        let cadence = detector.detect(&outputs).expect("Should detect cadence");
        assert_eq!(cadence.cadence_type, CadenceType::PerfectAuthentic);
    }

    #[test]
    fn test_detect_half_cadence() {
        let detector = CadenceDetector::default();

        let outputs = vec![
            AgentOutput::new("Starting analysis", 0.5, false),
            AgentOutput::new("Still working...", 0.4, false),
        ];

        let cadence = detector.detect(&outputs).expect("Should detect a cadence");
        assert_eq!(cadence.cadence_type, CadenceType::Half);
    }

    #[test]
    fn test_detect_half_cadence_ellipsis() {
        let detector = CadenceDetector::default();

        let outputs = vec![
            AgentOutput::new("Let me think about this...", 0.9, false),
        ];

        let cadence = detector.detect(&outputs).expect("Should detect half cadence");
        assert_eq!(cadence.cadence_type, CadenceType::Half);
    }

    #[test]
    fn test_detect_deceptive_cadence() {
        let detector = CadenceDetector::default();

        let outputs = vec![
            AgentOutput::new("Starting", 0.3, false),
            AgentOutput::new("Almost done", 0.8, true),
            AgentOutput::new("Actually, I need to reconsider this entirely because...", 0.5, false),
        ];

        let cadence = detector.detect(&outputs).expect("Should detect a cadence");
        assert_eq!(cadence.cadence_type, CadenceType::Deceptive);
    }

    #[test]
    fn test_detect_plagal_cadence() {
        let detector = CadenceDetector::default();

        let outputs = vec![
            AgentOutput::new("Working steadily", 0.7, false),
            AgentOutput::new("Done, here's the result", 0.7, true),
        ];

        let cadence = detector.detect(&outputs).expect("Should detect a cadence");
        assert_eq!(cadence.cadence_type, CadenceType::Plagal);
    }

    #[test]
    fn test_no_cadence_on_random_output() {
        let detector = CadenceDetector::new(8, 0.9); // high threshold

        let outputs = vec![
            AgentOutput::new("Random thought", 0.3, false),
            AgentOutput::new("Another thought", 0.35, false),
            AgentOutput::new("Yet another", 0.4, false),
        ];

        // With high threshold, these weak signals shouldn't produce confident detection
        let result = detector.detect(&outputs);
        // Might detect a half cadence at low confidence, but with threshold 0.9 it should be None
        assert!(result.is_none() || result.as_ref().unwrap().confidence < 0.9);
    }

    #[test]
    fn test_empty_outputs() {
        let detector = CadenceDetector::default();
        assert!(detector.detect(&[]).is_none());
    }

    #[test]
    fn test_single_output() {
        let detector = CadenceDetector::default();
        let outputs = vec![AgentOutput::new("Hello", 0.5, false)];
        // Should get a half cadence from single non-terminal output
        let cadence = detector.detect(&outputs);
        assert!(cadence.is_some());
    }

    #[test]
    fn test_cadence_map_tracks_agents() {
        let mut map = CadenceMap::new();

        map.record("agent-a", CadenceType::PerfectAuthentic);
        map.record("agent-a", CadenceType::PerfectAuthentic);
        map.record("agent-a", CadenceType::Plagal);
        map.record("agent-b", CadenceType::Deceptive);
        map.record("agent-b", CadenceType::Half);

        assert_eq!(map.dominant_cadence("agent-a"), Some(&CadenceType::PerfectAuthentic));
        assert_eq!(map.observation_count("agent-a"), 3);
        assert_eq!(map.observation_count("agent-b"), 2);
        assert_eq!(map.observation_count("agent-c"), 0);
    }

    #[test]
    fn test_cadence_map_frequency() {
        let mut map = CadenceMap::new();

        map.record("agent-a", CadenceType::PerfectAuthentic);
        map.record("agent-a", CadenceType::PerfectAuthentic);
        map.record("agent-a", CadenceType::Half);

        let freq = map.frequency("agent-a", &CadenceType::PerfectAuthentic);
        assert!((freq - 0.6667).abs() < 0.01);

        let freq_half = map.frequency("agent-a", &CadenceType::Half);
        assert!((freq_half - 0.3333).abs() < 0.01);

        let freq_deceptive = map.frequency("agent-a", &CadenceType::Deceptive);
        assert_eq!(freq_deceptive, 0.0);
    }

    #[test]
    fn test_cadence_map_similarity() {
        let mut map = CadenceMap::new();

        // Agent A: mostly perfect cadences
        for _ in 0..8 {
            map.record("agent-a", CadenceType::PerfectAuthentic);
        }
        map.record("agent-a", CadenceType::Half);

        // Agent B: similar pattern
        for _ in 0..7 {
            map.record("agent-b", CadenceType::PerfectAuthentic);
        }
        map.record("agent-b", CadenceType::Plagal);

        // Agent C: completely different
        for _ in 0..5 {
            map.record("agent-c", CadenceType::Deceptive);
        }

        let sim_ab = map.similarity("agent-a", "agent-b");
        let sim_ac = map.similarity("agent-a", "agent-c");

        assert!(sim_ab > sim_ac, "Similar agents should have higher similarity");
        assert!(sim_ab > 0.9, "Very similar agents: got {}", sim_ab);
    }

    #[test]
    fn test_cadence_map_agents_list() {
        let mut map = CadenceMap::new();
        map.record("alpha", CadenceType::Half);
        map.record("beta", CadenceType::Plagal);

        let agents = map.agents();
        assert_eq!(agents.len(), 2);
        assert!(agents.contains(&"alpha"));
        assert!(agents.contains(&"beta"));
    }

    #[test]
    fn test_agent_output_builder() {
        let output = AgentOutput::new("test", 0.8, true).with_label("final");
        assert_eq!(output.content, "test");
        assert_eq!(output.confidence, 0.8);
        assert!(output.is_terminal);
        assert_eq!(output.label.as_deref(), Some("final"));
    }

    #[test]
    fn test_detector_custom_settings() {
        let detector = CadenceDetector::new(3, 0.3);
        let outputs = vec![
            AgentOutput::new("Thinking", 0.3, false),
            AgentOutput::new("Done", 0.9, true),
        ];
        // With low threshold, strong terminal should trigger perfect authentic
        let cadence = detector.detect(&outputs).expect("should detect cadence");
        assert_eq!(cadence.cadence_type, CadenceType::PerfectAuthentic);
    }
}
