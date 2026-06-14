//! Minimal OpenAI-compatible chat client.
//!
//! Everything provider-specific lives in this file. If your custom provider
//! uses a different request/response shape, this is the only place you need to
//! touch — `build_request` and the response structs.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// How long to wait for the provider before giving up, so a hung endpoint can't
/// leave a request (and its typing indicator) pending forever.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Client for an OpenAI-compatible `/v1/chat/completions` endpoint.
#[derive(Clone)]
pub struct AiClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    /// Optional generation caps passed through to the provider.
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}

impl AiClient {
    pub fn new(
        base_url: String,
        api_key: String,
        model: String,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("failed to build HTTP client");
        Self {
            http,
            // Trim a trailing slash so we can join paths predictably.
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
            max_tokens,
            temperature,
        }
    }

    /// Send a completion. `system` is the personality and `history` is the recent
    /// conversation (oldest first), the last entry being the message to answer.
    pub async fn complete(&self, system: &str, history: &[ChatTurn]) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut messages = Vec::with_capacity(history.len() + 1);
        messages.push(ChatTurn {
            role: "system",
            content: system.to_string(),
        });
        messages.extend(history.iter().cloned());

        let body = ChatRequest {
            model: &self.model,
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("failed to reach AI provider")?;

        // Surface the provider's error body instead of a bare status code.
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("AI provider returned {status}: {text}");
        }

        let parsed: ChatResponse = resp.json().await.context("failed to parse AI response")?;

        // `content` is optional because some providers send `null` (e.g. when a
        // model returns only reasoning); treat a missing body as empty.
        let content = parsed
            .choices
            .into_iter()
            .next()
            .context("AI response contained no choices")?
            .message
            .content
            .unwrap_or_default();

        let reply = strip_reasoning(&content);
        if reply.is_empty() {
            bail!("AI returned an empty reply");
        }
        Ok(reply)
    }
}

/// Some reasoning models (DeepSeek-R1, Qwen3, etc.) embed their chain-of-thought
/// in `<think>…</think>` tags inside the reply. We only want the final answer in
/// Discord, so strip those blocks out.
///
/// Handles multiple blocks and the case where a model omits the opening `<think>`
/// and emits only a closing `</think>` after its reasoning.
fn strip_reasoning(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    // Remove every complete <think>…</think> block.
    while let Some(open) = find_ci(rest, "<think>") {
        out.push_str(&rest[..open]);
        let after_open = &rest[open + "<think>".len()..];
        match find_ci(after_open, "</think>") {
            Some(close) => rest = &after_open[close + "</think>".len()..],
            // Unclosed block (e.g. truncated response): drop the rest.
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);

    // Handle an implicit opening tag: reasoning then a lone </think> then answer.
    if let Some(close) = find_ci(&out, "</think>") {
        out = out[close + "</think>".len()..].to_string();
    }

    out.trim().to_string()
}

/// ASCII-case-insensitive substring search returning a byte index into
/// `haystack`. The needles here are pure ASCII, so the returned index always
/// lands on a char boundary.
fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    let (hb, nb) = (haystack.as_bytes(), needle.as_bytes());
    if nb.is_empty() || hb.len() < nb.len() {
        return None;
    }
    (0..=hb.len() - nb.len()).find(|&i| hb[i..i + nb.len()].eq_ignore_ascii_case(nb))
}

#[cfg(test)]
mod tests {
    use super::strip_reasoning;

    #[test]
    fn strips_complete_block() {
        assert_eq!(
            strip_reasoning("<think>let me think...</think>The answer is 42."),
            "The answer is 42."
        );
    }

    #[test]
    fn strips_multiline_and_whitespace() {
        let input = "<think>\nreasoning\nover lines\n</think>\n\nHola, ¿qué tal?";
        assert_eq!(strip_reasoning(input), "Hola, ¿qué tal?");
    }

    #[test]
    fn handles_implicit_opening_tag() {
        assert_eq!(
            strip_reasoning("internal reasoning</think>Visible reply."),
            "Visible reply."
        );
    }

    #[test]
    fn leaves_normal_text_untouched() {
        assert_eq!(
            strip_reasoning("  just a normal reply  "),
            "just a normal reply"
        );
    }

    #[test]
    fn is_case_insensitive() {
        assert_eq!(strip_reasoning("<THINK>x</THINK>answer"), "answer");
    }
}

/// One turn of conversation sent to the model.
#[derive(Serialize, Clone)]
pub struct ChatTurn {
    role: &'static str,
    content: String,
}

impl ChatTurn {
    /// A message from a human in the chat (prefix `content` with their name so
    /// the model can tell speakers apart).
    pub fn user(content: String) -> Self {
        Self {
            role: "user",
            content,
        }
    }

    /// A message the bot itself previously sent.
    pub fn assistant(content: String) -> Self {
        Self {
            role: "assistant",
            content,
        }
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatTurn>,
    // Omitted entirely when unset so providers that reject them aren't affected.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    // Optional: providers may send `null` or omit it. Reasoning-only fields such
    // as `reasoning_content` are ignored (serde drops unknown fields), since the
    // answer we display always comes from `content`.
    #[serde(default)]
    content: Option<String>,
}
