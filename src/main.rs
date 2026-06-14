mod ai;

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context as _, Result};
use serenity::async_trait;
use serenity::model::channel::Message;
use serenity::model::gateway::Ready;
use serenity::model::id::{ChannelId, UserId};
use serenity::prelude::*;

use ai::{AiClient, ChatTurn};

/// Word that triggers the bot, e.g. `ataulfo what is the capital of France?`.
const TRIGGER: &str = "ataulfo";

/// Discord rejects messages longer than 2000 characters.
const DISCORD_MAX_LEN: usize = 2000;

/// How many recent messages to remember per channel (short-term memory).
const HISTORY_LIMIT: usize = 12;

/// Cap on how much of a replied-to message we quote into a turn, in chars.
const QUOTE_MAX_LEN: usize = 280;

/// Fallback personality if `personality.txt` is missing or empty.
const DEFAULT_PERSONALITY: &str = "You are Ataulfo, a helpful and friendly assistant.";

struct Handler {
    ai: AiClient,
    /// Path to the personality file, re-read on every message (hot reload).
    personality_path: PathBuf,
    /// The bot's own user id, learned from the `ready` event. Used to detect
    /// replies to the bot's own messages.
    bot_id: OnceLock<UserId>,
    /// Short-term memory: the last few turns per channel, oldest first.
    history: Mutex<HashMap<ChannelId, VecDeque<ChatTurn>>>,
}

impl Handler {
    /// True if this message is a reply to one of the bot's own messages.
    fn is_reply_to_self(&self, msg: &Message) -> bool {
        let Some(referenced) = &msg.referenced_message else {
            return false;
        };
        match self.bot_id.get() {
            Some(id) => referenced.author.id == *id,
            // Our own id isn't known yet; fall back to "any bot" as a best guess.
            None => referenced.author.bot,
        }
    }

    /// Append a turn to a channel's short-term memory, dropping the oldest once
    /// it exceeds `HISTORY_LIMIT`.
    fn remember(&self, channel: ChannelId, turn: ChatTurn) {
        let mut map = self.history.lock().unwrap();
        let buf = map.entry(channel).or_default();
        buf.push_back(turn);
        while buf.len() > HISTORY_LIMIT {
            buf.pop_front();
        }
    }

    /// A copy of a channel's remembered turns, oldest first.
    fn history_snapshot(&self, channel: ChannelId) -> Vec<ChatTurn> {
        let map = self.history.lock().unwrap();
        map.get(&channel)
            .map(|buf| buf.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Read the personality fresh from disk so edits apply without a restart.
    fn load_personality(&self) -> String {
        match std::fs::read_to_string(&self.personality_path) {
            Ok(text) if !text.trim().is_empty() => text,
            Ok(_) => {
                tracing::warn!(
                    "{} is empty, using default personality",
                    self.personality_path.display()
                );
                DEFAULT_PERSONALITY.to_string()
            }
            Err(e) => {
                tracing::warn!(
                    "could not read {} ({e}), using default personality",
                    self.personality_path.display()
                );
                DEFAULT_PERSONALITY.to_string()
            }
        }
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, ctx: Context, msg: Message) {
        // Never react to ourselves or other bots (avoids feedback loops). The
        // bot's own replies are recorded separately, when we send them.
        if msg.author.bot {
            return;
        }

        let text = msg.content.trim();
        if text.is_empty() {
            return;
        }

        // Record every human message into this channel's short-term memory, so
        // Ataulfo has context of the ongoing conversation (and who said what),
        // even for messages that don't trigger him.
        self.remember(msg.channel_id, ChatTurn::user(format_turn(&msg, text)));

        // Two ways to get Ataulfo to talk:
        //   1. The message starts with the "ataulfo" keyword.
        //   2. The message is a reply to one of his own messages — no keyword
        //      needed, so a conversation can flow back and forth.
        let respond = match extract_prompt(&msg.content) {
            Some(after_keyword) => {
                if after_keyword.is_empty() {
                    let _ = msg
                        .reply(&ctx.http, "¿Sí? Dime algo después de `ataulfo`.")
                        .await;
                    return;
                }
                true
            }
            None => self.is_reply_to_self(&msg),
        };
        if !respond {
            return;
        }

        // Show a typing indicator while the AI thinks.
        let _ = msg.channel_id.broadcast_typing(&ctx.http).await;

        let personality = self.load_personality();
        let history = self.history_snapshot(msg.channel_id);

        match self.ai.complete(&personality, &history).await {
            Ok(reply) => {
                // Remember his own reply so it's part of the next turn's context.
                self.remember(msg.channel_id, ChatTurn::assistant(reply.clone()));
                for chunk in split_message(&reply) {
                    if let Err(e) = msg.reply(&ctx.http, chunk).await {
                        tracing::error!("failed to send reply: {e}");
                        break;
                    }
                }
            }
            Err(e) => {
                tracing::error!("AI request failed: {e:#}");
                let _ = msg
                    .reply(&ctx.http, "Ahora mismo no puedo responder. 🤕")
                    .await;
            }
        }
    }

    async fn ready(&self, _ctx: Context, ready: Ready) {
        // Remember our own id so we can recognise replies to our messages.
        let _ = self.bot_id.set(ready.user.id);
        tracing::info!("connected as {}", ready.user.name);
    }
}

/// If `content` starts with the trigger word (case-insensitive) as a whole
/// word, return the trimmed remainder. Otherwise return `None`.
fn extract_prompt(content: &str) -> Option<&str> {
    let trimmed = content.trim_start();
    let prefix = trimmed.get(..TRIGGER.len())?;
    if !prefix.eq_ignore_ascii_case(TRIGGER) {
        return None;
    }
    let rest = &trimmed[TRIGGER.len()..];
    // Require the trigger to be a standalone word: end of string or followed
    // by whitespace/punctuation (so "ataulfos" does not match).
    match rest.chars().next() {
        None => Some(""),
        Some(c) if !c.is_alphanumeric() => Some(rest.trim()),
        Some(_) => None,
    }
}

/// Render a message as a history turn: `author: text`, with an inline note of
/// the message it replied to (if any) so the model can follow references.
fn format_turn(msg: &Message, text: &str) -> String {
    match &msg.referenced_message {
        Some(referenced) if !referenced.content.trim().is_empty() => format!(
            "{} (en respuesta a {}: \"{}\"): {}",
            msg.author.name,
            referenced.author.name,
            truncate(referenced.content.trim(), QUOTE_MAX_LEN),
            text,
        ),
        _ => format!("{}: {}", msg.author.name, text),
    }
}

/// Shorten `text` to at most `max` chars, appending `…` if it was cut.
fn truncate(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((idx, _)) => format!("{}…", &text[..idx]),
        None => text.to_string(),
    }
}

/// Split a reply into Discord-sized chunks without cutting in the middle of a
/// line where possible.
fn split_message(text: &str) -> Vec<String> {
    if text.len() <= DISCORD_MAX_LEN {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    for line in text.split_inclusive('\n') {
        if current.len() + line.len() > DISCORD_MAX_LEN {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            // A single line longer than the limit must be hard-split by chars.
            if line.len() > DISCORD_MAX_LEN {
                for ch in line.chars() {
                    if current.len() + ch.len_utf8() > DISCORD_MAX_LEN {
                        chunks.push(std::mem::take(&mut current));
                    }
                    current.push(ch);
                }
                continue;
            }
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Read a required environment variable with a helpful error message.
fn env_var(key: &str) -> Result<String> {
    std::env::var(key).with_context(|| format!("missing required env var {key}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env if present (ignored if it isn't).
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,serenity=warn".into()),
        )
        .init();

    let discord_token = env_var("DISCORD_TOKEN")?;
    let ai = AiClient::new(
        env_var("AI_BASE_URL")?,
        env_var("AI_API_KEY")?,
        env_var("AI_MODEL")?,
    );

    let personality_path: PathBuf = std::env::var("PERSONALITY_FILE")
        .unwrap_or_else(|_| "personality.txt".to_string())
        .into();

    let handler = Handler {
        ai,
        personality_path,
        bot_id: OnceLock::new(),
        history: Mutex::new(HashMap::new()),
    };

    // MESSAGE_CONTENT is a privileged intent — enable it in the Discord
    // Developer Portal (Bot → Privileged Gateway Intents) or the bot will
    // connect but see empty message content.
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    let mut client = Client::builder(&discord_token, intents)
        .event_handler(handler)
        .await
        .context("failed to create Discord client")?;

    tracing::info!("starting bot…");
    client.start().await.context("client error")?;

    Ok(())
}
