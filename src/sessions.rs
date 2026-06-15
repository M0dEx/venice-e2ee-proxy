//! Per-agent-session lifecycle and attestation/model-key state.
//!
//! Sessions are keyed by `model_id:agent_session_id`, with identifiers resolved
//! from the configured session-id header, request metadata, or configured
//! fallback behavior. Expired sessions are discarded before reuse so E2EE and
//! attestation state can be refreshed safely.

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};

use axum::http::{HeaderMap, HeaderName};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

use crate::config::{SessionConfig, SessionFallbackScope};

/// Scope of the resolved session identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionScope {
    /// Stable caller-provided or configured agent-level fallback session.
    Agent,
    /// Generated request-level fallback session.
    Request,
}

impl SessionScope {
    /// Returns the lowercase header value for this session scope.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Request => "request",
        }
    }
}

impl fmt::Display for SessionScope {
    /// Formats the session scope using its lowercase header value.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Reason a previous session was no longer eligible for reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionExpirationReason {
    IdleTtl,
    MaxTtl,
    MaxRequests,
}

/// Immutable snapshot returned to callers after lookup or creation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionContext {
    pub session_key: String,
    pub model_id: String,
    pub agent_session_id: String,
    pub scope: SessionScope,
    pub created_at: SystemTime,
    pub last_used_at: SystemTime,
    pub expires_at: SystemTime,
    pub request_count: u64,
    pub attested_model_public_key: Option<String>,
    pub attestation_report: Option<Value>,
    pub verified_at: Option<SystemTime>,
}

/// Result of resolving a request into a valid session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionResolution {
    pub session: SessionContext,
    pub created: bool,
    pub replaced_expired: Option<SessionExpirationReason>,
}

/// Attestation/model-key state cached with a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestedModelState {
    pub model_public_key: String,
    pub attestation_report: Value,
    pub verified_at: SystemTime,
}

/// Request data needed to resolve a Venice proxy session.
#[derive(Debug, Clone, Copy)]
pub struct SessionRequest<'a> {
    pub model_id: &'a str,
    pub headers: &'a HeaderMap,
    pub body: Option<&'a Value>,
}

impl<'a> SessionRequest<'a> {
    /// Creates a session-resolution request from the model id and HTTP headers.
    pub fn new(model_id: &'a str, headers: &'a HeaderMap) -> Self {
        Self {
            model_id,
            headers,
            body: None,
        }
    }

    /// Adds the JSON request body so metadata can contribute session identifiers.
    pub fn with_body(mut self, body: &'a Value) -> Self {
        self.body = Some(body);
        self
    }
}

/// In-memory session manager that resolves request identifiers and tracks session expiry.
#[derive(Debug, Clone)]
pub struct SessionManager {
    config: SessionConfig,
    sessions: Arc<Mutex<HashMap<String, SessionContext>>>,
    agent_fallback_session_id: Arc<str>,
}

impl SessionManager {
    /// Creates an empty session manager using the supplied session policy.
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            agent_fallback_session_id: Arc::from(Uuid::new_v4().to_string()),
        }
    }

    /// Resolves, creates, or refreshes the session for a request using the
    /// current wall-clock time.
    pub fn get_or_create(
        &self,
        request: SessionRequest<'_>,
    ) -> Result<SessionResolution, SessionError> {
        self.get_or_create_at(request, SystemTime::now())
    }

    /// Testable variant of [`Self::get_or_create`] with an injected clock.
    pub fn get_or_create_at(
        &self,
        request: SessionRequest<'_>,
        now: SystemTime,
    ) -> Result<SessionResolution, SessionError> {
        if request.model_id.trim().is_empty() {
            return Err(SessionError::InvalidModelId);
        }

        let resolved = self.resolve_identifier(request)?;
        let session_key = session_key(request.model_id, &resolved.agent_session_id);
        let mut sessions = self.lock_sessions();
        let replaced_expired = match sessions.get(&session_key) {
            Some(existing) => self.expiration_reason(existing, now),
            None => None,
        };

        if replaced_expired.is_some() {
            sessions.remove(&session_key);
        }

        if let Some(existing) = sessions.get_mut(&session_key) {
            existing.request_count += 1;
            existing.last_used_at = now;
            return Ok(SessionResolution {
                session: existing.clone(),
                created: false,
                replaced_expired: None,
            });
        }

        let context = SessionContext::new(
            request.model_id,
            resolved.agent_session_id,
            resolved.scope,
            now,
            &self.config,
        );
        sessions.insert(session_key, context.clone());

        Ok(SessionResolution {
            session: context,
            created: true,
            replaced_expired,
        })
    }

    /// Stores verified attestation/model-key state in an existing, unexpired session.
    pub fn set_attested_model_state(
        &self,
        session_key: &str,
        state: AttestedModelState,
    ) -> Result<SessionContext, SessionError> {
        self.set_attested_model_state_at(session_key, state, SystemTime::now())
    }

    /// Testable variant of [`Self::set_attested_model_state`] with an injected clock.
    pub fn set_attested_model_state_at(
        &self,
        session_key: &str,
        state: AttestedModelState,
        now: SystemTime,
    ) -> Result<SessionContext, SessionError> {
        let mut sessions = self.lock_sessions();
        let expired = sessions
            .get(session_key)
            .and_then(|session| self.expiration_reason(session, now));

        if let Some(reason) = expired {
            sessions.remove(session_key);
            return Err(SessionError::SessionExpired { reason });
        }

        let session =
            sessions
                .get_mut(session_key)
                .ok_or_else(|| SessionError::SessionNotFound {
                    session_key: session_key.to_owned(),
                })?;
        session.attested_model_public_key = Some(state.model_public_key);
        session.attestation_report = Some(state.attestation_report);
        session.verified_at = Some(state.verified_at);

        Ok(session.clone())
    }

    /// Removes expired sessions and returns the number removed.
    pub fn cleanup_expired(&self) -> usize {
        self.cleanup_expired_at(SystemTime::now())
    }

    /// Testable cleanup variant with an injected clock.
    pub fn cleanup_expired_at(&self, now: SystemTime) -> usize {
        let mut sessions = self.lock_sessions();
        let before = sessions.len();
        sessions.retain(|_, session| self.expiration_reason(session, now).is_none());
        before - sessions.len()
    }

    /// Returns the number of sessions currently stored by the manager.
    pub fn len(&self) -> usize {
        self.lock_sessions().len()
    }

    /// Returns whether the manager currently stores no sessions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resolves the request's session identifier from headers, metadata, or configured fallback behavior.
    fn resolve_identifier(
        &self,
        request: SessionRequest<'_>,
    ) -> Result<ResolvedSessionIdentifier, SessionError> {
        if let Some(value) = self.explicit_identifier(&request)? {
            return Ok(ResolvedSessionIdentifier::agent(value));
        }

        match self.config.fallback_scope {
            SessionFallbackScope::Agent => Ok(ResolvedSessionIdentifier::agent(
                self.agent_fallback_session_id.to_string(),
            )),
            SessionFallbackScope::Request => Ok(ResolvedSessionIdentifier {
                agent_session_id: Uuid::new_v4().to_string(),
                scope: SessionScope::Request,
            }),
            SessionFallbackScope::Disabled => Err(SessionError::MissingSessionIdentifier),
        }
    }

    /// Returns the first caller-provided session identifier in precedence order.
    fn explicit_identifier(
        &self,
        request: &SessionRequest<'_>,
    ) -> Result<Option<String>, SessionError> {
        if let Some(value) =
            header_identifier(request.headers, &self.config.headers.incoming_session_id)?
        {
            return Ok(Some(value));
        }

        Ok(metadata_identifier(request.body, "session_id"))
    }

    /// Returns why a session is expired at `now`, or `None` when it remains reusable.
    fn expiration_reason(
        &self,
        session: &SessionContext,
        now: SystemTime,
    ) -> Option<SessionExpirationReason> {
        if session.request_count >= self.config.max_requests {
            return Some(SessionExpirationReason::MaxRequests);
        }

        if now >= session.expires_at {
            return Some(SessionExpirationReason::MaxTtl);
        }

        if elapsed_since(session.last_used_at, now) >= self.config.idle_ttl {
            return Some(SessionExpirationReason::IdleTtl);
        }

        None
    }

    /// Locks the session map and recovers the map if a previous holder panicked.
    fn lock_sessions(&self) -> std::sync::MutexGuard<'_, HashMap<String, SessionContext>> {
        self.sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Session identifier resolved from a request before it is combined with the model id.
#[derive(Debug, Clone)]
struct ResolvedSessionIdentifier {
    agent_session_id: String,
    scope: SessionScope,
}

impl ResolvedSessionIdentifier {
    /// Creates an agent-scoped resolved identifier from a caller-provided id.
    fn agent(agent_session_id: String) -> Self {
        Self {
            agent_session_id,
            scope: SessionScope::Agent,
        }
    }
}

impl SessionContext {
    /// Creates a new session snapshot for a model/id pair at the supplied time.
    fn new(
        model_id: &str,
        agent_session_id: String,
        scope: SessionScope,
        now: SystemTime,
        config: &SessionConfig,
    ) -> Self {
        let session_key = session_key(model_id, &agent_session_id);
        Self {
            session_key,
            model_id: model_id.to_owned(),
            agent_session_id,
            scope,
            created_at: now,
            last_used_at: now,
            expires_at: now + config.max_ttl,
            request_count: 1,
            attested_model_public_key: None,
            attestation_report: None,
            verified_at: None,
        }
    }
}

/// Errors returned while resolving, reusing, or updating proxy sessions.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SessionError {
    #[error("request model id must not be empty")]
    InvalidModelId,
    #[error("request does not include a session identifier and session fallback is disabled")]
    MissingSessionIdentifier,
    #[error("configured session header name {header:?} is invalid")]
    InvalidHeaderName { header: String },
    #[error("session header {header} contains non-UTF-8 data")]
    InvalidHeaderValue { header: String },
    #[error("session {session_key} was not found")]
    SessionNotFound { session_key: String },
    #[error("session expired before attestation state could be stored: {reason:?}")]
    SessionExpired { reason: SessionExpirationReason },
}

/// Reads a configured header name from request headers and returns a trimmed non-empty value.
fn header_identifier(
    headers: &HeaderMap,
    configured_name: &str,
) -> Result<Option<String>, SessionError> {
    let name = HeaderName::from_bytes(configured_name.as_bytes()).map_err(|_| {
        SessionError::InvalidHeaderName {
            header: configured_name.to_owned(),
        }
    })?;

    let Some(value) = headers.get(&name) else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .map_err(|_| SessionError::InvalidHeaderValue {
            header: configured_name.to_owned(),
        })?;
    Ok(non_empty_string(value))
}

/// Reads a non-empty session identifier from `metadata[key]` in the request body.
fn metadata_identifier(body: Option<&Value>, key: &str) -> Option<String> {
    body.and_then(|body| body.get("metadata"))
        .and_then(|metadata| metadata.get(key))
        .and_then(Value::as_str)
        .and_then(non_empty_string)
}

/// Returns a trimmed owned string when the input contains non-whitespace text.
fn non_empty_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// Builds the storage key that scopes a session identifier to a model id.
fn session_key(model_id: &str, agent_session_id: &str) -> String {
    format!("{model_id}:{agent_session_id}")
}

/// Returns elapsed wall-clock time between two instants, or zero if `now` is earlier.
fn elapsed_since(start: SystemTime, now: SystemTime) -> Duration {
    now.duration_since(start).unwrap_or(Duration::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use serde_json::json;

    fn test_config() -> SessionConfig {
        SessionConfig {
            idle_ttl: Duration::from_secs(10),
            max_ttl: Duration::from_secs(30),
            max_requests: 3,
            fallback_scope: SessionFallbackScope::Request,
            headers: Default::default(),
        }
    }

    fn manager() -> SessionManager {
        SessionManager::new(test_config())
    }

    fn now(seconds: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)
    }

    fn request<'a>(model_id: &'a str, headers: &'a HeaderMap) -> SessionRequest<'a> {
        SessionRequest::new(model_id, headers)
    }

    #[test]
    fn creates_new_agent_session_from_incoming_session_id_header() {
        let manager = manager();
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Venice-Proxy-Session-Id",
            HeaderValue::from_static("chat-1"),
        );

        let resolved = manager
            .get_or_create_at(request("model-a", &headers), now(0))
            .expect("session should resolve");

        assert!(resolved.created);
        assert_eq!(resolved.replaced_expired, None);
        assert_eq!(resolved.session.session_key, "model-a:chat-1");
        assert_eq!(resolved.session.model_id, "model-a");
        assert_eq!(resolved.session.agent_session_id, "chat-1");
        assert_eq!(resolved.session.scope, SessionScope::Agent);
        assert_eq!(resolved.session.request_count, 1);
    }

    #[test]
    fn reuses_existing_session_from_configured_header() {
        let mut config = test_config();
        config.headers.incoming_session_id = "X-Custom-Session-Id".to_owned();
        let manager = SessionManager::new(config);
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Custom-Session-Id",
            HeaderValue::from_static("configured-chat"),
        );

        let first = manager
            .get_or_create_at(request("model-a", &headers), now(0))
            .expect("first request should create");
        let second = manager
            .get_or_create_at(request("model-a", &headers), now(5))
            .expect("second request should reuse");

        assert!(first.created);
        assert!(!second.created);
        assert_eq!(second.session.session_key, first.session.session_key);
        assert_eq!(second.session.request_count, 2);
        assert_eq!(second.session.last_used_at, now(5));
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn configured_header_wins_over_metadata() {
        let manager = manager();
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Venice-Proxy-Session-Id",
            HeaderValue::from_static("header-session"),
        );
        let body = json!({ "metadata": { "session_id": "body-session" } });

        let resolved = manager
            .get_or_create_at(
                SessionRequest::new("model-a", &headers).with_body(&body),
                now(0),
            )
            .expect("session should resolve");

        assert_eq!(resolved.session.session_key, "model-a:header-session");
    }

    #[test]
    fn metadata_session_id_is_used_when_headers_are_missing() {
        let manager = manager();
        let headers = HeaderMap::new();
        let body = json!({ "metadata": { "session_id": "metadata-session" } });

        let resolved = manager
            .get_or_create_at(
                SessionRequest::new("model-a", &headers).with_body(&body),
                now(0),
            )
            .expect("session should resolve");

        assert_eq!(resolved.session.session_key, "model-a:metadata-session");
        assert_eq!(resolved.session.scope, SessionScope::Agent);
    }

    #[test]
    fn idle_ttl_expiration_discards_old_session_and_creates_fresh_one() {
        let manager = manager();
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Venice-Proxy-Session-Id",
            HeaderValue::from_static("chat-1"),
        );

        let first = manager
            .get_or_create_at(request("model-a", &headers), now(0))
            .expect("first request should create");
        let second = manager
            .get_or_create_at(request("model-a", &headers), now(10))
            .expect("idle-expired request should recreate");

        assert!(second.created);
        assert_eq!(
            second.replaced_expired,
            Some(SessionExpirationReason::IdleTtl)
        );
        assert_eq!(second.session.session_key, first.session.session_key);
        assert_eq!(second.session.request_count, 1);
        assert_eq!(second.session.created_at, now(10));
    }

    #[test]
    fn max_ttl_expiration_discards_old_session_and_creates_fresh_one() {
        let mut config = test_config();
        config.idle_ttl = Duration::from_secs(20);
        config.max_ttl = Duration::from_secs(30);
        let manager = SessionManager::new(config);
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Venice-Proxy-Session-Id",
            HeaderValue::from_static("chat-1"),
        );

        let first = manager
            .get_or_create_at(request("model-a", &headers), now(0))
            .expect("first request should create");
        manager
            .get_or_create_at(request("model-a", &headers), now(15))
            .expect("within idle ttl should reuse");
        let third = manager
            .get_or_create_at(request("model-a", &headers), now(30))
            .expect("max-ttl-expired request should recreate");

        assert!(third.created);
        assert_eq!(
            third.replaced_expired,
            Some(SessionExpirationReason::MaxTtl)
        );
        assert_eq!(third.session.session_key, first.session.session_key);
        assert_eq!(third.session.request_count, 1);
        assert_eq!(third.session.created_at, now(30));
    }

    #[test]
    fn max_request_expiration_discards_old_session_and_creates_fresh_one() {
        let manager = manager();
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Venice-Proxy-Session-Id",
            HeaderValue::from_static("chat-1"),
        );

        manager
            .get_or_create_at(request("model-a", &headers), now(0))
            .expect("first request should create");
        manager
            .get_or_create_at(request("model-a", &headers), now(1))
            .expect("second request should reuse");
        let third = manager
            .get_or_create_at(request("model-a", &headers), now(2))
            .expect("third request should reuse and reach max");
        let fourth = manager
            .get_or_create_at(request("model-a", &headers), now(3))
            .expect("fourth request should recreate");

        assert!(!third.created);
        assert_eq!(third.session.request_count, 3);
        assert!(fourth.created);
        assert_eq!(
            fourth.replaced_expired,
            Some(SessionExpirationReason::MaxRequests)
        );
        assert_eq!(fourth.session.request_count, 1);
    }

    #[test]
    fn request_fallback_creates_distinct_request_scoped_sessions() {
        let manager = manager();
        let headers = HeaderMap::new();

        let first = manager
            .get_or_create_at(request("model-a", &headers), now(0))
            .expect("fallback should create");
        let second = manager
            .get_or_create_at(request("model-a", &headers), now(1))
            .expect("fallback should create again");

        assert!(first.created);
        assert!(second.created);
        assert_eq!(first.session.scope, SessionScope::Request);
        assert_eq!(second.session.scope, SessionScope::Request);
        assert_ne!(
            first.session.agent_session_id,
            second.session.agent_session_id
        );
        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn agent_fallback_reuses_generated_agent_scoped_session() {
        let mut config = test_config();
        config.fallback_scope = SessionFallbackScope::Agent;
        let manager = SessionManager::new(config);
        let headers = HeaderMap::new();

        let first = manager
            .get_or_create_at(request("model-a", &headers), now(0))
            .expect("fallback should create");
        let second = manager
            .get_or_create_at(request("model-a", &headers), now(1))
            .expect("fallback should reuse");

        assert!(first.created);
        assert!(!second.created);
        assert_eq!(first.session.scope, SessionScope::Agent);
        assert_eq!(
            first.session.agent_session_id,
            second.session.agent_session_id
        );
        assert_eq!(second.session.request_count, 2);
    }

    #[test]
    fn disabled_fallback_returns_clear_error_without_creating_session() {
        let mut config = test_config();
        config.fallback_scope = SessionFallbackScope::Disabled;
        let manager = SessionManager::new(config);
        let headers = HeaderMap::new();

        let error = manager
            .get_or_create_at(request("model-a", &headers), now(0))
            .expect_err("missing session id should fail when fallback is disabled");

        assert_eq!(error, SessionError::MissingSessionIdentifier);
        assert_eq!(
            error.to_string(),
            "request does not include a session identifier and session fallback is disabled"
        );
        assert!(manager.is_empty());
    }

    #[test]
    fn cleanup_removes_expired_sessions_and_keeps_valid_sessions() {
        let manager = manager();
        let mut headers_a = HeaderMap::new();
        headers_a.insert(
            "X-Venice-Proxy-Session-Id",
            HeaderValue::from_static("chat-a"),
        );
        let mut headers_b = HeaderMap::new();
        headers_b.insert(
            "X-Venice-Proxy-Session-Id",
            HeaderValue::from_static("chat-b"),
        );

        manager
            .get_or_create_at(request("model-a", &headers_a), now(0))
            .expect("session a should create");
        manager
            .get_or_create_at(request("model-a", &headers_b), now(15))
            .expect("session b should create");

        let removed = manager.cleanup_expired_at(now(20));

        assert_eq!(removed, 1);
        assert_eq!(manager.len(), 1);
        let reused_b = manager
            .get_or_create_at(request("model-a", &headers_b), now(21))
            .expect("session b should remain valid");
        assert!(!reused_b.created);
    }

    #[test]
    fn stores_attested_model_state_on_existing_unexpired_session() {
        let manager = manager();
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Venice-Proxy-Session-Id",
            HeaderValue::from_static("chat-1"),
        );
        let session = manager
            .get_or_create_at(request("model-a", &headers), now(0))
            .expect("session should create")
            .session;

        let updated = manager
            .set_attested_model_state_at(
                &session.session_key,
                AttestedModelState {
                    model_public_key: "model-public-key".to_owned(),
                    attestation_report: json!({ "verified": true }),
                    verified_at: now(1),
                },
                now(1),
            )
            .expect("attestation state should update");

        assert_eq!(
            updated.attested_model_public_key.as_deref(),
            Some("model-public-key")
        );
        assert_eq!(
            updated.attestation_report,
            Some(json!({ "verified": true }))
        );
        assert_eq!(updated.verified_at, Some(now(1)));
    }
}
