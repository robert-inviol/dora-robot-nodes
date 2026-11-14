//! Who is calling the MCP endpoint. Every claude process gets its own agent id and a random bearer token, passed in
//! its environment and sent back on every tool call; the registry knows which duty each process serves.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

use rand::RngCore;

use crate::model::Duty;

/// One claude process, as named in its `X-Agent-Id` header.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AgentId(pub String);

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The bearer token of one claude process. It never appears in logs or debug output.
#[derive(Clone, PartialEq, Eq)]
pub struct AgentToken(String);

impl AgentToken {
    pub fn random() -> Self {
        let mut bytes = [0_u8; 32];
        rand::rng().fill_bytes(&mut bytes);
        Self(bytes.iter().map(|b| format!("{b:02x}")).collect())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }

    fn matches(&self, presented: &str) -> bool {
        let (ours, theirs) = (self.0.as_bytes(), presented.as_bytes());
        ours.len() == theirs.len() && ours.iter().zip(theirs).fold(0_u8, |diff, (a, b)| diff | (a ^ b)) == 0
    }
}

impl fmt::Debug for AgentToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AgentToken(..)")
    }
}

/// A request that passed authentication: which process made it, and the duty it serves. A warm process serves none
/// yet; it may connect and list tools but not call them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Authenticated {
    pub agent: AgentId,
    pub duty: Option<Duty>,
}

/// A tool call from a process that serves a duty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Caller {
    pub agent: AgentId,
    pub duty: Duty,
}

impl Authenticated {
    pub fn caller(self) -> Result<Caller, AuthError> {
        match self.duty {
            Some(duty) => Ok(Caller { agent: self.agent, duty }),
            None => Err(AuthError::Unassigned(self.agent)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthError {
    #[error("no X-Agent-Id header")]
    NoAgentId,
    #[error("no bearer token")]
    NoToken,
    #[error("unknown agent or wrong token")]
    Rejected,
    #[error("agent {0} serves no question yet")]
    Unassigned(AgentId),
}

struct Registration {
    token: AgentToken,
    duty: Option<Duty>,
}

#[derive(Default)]
pub struct AgentRegistry {
    agents: Mutex<HashMap<AgentId, Registration>>,
}

impl AgentRegistry {
    /// A new process's credentials. It serves no duty until assigned, so a warm process cannot act.
    pub fn register(&self, agent: AgentId) -> AgentToken {
        let token = AgentToken::random();
        let registration = Registration { token: token.clone(), duty: None };
        let previous = self.agents.lock().expect("registry lock").insert(agent.clone(), registration);
        assert!(previous.is_none(), "agent id {agent} registered twice");
        token
    }

    pub fn assign(&self, agent: &AgentId, duty: Duty) {
        let mut agents = self.agents.lock().expect("registry lock");
        agents.get_mut(agent).unwrap_or_else(|| panic!("assigning unregistered agent {agent}")).duty = Some(duty);
    }

    pub fn unregister(&self, agent: &AgentId) {
        self.agents.lock().expect("registry lock").remove(agent);
    }

    pub fn verify(&self, agent: Option<&str>, bearer: Option<&str>) -> Result<Authenticated, AuthError> {
        let agent = AgentId(agent.ok_or(AuthError::NoAgentId)?.to_owned());
        let bearer = bearer.ok_or(AuthError::NoToken)?;
        let agents = self.agents.lock().expect("registry lock");
        let registration = agents.get(&agent).filter(|r| r.token.matches(bearer)).ok_or(AuthError::Rejected)?;
        Ok(Authenticated { duty: registration.duty, agent })
    }
}

#[cfg(test)]
mod tests {
    use perception_wire::navigation::QuestionId;

    use super::*;

    const DUTY: Duty = Duty::Reconcile { question: QuestionId(4) };

    #[test]
    fn a_registered_and_assigned_agent_with_its_token_serves_its_duty() {
        let registry = AgentRegistry::default();
        let agent = AgentId("agent-1".to_owned());
        let token = registry.register(agent.clone());
        let spare = registry.verify(Some("agent-1"), Some(token.expose())).unwrap();
        assert_eq!(spare.caller(), Err(AuthError::Unassigned(agent.clone())));
        registry.assign(&agent, DUTY);
        let assigned = registry.verify(Some("agent-1"), Some(token.expose())).unwrap();
        assert_eq!(assigned.caller(), Ok(Caller { agent: agent.clone(), duty: DUTY }));
        registry.unregister(&agent);
        assert_eq!(registry.verify(Some("agent-1"), Some(token.expose())), Err(AuthError::Rejected));
    }

    #[test]
    fn another_agents_token_is_rejected() {
        let registry = AgentRegistry::default();
        let first = registry.register(AgentId("agent-1".to_owned()));
        registry.register(AgentId("agent-2".to_owned()));
        registry.assign(&AgentId("agent-2".to_owned()), DUTY);
        assert_eq!(registry.verify(Some("agent-2"), Some(first.expose())), Err(AuthError::Rejected));
        assert_eq!(registry.verify(None, Some(first.expose())), Err(AuthError::NoAgentId));
        assert_eq!(registry.verify(Some("agent-2"), None), Err(AuthError::NoToken));
    }

    #[test]
    fn a_token_never_shows_in_debug_output() {
        let token = AgentToken::random();
        assert_eq!(format!("{token:?}"), "AgentToken(..)");
        assert_eq!(token.expose().len(), 64);
    }
}
