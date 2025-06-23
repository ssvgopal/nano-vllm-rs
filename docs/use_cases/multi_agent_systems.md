# Multi-Agent Systems with nano-vllm-rs

Build sophisticated multi-agent systems (MAS) where multiple AI agents collaborate, compete, or coordinate to solve complex problems.

## 📋 Prerequisites

- Rust toolchain
- `nano-vllm-rs`
- `tokio` for async runtime
- `serde` for serialization
- `dashmap` for concurrent data structures
- `tracing` for distributed tracing

## 🏗️ Core Architecture

### 1. Agent Trait and Base Implementation

```rust
// src/agent/mod.rs
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentMessage {
    pub from: String,
    pub to: String,
    pub content: Value,
    pub message_type: String,
}

#[async_trait]
pub trait Agent: Send + Sync + 'static {
    fn id(&self) -> &str;
    
    async fn handle_message(
        &self,
        message: AgentMessage,
        context: &mut AgentContext,
    ) -> Result<(), Box<dyn std::error::Error>>;
    
    fn capabilities(&self) -> &[String];
}

pub struct AgentContext {
    pub message_tx: mpsc::Sender<AgentMessage>,
    pub shared_state: Arc<dyn SharedState>,
}

#[async_trait]
pub trait SharedState: Send + Sync {
    async fn get(&self, key: &str) -> Option<Value>;
    async fn set(&mut self, key: String, value: Value) -> Result<(), Box<dyn std::error::Error>>;
}
```

### 2. Agent Manager

```rust
// src/agent/manager.rs
use super::*;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AgentManager {
    agents: DashMap<String, Arc<dyn Agent>>,
    message_router: MessageRouter,
    shared_state: Arc<Mutex<dyn SharedState>>,
}

impl AgentManager {
    pub fn new(shared_state: Arc<Mutex<dyn SharedState>>) -> Self {
        Self {
            agents: DashMap::new(),
            message_router: MessageRouter::new(),
            shared_state,
        }
    }
    
    pub fn register_agent(&self, agent: Arc<dyn Agent>) -> Result<(), String> {
        if self.agents.contains_key(agent.id()) {
            return Err(format!("Agent with ID {} already exists", agent.id()));
        }
        
        self.agents.insert(agent.id().to_string(), agent.clone());
        self.message_router.register_agent(agent.id());
        
        Ok(())
    }
    
    pub async fn start(&self) {
        let (tx, mut rx) = mpsc::channel(100);
        
        // Spawn message processing task
        let agents = self.agents.clone();
        let router = self.message_router.clone();
        let shared_state = self.shared_state.clone();
        
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                let agents = agents.clone();
                let router = router.clone();
                let shared_state = shared_state.clone();
                
                tokio::spawn(async move {
                    if let Some(agent) = agents.get(&msg.to) {
                        let mut context = AgentContext {
                            message_tx: tx.clone(),
                            shared_state: shared_state.lock().await,
                        };
                        
                        if let Err(e) = agent.handle_message(msg, &mut context).await {
                            eprintln!("Error processing message: {}", e);
                        }
                    } else {
                        eprintln!("Unknown agent: {}", msg.to);
                    }
                });
            }
        });
    }
    
    pub fn get_message_sender(&self) -> mpsc::Sender<AgentMessage> {
        self.message_router.get_sender()
    }
}
```

### 3. Message Router

```rust
// src/agent/router.rs
use super::*;
use dashmap::DashSet;
use std::sync::Arc;

#[derive(Clone)]
pub struct MessageRouter {
    agents: Arc<DashSet<String>>,
    senders: Arc<dashmap::DashMap<String, mpsc::Sender<AgentMessage>>>,
}

impl MessageRouter {
    pub fn new() -> Self {
        Self {
            agents: Arc::new(DashSet::new()),
            senders: Arc::new(dashmap::DashMap::new()),
        }
    }
    
    pub fn register_agent(&self, agent_id: &str) {
        self.agents.insert(agent_id.to_string());
    }
    
    pub fn get_sender(&self) -> mpsc::Sender<AgentMessage> {
        let (tx, rx) = mpsc::channel(100);
        
        // Store the sender in the router
        let agent_id = uuid::Uuid::new_v4().to_string();
        self.senders.insert(agent_id.clone(), tx.clone());
        
        // Spawn a task to handle incoming messages
        let agents = self.agents.clone();
        let senders = self.senders.clone();
        
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if let Some(sender) = senders.get(&msg.to) {
                    if let Err(e) = sender.send(msg).await {
                        eprintln!("Failed to forward message: {}", e);
                    }
                } else {
                    eprintln!("No sender found for agent: {}", msg.to);
                }
            }
            
            // Clean up
            senders.remove(&agent_id);
        });
        
        tx
    }
}
```

## 🏗️ Agent Patterns

### 1. Hierarchical Agents

```rust
// src/agent/patterns/hierarchical.rs
use super::*;

pub struct ManagerAgent {
    id: String,
    worker_agents: Vec<String>,
}

#[async_trait]
impl Agent for ManagerAgent {
    fn id(&self) -> &str {
        &self.id
    }
    
    async fn handle_message(
        &self,
        message: AgentMessage,
        context: &mut AgentContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match message.message_type.as_str() {
            "task_request" => {
                // Delegate to worker agents
                for worker_id in &self.worker_agents {
                    context.message_tx.send(AgentMessage {
                        from: self.id.clone(),
                        to: worker_id.clone(),
                        content: message.content.clone(),
                        message_type: "task".to_string(),
                    }).await?;
                }
            }
            "task_result" => {
                // Aggregate results from workers
                println!("Received result from {}: {:?}", message.from, message.content);
            }
            _ => {}
        }
        Ok(())
    }
    
    fn capabilities(&self) -> &[String] {
        &["delegation".to_string(), "coordination".to_string()]
    }
}
```

### 2. Market-Based Agents

```rust
// src/agent/patterns/market.rs
use super::*;
use std::collections::HashMap;

pub struct AuctioneerAgent {
    id: String,
    bids: HashMap<String, (String, f64)>, // task_id -> (bidder_id, amount)
}

#[async_trait]
impl Agent for AuctioneerAgent {
    fn id(&self) -> &str {
        &self.id
    }
    
    async fn handle_message(
        &self,
        message: AgentMessage,
        context: &mut AgentContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match message.message_type.as_str() {
            "bid" => {
                let task_id = message.content["task_id"].as_str().unwrap();
                let amount = message.content["amount"].as_f64().unwrap();
                
                // Store the bid if it's the highest so far
                if let Some((_, current_amount)) = self.bids.get(task_id) {
                    if amount > *current_amount {
                        self.bids.insert(task_id.to_string(), (message.from, amount));
                    }
                } else {
                    self.bids.insert(task_id.to_string(), (message.from, amount));
                }
            }
            "close_auction" => {
                let task_id = message.content["task_id"].as_str().unwrap();
                if let Some((winner, _)) = self.bids.remove(task_id) {
                    // Notify the winner
                    context.message_tx.send(AgentMessage {
                        from: self.id.clone(),
                        to: winner,
                        content: json!({ "task_id": task_id, "status": "awarded" }),
                        message_type: "auction_result".to_string(),
                    }).await?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    fn capabilities(&self) -> &[String] {
        &["auction".to_string(), "coordination".to_string()]
    }
}
```

### 3. Swarm Intelligence

```rust
// src/agent/patterns/swarm.rs
use super::*;
use std::collections::HashMap;

pub struct SwarmAgent {
    id: String,
    position: (f64, f64),
    velocity: (f64, f64),
    neighbors: HashMap<String, (f64, f64)>,
}

#[async_trait]
impl Agent for SwarmAgent {
    fn id(&self) -> &str {
        &self.id
    }
    
    async fn handle_message(
        &self,
        message: AgentMessage,
        context: &mut AgentContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match message.message_type.as_str() {
            "position_update" => {
                let x = message.content["x"].as_f64().unwrap();
                let y = message.content["y"].as_f64().unwrap();
                self.neighbors.insert(message.from, (x, y));
                
                // Apply swarm rules
                self.apply_flocking_rules();
                
                // Broadcast new position
                for neighbor_id in self.neighbors.keys() {
                    context.message_tx.send(AgentMessage {
                        from: self.id.clone(),
                        to: neighbor_id.clone(),
                        content: json!({ "x": self.position.0, "y": self.position.1 }),
                        message_type: "position_update".to_string(),
                    }).await?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    fn capabilities(&self) -> &[String] {
        &["swarm".to_string(), "coordination".to_string()]
    }
}

impl SwarmAgent {
    fn apply_flocking_rules(&mut self) {
        // Implement boids-like flocking behavior
        // 1. Alignment: Steer towards the average heading of local flockmates
        // 2. Cohesion: Steer to move toward the average position of local flockmates
        // 3. Separation: Avoid crowding local flockmates
        
        // Simplified implementation
        let mut avg_velocity = (0.0, 0.0);
        let mut avg_position = (0.0, 0.0);
        let mut separation = (0.0, 0.0);
        let mut count = 0;
        
        for (_, &(x, y)) in &self.neighbors {
            avg_velocity.0 += x - self.position.0;
            avg_velocity.1 += y - self.position.1;
            
            avg_position.0 += x;
            avg_position.1 += y;
            
            // Avoid crowding
            let diff_x = self.position.0 - x;
            let diff_y = self.position.1 - y;
            let distance = (diff_x * diff_x + diff_y * diff_y).sqrt();
            
            if distance > 0.0 {
                separation.0 += diff_x / distance;
                separation.1 += diff_y / distance;
            }
            
            count += 1;
        }
        
        if count > 0 {
            // Update velocity based on swarm rules
            self.velocity.0 += (avg_velocity.0 / count as f64) * 0.1;  // Alignment
            self.velocity.0 += (avg_position.0 / count as f64 - self.position.0) * 0.01;  // Cohesion
            self.velocity.0 += separation.0 * 0.05;  // Separation
            
            self.velocity.1 += (avg_velocity.1 / count as f64) * 0.1;  // Alignment
            self.velocity.1 += (avg_position.1 / count as f64 - self.position.1) * 0.01;  // Cohesion
            self.velocity.1 += separation.1 * 0.05;  // Separation
            
            // Limit velocity
            let speed = (self.velocity.0 * self.velocity.0 + self.velocity.1 * self.velocity.1).sqrt();
            let max_speed = 2.0;
            if speed > max_speed {
                self.velocity.0 = (self.velocity.0 / speed) * max_speed;
                self.velocity.1 = (self.velocity.1 / speed) * max_speed;
            }
            
            // Update position
            self.position.0 += self.velocity.0;
            self.position.1 += self.velocity.1;
        }
    }
}
```

## 🚀 Usage Example: Autonomous Research Team

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize shared state
    let shared_state = Arc::new(Mutex::new(InMemorySharedState::new()));
    
    // Create agent manager
    let manager = AgentManager::new(shared_state.clone());
    
    // Create and register agents
    let researcher = Arc::new(ResearcherAgent::new("researcher_1"));
    let analyst = Arc::new(AnalystAgent::new("analyst_1"));
    let writer = Arc::new(WriterAgent::new("writer_1"));
    
    manager.register_agent(researcher.clone())?;
    manager.register_agent(analyst.clone())?;
    manager.register_agent(writer.clone())?;
    
    // Start the system
    manager.start().await;
    
    // Get message sender
    let sender = manager.get_message_sender();
    
    // Start a research task
    sender.send(AgentMessage {
        from: "user".to_string(),
        to: "researcher_1".to_string(),
        content: json!({
            "topic": "Recent advances in quantum computing",
            "depth": "comprehensive"
        }),
        message_type: "research_request".to_string(),
    }).await?;
    
    // Keep the application running
    tokio::signal::ctrl_c().await?;
    
    Ok(())
}
```

## 🧠 Best Practices

1. **Agent Design**
   - Single Responsibility Principle for each agent
   - Clear message protocols
   - Idempotent message handling

2. **Scalability**
   - Shard agents by domain
   - Use consistent hashing for load balancing
   - Implement backpressure

3. **Fault Tolerance**
   - Implement heartbeats
   - Use circuit breakers
   - Implement retry mechanisms

4. **Observability**
   - Distributed tracing
   - Metrics collection
   - Structured logging

## 📚 Additional Resources

- [Multi-Agent Systems: A Survey](https://arxiv.org/abs/2003.09050)
- [JADE: Java Agent DEvelopment Framework](https://jade.tilab.com/)
- [Autonomous Agents on the Web](https://www.w3.org/TR/2023/WD-web-autonomous-agents-20230404/)

## 🛠️ Troubleshooting

### Common Issues

1. **Deadlocks**
   - Avoid holding locks across await points
   - Use timeouts for message passing
   - Implement deadlock detection

2. **Message Flooding**
   - Implement rate limiting
   - Use backpressure mechanisms
   - Aggregate messages when possible

3. **State Consistency**
   - Use CRDTs for distributed state
   - Implement version vectors
   - Use transactions when needed

4. **Performance Bottlenecks**
   - Profile agent message processing
   - Optimize serialization
   - Consider actor model optimizations

## 🚀 Future Extensions

1. **Federated Learning**
   - Distributed model training across agents
   - Privacy-preserving updates

2. **Blockchain Integration**
   - Smart contracts for agent coordination
   - Decentralized reputation systems

3. **Edge Computing**
   - Deploy agents on edge devices
   - Local processing with periodic sync
