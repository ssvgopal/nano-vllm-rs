# Building AI Agents with Function Calling

Create autonomous AI agents that can use tools and APIs through function calling capabilities.

## 📋 Prerequisites

- Rust toolchain
- `nano-vllm-rs` with function calling support
- `serde_json` for JSON handling
- `reqwest` for API calls (if needed)

## 🛠️ Implementation

### 1. Define Tool Interface

```rust
// src/agent/tools.rs
use serde_json::Value;
use std::collections::HashMap;
use async_trait::async_trait;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    async fn execute(&self, parameters: Value) -> Result<Value, String>;
}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }
    
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name().to_string(), Box::new(tool));
    }
    
    pub fn get_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }
    
    pub fn get_tools_schema(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|t| {
                json!({{
                    "name": t.name(),
                    "description": t.description(),
                    "parameters": t.parameters()
                }})
            })
            .collect()
    }
}
```

### 2. Implement Example Tools

```rust
// src/agent/builtin_tools.rs
use super::tools::{Tool, ToolRegistry};
use serde_json::{json, Value};
use chrono::Local;

pub struct CurrentTimeTool;

#[async_trait::async_trait]
impl Tool for CurrentTimeTool {
    fn name(&self) -> &str {
        "get_current_time"
    }
    
    fn description(&self) -> &str {
        "Get the current date and time"
    }
    
    fn parameters(&self) -> Value {
        json!({})
    }
    
    async fn execute(&self, _parameters: Value) -> Result<Value, String> {
        let now = Local::now();
        Ok(json!({
            "timestamp": now.timestamp(),
            "datetime": now.to_rfc3339(),
            "timezone": "local"
        }))
    }
}

pub struct CalculatorTool;

#[async_trait::async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculate"
    }
    
    fn description(&self) -> &str {
        "Perform mathematical calculations"
    }
    
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        })
    }
    
    async fn execute(&self, parameters: Value) -> Result<Value, String> {
        let expr = parameters["expression"]
            .as_str()
            .ok_or("Expression must be a string")?;
            
        // Note: In production, use a proper expression evaluator
        // This is a simplified example
        if let Ok(result) = evalexpr::eval(expr) {
            Ok(json!({"result": result.to_string()}))
        } else {
            Err("Failed to evaluate expression".to_string())
        }
    }
}

pub fn register_builtin_tools(registry: &mut ToolRegistry) {
    registry.register(CurrentTimeTool);
    registry.register(CalculatorTool);
}
```

### 3. Agent Implementation

```rust
// src/agent/mod.rs
mod tools;
mod builtin_tools;

use std::sync::Arc;
use tokio::sync::Mutex;
use nano_vllm_rs::{
    LLMEngine, 
    FunctionCall,
    Message,
    Role,
};
use serde_json::{json, Value};
use anyhow::{Result, anyhow};

pub use tools::{Tool, ToolRegistry};

pub struct Agent {
    llm: LLMEngine,
    tool_registry: Arc<Mutex<ToolRegistry>>,
    conversation_history: Vec<Message>,
}

impl Agent {
    pub async fn new() -> Result<Self> {
        let mut tool_registry = ToolRegistry::new();
        builtin_tools::register_builtin_tools(&mut tool_registry);
        
        let llm = LLMEngine::new(Default::default()).await?;
        
        Ok(Self {
            llm,
            tool_registry: Arc::new(Mutex::new(tool_registry)),
            conversation_history: Vec::new(),
        })
    }
    
    pub async fn register_tool<T: Tool + 'static>(&self, tool: T) {
        self.tool_registry.lock().await.register(tool);
    }
    
    pub async fn process_message(&mut self, message: &str) -> Result<String> {
        // Add user message to history
        self.conversation_history.push(Message {
            role: Role::User,
            content: message.to_string(),
            function_call: None,
        });
        
        loop {
            // Generate LLM response
            let response = self.llm.chat_completion(
                &self.conversation_history,
                Some(&self.get_tools_schema().await),
                None,
            ).await?;
            
            // Add assistant message to history
            self.conversation_history.push(Message {
                role: Role::Assistant,
                content: response.content.clone(),
                function_call: response.function_call.clone(),
            });
            
            // Check if function call is needed
            if let Some(function_call) = response.function_call {
                let result = self.execute_function(&function_call).await?;
                
                // Add function result to history
                self.conversation_history.push(Message {
                    role: Role::Function,
                    content: serde_json::to_string(&result)?,
                    function_call: None,
                });
                
                continue;
            }
            
            return Ok(response.content);
        }
    }
    
    async fn get_tools_schema(&self) -> Value {
        let registry = self.tool_registry.lock().await;
        json!(registry.get_tools_schema())
    }
    
    async fn execute_function(
        &self, 
        function_call: &FunctionCall,
    ) -> Result<Value, anyhow::Error> {
        let registry = self.tool_registry.lock().await;
        let tool = registry
            .get_tool(&function_call.name)
            .ok_or_else(|| anyhow!("Unknown tool: {}", function_call.name))?;
            
        let parameters: Value = serde_json::from_str(&function_call.arguments)?;
        
        match tool.execute(parameters).await {
            Ok(result) => Ok(result),
            Err(e) => Err(anyhow!("Tool execution error: {}", e)),
        }
    }
}
```

## 🚀 Usage Example

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize agent
    let mut agent = Agent::new().await?;
    
    // Register custom tools
    agent.register_tool(MyCustomTool).await;
    
    // Chat loop
    let mut rl = rustyline::DefaultEditor::new()?;
    
    loop {
        let input = rl.readline("You: ")?;
        if input.trim().eq_ignore_ascii_case("exit") {
            break;
        }
        
        match agent.process_message(&input).await {
            Ok(response) => {
                println!("\nAssistant: {}\n", response);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
    
    Ok(())
}
```

## 📈 Advanced Techniques

### Parallel Function Execution

```rust
impl Agent {
    async fn execute_functions_parallel(
        &self,
        function_calls: Vec<FunctionCall>,
    ) -> Result<Vec<(String, Value)>> {
        let mut tasks = Vec::new();
        
        for call in function_calls {
            let registry = self.tool_registry.clone();
            tasks.push(tokio::spawn(async move {
                let registry = registry.lock().await;
                let tool = registry.get_tool(&call.name)
                    .ok_or_else(|| format!("Unknown tool: {}", call.name))?;
                    
                let parameters: Value = serde_json::from_str(&call.arguments)
                    .map_err(|e| e.to_string())?;
                    
                tool.execute(parameters).await
                    .map(|result| (call.name, result))
                    .map_err(|e| e.into())
            }));
        }
        
        let mut results = Vec::new();
        for task in tasks {
            match task.await {
                Ok(Ok((name, result))) => results.push((name, result)),
                Ok(Err(e)) => eprintln!("Tool error: {}", e),
                Err(e) => eprintln!("Task error: {}", e),
            }
        }
        
        Ok(results)
    }
}
```

### Tool Memory and State

```rust
pub struct StatefulTool<T> {
    state: Arc<Mutex<T>>,
    name: String,
    description: String,
    parameters: Value,
    execute_fn: fn(&mut T, Value) -> Result<Value, String>,
}

#[async_trait::async_trait]
impl<T: Send + 'static> Tool for StatefulTool<T> {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn parameters(&self) -> Value {
        self.parameters.clone()
    }
    
    async fn execute(&self, parameters: Value) -> Result<Value, String> {
        let mut state = self.state.lock().await;
        (self.execute_fn)(&mut state, parameters)
    }
}
```

## 🧠 Best Practices

1. **Tool Design**
   - Keep tools focused and single-purpose
   - Provide clear documentation
   - Validate inputs thoroughly
   - Handle errors gracefully

2. **Security**
   - Sanitize all inputs
   - Implement rate limiting
   - Validate function call parameters
   - Use authentication for external services

3. **Performance**
   - Cache frequent function calls
   - Implement timeouts for long-running operations
   - Use streaming for large outputs

4. **User Experience**
   - Provide helpful error messages
   - Include usage examples in tool descriptions
   - Support natural language queries

## 📚 Additional Resources

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)

## 🛠️ Troubleshooting

### Common Issues

1. **Function Calling Not Working**
   - Verify tool schemas are correctly formatted
   - Check parameter validation
   - Ensure the model supports function calling

2. **Poor Tool Selection**
   - Improve tool descriptions
   - Add more examples
   - Adjust temperature for more deterministic behavior

3. **Performance Bottlenecks**
   - Profile tool execution time
   - Implement caching where appropriate
   - Consider batching requests

4. **State Management**
   - Use proper synchronization
   - Implement timeouts for locks
   - Consider using actor model for complex state
