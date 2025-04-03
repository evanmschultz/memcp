# Phase 5: MCP Server and API Implementation

## Overview

This phase focuses on implementing the MCP server and API components, leveraging the existing GraphitiAdapter implementation and using Pydantic v2 models for robust state management and validation. The goal is to create a clean server layer that coordinates operations without duplicating functionality.

## Components to Implement

### 1. Server Models (`server/models.py`)

Core Pydantic models for server state management.

```python
class ServerState(Enum):
    """Server state enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"
    STOPPED = "stopped"

class ServerStatus(BaseModel):
    """Server status model."""
    state: ServerState
    uptime: float
    active_connections: int
    queued_operations: int
    error: Optional[str] = None
    graphiti_connected: bool
    mcp_initialized: bool

class ServerMetrics(BaseModel):
    """Server metrics model."""
    request_count: int = Field(default=0, description="Total number of requests processed")
    error_count: int = Field(default=0, description="Total number of errors encountered")
    average_response_time: float = Field(default=0.0, description="Average response time in seconds")
    active_workers: int = Field(default=0, description="Number of active worker tasks")

class MemCPServerConfig(BaseModel):
    """Configuration for MemCP's MCP server implementation."""

    # Basic MCP server settings
    transport: Literal["sse", "stdio"] = Field(
        default="sse",
        description="Transport mechanism: 'sse' (Server-Sent Events) or 'stdio' (standard input/output)"
    )
    server_name: str = Field(
        default="memcp",
        description="Name of this MCP server for identification purposes"
    )
    server_description: str = Field(
        default="MemCP knowledge graph memory for AI agents",
        description="Human-readable server description"
    )
    instructions_template: str | None = Field(
        default=None,
        description="Custom instructions template to use instead of default"
    )

    # Network settings for SSE transport
    host: str = Field(
        default="127.0.0.1",
        description="Host to bind server to when using SSE transport"
    )
    port: int = Field(
        default=8000,
        description="Port for SSE transport"
    )

    # Performance settings
    request_timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds"
    )
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum number of concurrent requests"
    )
    max_queue_size: int = Field(
        default=100,
        description="Maximum size of request queue before rejecting new requests"
    )

    # Security settings
    enable_auth: bool = Field(
        default=False,
        description="Enable authentication for the server"
    )
    auth_token: str | None = Field(
        default=None,
        description="API token for simple authentication when enable_auth=True"
    )
    allow_origins: list[str] = Field(
        default=["*"],
        description="List of allowed origins for CORS (for SSE transport)"
    )
    rate_limit: int = Field(
        default=0,
        description="Rate limit in requests per minute (0 = unlimited)"
    )

    # Operational settings
    shutdown_timeout: float = Field(
        default=30.0,
        description="Maximum time in seconds to wait for graceful shutdown"
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with additional logging"
    )
    health_check_path: str = Field(
        default="/health",
        description="HTTP path for health checks (SSE transport only)"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "transport": "sse",
                    "host": "0.0.0.0",
                    "port": 8080,
                    "server_name": "memcp",
                    "debug_mode": True,
                    "max_concurrent_requests": 20
                }
            ]
        }
    )

    @field_validator("auth_token")
    @classmethod
    def validate_auth_token(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate that auth_token is provided when enable_auth is True."""
        if info.data.get("enable_auth", False) and not v:
            raise ValueError("auth_token must be provided when enable_auth is True")
        return v
```

### 2. MemCPServer Class (`server/mcp.py`)

Server implementation using Pydantic models for state management.

```python
class MemCPServer:
    """MCP server wrapper that manages the FastMCP instance and coordinates server operations."""

    def __init__(self, config: GraphitiAdapterConfig, server_config: MemCPServerConfig):
        """Initialize the server with configuration.

        Args:
            config: GraphitiAdapter configuration
            server_config: Server configuration including transport settings
        """
        self.graphiti_config = config
        self.server_config = server_config
        self.state = ServerState.INITIALIZING
        self.metrics = ServerMetrics()
        self.status = ServerStatus(
            state=ServerState.INITIALIZING,
            uptime=0.0,
            active_connections=0,
            queued_operations=0,
            graphiti_connected=False,
            mcp_initialized=False
        )
        self.mcp: Optional[FastMCP] = None
        self.graphiti: Optional[GraphitiAdapter] = None
        self.shutdown_manager: Optional[ShutdownManager] = None
        self.logger = get_logger(__name__)

    async def initialize(self) -> None:
        """Initialize server components."""
        pass

    def register_tools(self) -> None:
        """Register MCP tools using GraphitiAdapter methods."""
        pass

    def register_resources(self) -> None:
        """Register MCP resources for status and shutdown."""
        pass

    async def run(self) -> None:
        """Run the server with configured transport."""
        pass

    async def shutdown(self) -> None:
        """Coordinate graceful shutdown sequence."""
        pass
```

### 3. Shutdown Models (`server/shutdown_models.py`)

Models for shutdown state management.

```python
class ShutdownOperation(BaseModel):
    """Model for tracking shutdown operations."""
    name: str
    priority: OperationPriority
    state: OperationState
    current_step: int
    total_steps: int
    error: Optional[str] = None
    description: str = ""

class ShutdownStatus(BaseModel):
    """Model for shutdown status."""
    accepting_new: bool = True
    shutdown_requested: bool = False
    operations: dict[str, ShutdownOperation] = Field(default_factory=dict)
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
```

### 4. ShutdownManager Class (`server/shutdown.py`)

Shutdown management using Pydantic models.

```python
class ShutdownManager:
    """Manages graceful shutdown process."""

    def __init__(self):
        """Initialize shutdown manager."""
        self._status = ShutdownStatus()
        self.logger = get_logger(__name__)

    @property
    def status(self) -> ShutdownStatus:
        """Get current shutdown status."""
        return self._status

    def register_signal_handlers(self) -> None:
        """Set up handlers for SIGTERM and SIGINT."""
        pass

    async def shutdown(self) -> None:
        """Execute shutdown sequence."""
        pass

    async def broadcast_status(self) -> None:
        """Update and broadcast shutdown status."""
        pass
```

### 5. Status Resource (`api/resources/status.py`)

Health check endpoint using server models.

```python
class StatusResponse(BaseModel):
    """Status response model."""
    server_status: ServerStatus
    metrics: ServerMetrics
    timestamp: datetime

class StatusResource:
    """Status resource for health checks."""

    async def get_status(self) -> StatusResponse:
        """Get server and connection status."""
        pass
```

### 6. Shutdown Resource (`api/resources/shutdown.py`)

Shutdown status endpoint using shutdown models.

```python
class ShutdownResponse(BaseModel):
    """Shutdown response model."""
    status: ShutdownStatus
    timestamp: datetime

class ShutdownResource:
    """Shutdown status resource."""

    async def get_shutdown_status(self) -> ShutdownResponse:
        """Get current shutdown status."""
        pass
```

## Implementation Steps

1. **Model Implementation** (1-2 days)

    - Implement server state models
    - Implement shutdown models
    - Implement response models
    - Add validation rules
    - Write model tests

2. **Server Core Implementation** (2-3 days)

    - Implement MemCPServer class
    - Set up state management
    - Configure GraphitiAdapter integration
    - Implement tool registration
    - Add metric tracking

3. **Shutdown Management** (1-2 days)

    - Implement ShutdownManager class
    - Set up state tracking
    - Configure signal handlers
    - Add operation management
    - Implement status updates

4. **Resource Implementation** (1 day)

    - Implement StatusResource
    - Implement ShutdownResource
    - Add response serialization
    - Configure endpoints

5. **Testing** (2-3 days)
    - Test model validation
    - Test state transitions
    - Test resource responses
    - Test error handling

## Testing Strategy

-   Unit tests for models and validation logic
-   Integration tests for server startup and shutdown sequences
-   End-to-end tests for API endpoints
-   Mock GraphitiAdapter for isolated testing
-   Test configuration parsing and validation

## Success Criteria

-   Complete Pydantic v2 models with proper validation
-   Functional MCP server with tool and resource registration
-   Graceful shutdown handling with operation tracking
-   Health check and status endpoints
-   Comprehensive test coverage

## Dependencies

### External

-   FastMCP
-   GraphitiAdapter
-   Pydantic v2
-   Rich

### Internal

-   Configuration system
-   Logging system
-   Queue management
-   Error handling

## Risks and Mitigations

### Risks

-   Model complexity
-   State management
-   Validation overhead
-   Type safety

### Mitigations

-   Clear documentation
-   Comprehensive testing
-   Performance monitoring
-   Type checking

## Review Points

### Code Review

-   Model design
-   State management
-   Error handling
-   Type safety

### Testing Review

-   Validation coverage
-   State transitions
-   Error scenarios
-   Performance impact

## Next Steps

1. Implement state models
2. Build server core
3. Add shutdown management
4. Create resources
5. Write tests
6. Document components
7. Review and refine
