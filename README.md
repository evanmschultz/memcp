<!-- # MemCP - Memory for AI Agents

MemCP is a modular, extensible framework for AI agent memory based on the Model Context Protocol (MCP). It extends the Graphiti knowledge graph framework, allowing AI agents to build, maintain, and query a temporal knowledge graph of facts, entities, and relationships.

## Features

-   **Extensible Memory Framework**: Store and retrieve both structured and unstructured knowledge
-   **Temporal Knowledge Graph**: Support for time-aware queries and knowledge versioning
-   **Flexible Configuration**: Simple configuration via environment variables, TOML files, or CLI arguments
-   **MCP Integration**: Works with any MCP-compatible client (Claude, Claude Opus, GPT)
-   **Entity Extraction**: Automatic entity and relationship extraction
-   **Local Development**: Runs locally for development and testing

## Installation

### Prerequisites

1. Python 3.10 or higher
2. Neo4j database (version 5.26 or later)
3. OpenAI API key for LLM operations

### Using pip

```bash
pip install memcp
```

### From source

```bash
git clone https://github.com/yourusername/memcp.git
cd memcp
pip install -e .
```

## Configuration

MemCP uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration management. You can configure MemCP in several ways (in order of precedence):

1. **Command Line Arguments** (highest priority)
2. **Environment Variables**
3. **Configuration File** (.toml)
4. **Default Values** (lowest priority)

### Configuration File

Create a `config.toml` file in your working directory or specify a path with the `--config_file` argument:

```toml
# config.toml

[neo4j]
uri = "bolt://localhost:7687"
user = "neo4j"
# DO NOT set password here - use environment variables!

[openai]
# DO NOT set api_key here - use environment variables!
base_url = "https://api.openai.com/v1"  # Optional, for custom OpenAI endpoints

[model]
name = "gpt-4o-mini"

[graph]
# id = "my-graph"  # Optional, random ID will be generated if not provided
use_custom_entities = false

[server]
transport = "sse"  # or "stdio"
host = "127.0.0.1"
port = 8000
```

### Environment Variables

Set these in your environment or in a `.env` file:

```
# Required
NEO4J_PASSWORD=your_neo4j_password
OPENAI_API_KEY=your_openai_api_key

# Optional
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
MODEL_NAME=gpt-4o
GRAPH_ID=my-graph-id
```

### Command Line Arguments

```bash
memcp --neo4j.uri="bolt://localhost:7687" --model.name="gpt-4o" --graph.id="my-graph"
```

Nested configuration can be accessed with dot notation (e.g., `--graph.id`).

### Security Note

For security reasons, sensitive information like API keys and passwords should be provided through environment variables, not in the configuration file. MemCP will warn you if it detects sensitive information in a TOML file.

## Running MemCP

### Basic Usage

```bash
# Using defaults
memcp

# With custom graph ID
memcp --graph.id my-graph

# With custom config file
memcp --config_file ~/my-config.toml
```

### Advanced Options

```bash
# Use custom entities for extraction
memcp --graph.use_custom_entities

# Use a specific OpenAI model
memcp --model.name gpt-4o

# Destroy existing graph (WARNING: destructive!)
memcp --destroy_graph
```

## Integrating with MCP Clients

### Configuration for Cursor IDE

Add the following to your Cursor plugin configuration:

```json
{
	"mcpServers": {
		"graphiti": {
			"transport": "sse",
			"url": "http://localhost:8000/sse"
		}
	}
}
```

### Starting with stdio for CLI Interfaces

```bash
memcp --server.transport stdio
```

## Available Tools

MemCP exposes these MCP tools:

-   `add_episode`: Add an episode to the knowledge graph
-   `search_nodes`: Search for entity nodes
-   `search_facts`: Search for relationships between entities
-   `delete_entity_edge`: Remove a relationship
-   `delete_episode`: Remove an episode
-   `get_entity_edge`: Retrieve details of a relationship
-   `get_episodes`: Get recent episodes
-   `clear_graph`: Reset the knowledge graph

## Working with Different Data Types

MemCP can process:

### Text Data

```
add_episode(
    name="Meeting Notes",
    episode_body="Met with John from Engineering about the new feature...",
    source="text"
)
```

### Structured JSON

```
add_episode(
    name="Customer Profile",
    episode_body="{\"company\": \"Acme Corp\", \"contact\": \"John Doe\"}",
    source="json"
)
```

### Message Format

```
add_episode(
    name="User Conversation",
    episode_body="...",
    source="message"
)
```

## Docker Deployment

A Dockerfile and docker-compose.yml file are provided for containerized deployment:

```bash
docker compose up
```

Environment variables can be set in a `.env` file or passed directly to docker-compose.

## Project Architecture

MemCP follows a modular design pattern:

-   `config/`: Configuration management with pydantic-settings
-   `api/`: MCP server and API endpoints
-   `core/`: Core memory and queue functionality
-   `models/`: Pydantic data models
-   `console/`: Terminal UI components
-   `utils/`: Utility functions and helpers

## Contributing

Contributions are welcome! Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

## Above is all outdated.

## Optional Dependencies

-   `anthropic`: Support for Anthropic's Claude models

    ```bash
    pip install your-package[anthropic]
    ```

-   `dev`: Development dependencies
    ```bash
    pip install -e ".[dev]"
    ```
