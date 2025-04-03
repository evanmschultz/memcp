# Instructions for MemCP IDE Agent Memory

## Configuration and Setup

### Initial Configuration Protocol

IMPORTANT: These configurations MUST be established before any other actions unless explicitly overridden by the user.

At the start of each conversation, offer to configure these parameters. Present defaults and only ask for changes if user wants to customize:

1. **Memory Operations**:

    - `MEMORY_SEARCH_FREQUENCY`:
        - Default: `EVERY_TURN` (search before each response and before any memory saving to see what should be updated)
        - Options: `TOPIC_CHANGE`, `ON_DEMAND`
    - `MEMORY_UPDATE_FREQUENCY`:
        - Default: `IMMEDIATE` (store new information right away)
        - Options: `BATCHED`, `END_OF_TOPIC`
    - `MEMORY_SYNC_MODE`:
        - Default: `WAIT_SHORT` (wait up to 5s for memory operations)
        - Options: `NO_WAIT`, `WAIT_LONG`

2. **Verification and Feedback**:

    - `VERIFICATION_LEVEL`:
        - Default: `ADAPTIVE` (starts high, adjusts based on user interaction)
        - Options: `HIGH` (confirm everything), `MEDIUM` (major changes), `LOW` (minimal)
    - `PATTERN_TRACKING`:
        - Default: `ASK_ALWAYS` (confirm before saving patterns)
        - Options: `AUTO_SAVE`, `ASK_ONCE_PER_TYPE`, `DISABLED`
    - `UPDATE_CONFIRMATION`:
        - Default: `SIGNIFICANT` (confirm only significant updates)
        - Options: `ALL` (confirm every update), `NONE` (no confirmation)

3. **Context Management**:

    - `CONTEXT_DEPTH`:
        - Default: `BALANCED` (moderate project context gathering)
        - Options: `THOROUGH`, `MINIMAL`
    - `DOCUMENTATION_PRIORITY`:
        - Default: `LOCAL_FIRST` (project docs before web search)
        - Options: `WEB_FIRST`, `LOCAL_ONLY`
    - `STYLE_ENFORCEMENT`:
        - Default: `ADAPTIVE` (infer and follow project style)
        - Options: `STRICT` (always ask), `RELAXED`

4. **Error Handling**:

    - `API_TIMEOUT`:
        - Default: `MODERATE` (10s search, 30s updates)
        - Options: `AGGRESSIVE` (5s/15s), `PATIENT` (30s/90s)
    - `RETRY_STRATEGY`:
        - Default: `EXPONENTIAL` (exponential backoff with 3 retries)
        - Options: `AGGRESSIVE` (quick retries), `SINGLE`, `NONE`
    - `FAILURE_MODE`:
        - Default: `GRACEFUL` (proceed with partial data if possible)
        - Options: `STRICT` (fail fast), `BEST_EFFORT`

5. **Session Preferences**:
    - `PROACTIVITY_LEVEL`:
        - Default: `BALANCED` (suggest important patterns/preferences)
        - Options: `HIGH` (suggest everything), `LOW` (minimal suggestions)
    - `LEARNING_MODE`:
        - Default: `INTERACTIVE` (learn from user corrections)
        - Options: `PASSIVE` (minimal learning), `AGGRESSIVE`
    - `MEMORY_RETENTION`:
        - Default: `SESSION` (maintain context within session)
        - Options: `PERSISTENT` (try to recall across sessions), `MINIMAL`

### Configuration Management

1. **Mid-conversation Updates**:

    - All configurations can be updated at any time
    - Use command: `update_config <parameter> <value>`
    - Changes take effect immediately
    - Previous operations are not affected

2. **Configuration Persistence**:

    - Configurations are stored in memory graph
    - Can be recalled for future sessions
    - User can reset to defaults at any time

3. **Configuration Conflicts**:

    - Explicit user commands always override defaults
    - Temporary overrides available for single operations
    - Conflicts are resolved in favor of user preferences

4. **Default Behavior**:
    - All operations follow configured settings
    - Deviations must be explicitly requested
    - User is informed of significant deviations

## Core Memory Operations

### Before Starting Any Task

1. **Comprehensive Context Gathering**:

    - Search project state and structure
    - Review relevant documentation
    - Check coding style guides and conventions
    - Identify project-specific tools and practices
    - Search web for unclear technical aspects

2. **Memory Search Protocol**:

    - Use `search_nodes` for preferences, procedures, requirements
    - Use `search_facts` for relationships and context
    - Filter by entity type (`Preference`, `Procedure`, `Requirement`)
    - Use multiple search queries for complex topics
    - Implement exponential backoff for failed searches

3. **Search Timeout Handling**:
    - Set timeout based on `API_TIMEOUT`
    - If timeout occurs:
        - Log attempt and failure
        - Notify user of delay
        - Offer to proceed with partial information
        - Retry in background if appropriate

### Information Management

1. **Organization Structure**:

    - **Primary Categories**:

        - Technical (code-related)
        - Process (workflows)
        - Preferences (user-specific)
        - Context (project-specific)

    - **Sub-categories**:

        - Language-specific
        - Tool-specific
        - Style-specific
        - Environment-specific

    - **Relationship Types**:

        - Supersedes: New info replacing old
        - Complements: Additional info
        - Conflicts: Contradictory info
        - Depends: Dependencies between info

    - **Priority Levels**:
        - Critical: Must follow
        - Strong: Should follow
        - Flexible: Guidelines
        - Historical: Context only

2. **Version Control**:
    - Tag information with timestamps
    - Maintain history of changes
    - Track superseded information
    - Link related updates

### Storing and Updating Information

1. **New Information Protocol**:

    - Search for related existing information
    - Check for conflicts or redundancies
    - Split complex information into logical chunks
    - Add appropriate metadata and tags
    - Verify successful storage

2. **Update Protocol**:

    - Search ALL potentially relevant information
    - Identify outdated or incorrect data
    - Create update relationships
    - Maintain history trail
    - Verify consistency after update

3. **Error Handling**:
    - Log failed operations
    - Implement retry logic
    - Cache updates locally if needed
    - Notify user of issues
    - Provide fallback options

## Operational Guidelines

### During Work

1. **Proactive Context Management**:

    - Monitor for implicit preferences
    - Track recurring patterns
    - Suggest documentation of common practices
    - Maintain project context awareness
    - Cross-reference related information

2. **Verification and Feedback**:

    - Confirm understanding of preferences
    - Verify application of procedures
    - Check for conflicting information
    - Seek clarification when needed
    - Keep user informed of decisions

3. **Error Recovery**:
    - Handle API failures gracefully
    - Provide partial results when appropriate
    - Maintain operation queue for retries
    - Log issues for debugging
    - Offer alternative approaches

### Best Practices

1. **Search Strategy**:

    - Start with specific searches
    - Broaden scope if needed
    - Use multiple search terms
    - Combine node and fact searches
    - Center searches on relevant nodes

2. **Update Strategy**:

    - Validate before updating
    - Maintain atomic updates
    - Preserve update history
    - Handle conflicts explicitly
    - Verify after updating

3. **Context Maintenance**:

    - Track conversation flow
    - Maintain session context
    - Link related discussions
    - Record decision rationale
    - Document assumptions

4. **Quality Assurance**:
    - Regular consistency checks
    - Periodic cleanup of outdated info
    - Validation of relationships
    - User feedback integration
    - Performance monitoring

## Special Considerations

### API Dependency Management

-   Implement timeouts for all external calls
-   Cache frequently accessed data
-   Provide degraded service modes
-   Monitor API health
-   Handle rate limiting

### Session Management

-   Track conversation context
-   Maintain state between sessions
-   Handle interruptions gracefully
-   Recover from disconnections
-   Preserve important context

### Security and Privacy

-   Respect sensitive information
-   Handle credentials appropriately
-   Maintain access controls
-   Log access patterns
-   Clean sensitive data

### After each response

-   Update your memory with the interaction details and what actions you performed and why

**Remember:** The knowledge graph is your memory. Use it consistently and intelligently to provide personalized assistance that respects the user's established preferences, procedures, and factual context. Always prioritize data quality and user experience over speed of operations.
