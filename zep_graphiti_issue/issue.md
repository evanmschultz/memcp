# Fix o3-mini model compatibility in MCP server

## Description

When using OpenAI's o3-mini model in the Graphiti MCP server, the API call fails with the following error:

```
ERROR - Error processing episode 'Project Description' for graph_id graph_51dbc653: Error code: 400 - {'error': {'message': "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.", 'type': 'invalid_request_error', 'param': 'max_tokens', 'code': 'unsupported_parameter'}}
```

This occurs because the o-series models (o1, o1-mini, o3-mini) introduced by OpenAI have deprecated the `max_tokens` parameter and require using `max_completion_tokens` instead. The current implementation in the OpenAI client classes in Graphiti is not model-aware and uses `max_tokens` for all models.

## Steps to reproduce

1. Configure the MCP server to use o3-mini as the model
2. Attempt to process an episode
3. Observe the error in the logs

## Expected behavior

The MCP server should properly handle o3-mini requests by using `max_completion_tokens` instead of `max_tokens`.

## Proposed solution

Implement a model-aware parameter selection approach in the OpenAI client classes. The solution should:

1. Detect when an o-series model is being used
2. Use `max_completion_tokens` instead of `max_tokens` for these models
3. Continue using `max_tokens` for non-o-series models for backward compatibility

Additionally, other unsupported parameters for o-series models should be handled appropriately:

-   temperature
-   top_p
-   presence_penalty
-   frequency_penalty
-   logprobs
-   top_logprobs
-   logit_bias

## Related documentation

-   [Azure OpenAI reasoning models documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/reasoning) states that o-series models "will only work with the max_completion_tokens parameter"
-   Multiple GitHub issues across different projects indicate this is a common problem with the new o3-mini model
