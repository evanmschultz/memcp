```[graphiti]server_description
    group_id            = ""                      # Optional, will be generated if not provided
    model_name          = ""                      # Optional
    neo4j_password      = "password"
    neo4j_uri           = "bolt://localhost:7687"
    neo4j_user          = "neo4j"
    openai_api_key      = ""                      # Set via environment variable
    openai_base_url     = ""                      # Optional
    use_custom_entities = false
```

-   update the above to not use the graphiti name, use memcp
-   change group_id to something like server_name or better
-   remove api keys from toml they should only be in the .env file
-   should the neo4j password be allowed here? should the name and uri be allowed? or should they be set in the .env file?
-   What is the use_custom_entities again? I forgot and it seems like it was a mistake to add it to the project.
