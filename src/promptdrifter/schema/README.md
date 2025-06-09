# Schema Directory

This directory contains the JSON schema definitions and validation logic for PromptDrifter configurations.

## Directory Structure

- `constants.py` - Version configuration using Pydantic
- `validator.py` - Schema validation functions
- `update_symlinks.py` - Utility to update the `latest` symlink
- `v0.1/` - Schema files for version 0.1
- `latest/` - Symlink to the current version

## Working with Schemas

### Validating a Configuration

Import `validate_config` from `promptdrifter.schema.validator` to validate configurations against the current or specific schema version.

### Adding a New Schema Version

1. Choose a new version number (e.g., "0.2")
2. Create a new directory in `schema/` with the version name
3. Copy and modify the schema files from the previous version
4. Update `supported_versions` and `current_version` in `constants.py`
5. Run the update_symlinks script
6. Update Pydantic models in `models/config.py`
7. Add migration logic for major version changes
8. Update documentation

### Schema Version Compatibility

- Minor changes should be backward compatible
- Major changes may break backward compatibility
- Always document changes between versions

## Current Schema Versions

### v0.1

Initial schema version with support for:
- Multiple adapter configurations
- Test case definitions
- Expectations (exact match, regex, substring)
- Basic adapter settings (model, temperature, max_tokens)
- Optional system prompts for adapters

## Schema Validation Process

JSON Schema validation (structural) and Pydantic model validation (runtime) are both required for a configuration to be considered valid.

## Development Notes

- Update both JSON schema and Pydantic models when making changes
- Run tests after schema changes
- Document changes in this README and the contribution guide
