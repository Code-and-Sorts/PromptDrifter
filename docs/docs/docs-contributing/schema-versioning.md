# Schema Versioning Guide

This document provides detailed information about the schema versioning system used in PromptDrifter. For basic contribution guidelines, please refer to the [general contributing guide](welcome.md).

## Schema Overview

PromptDrifter uses a versioned schema system to validate configuration files. The schema is defined in JSON format and used to validate YAML configuration files. This ensures that configuration files are valid and consistent, while allowing for future evolution of the schema.

## Schema Components

The schema system consists of several components:

1. **JSON Schema Files**: Formal schema definitions in JSON format
2. **Pydantic Models**: Python classes that mirror the schema structure
3. **Version Management**: Tools to manage schema versions
4. **Validation Logic**: Code that validates configurations against schemas

## Directory Structure

```
src/promptdrifter/
  ├── schema/                     # Schema-related code
  │   ├── constants.py            # Version definitions
  │   ├── validator.py            # Schema validation logic
  │   ├── migration.py            # Migration tools (for future)
  │   ├── update_symlinks.py      # Utility to update symlinks
  │   ├── v0.1/                   # Version-specific schemas
  │   │   ├── schema.json         # JSON Schema definition
  │   │   └── sample.yaml         # Example configuration
  │   └── latest/                 # Symlink to current version
  └── models/                     # Pydantic models
      └── config.py               # Configuration models
```

## Version Management

Schema versions are managed through the `SchemaVersions` Pydantic model in `constants.py`. This model defines the current version, supported versions, and provides a computed field for the latest version.

To add a new schema version:

1. Update the `supported_versions` list in `SchemaVersions`
2. Set `current_version` to the new version
3. Create a new version directory with schema files
4. Run the `update_symlinks.py` script to update symlinks

## Validation Process

Configuration validation follows these steps:

1. YAML file is loaded using `yaml.safe_load()`
2. The loaded data is validated against the JSON schema
3. The data is then converted to a Pydantic model for runtime type checking
4. Both validations must pass for the configuration to be considered valid

## Making Schema Changes

### Minor Changes (Backward Compatible)

For changes that don't break existing configurations:

1. Update the JSON schema file with the new fields or modifications
2. Update the corresponding Pydantic models in `models/config.py`
3. Add tests covering valid configurations, edge cases, and optional fields
4. Document the changes

Examples: Adding optional fields, extending enums with new values, adding new options to existing fields.

### Major Changes (Backward Incompatible)

For changes that break existing configurations:

1. Create a new version directory (e.g., `schema/v0.2/`)
2. Copy and modify the schema files from the previous version
3. Update the `SchemaVersions` model to include the new version
4. Implement migration functions to help users upgrade
5. Update symlinks and Pydantic models
6. Add comprehensive tests and documentation

Examples: Removing required fields, changing field types, renaming fields, restructuring the schema.

## Migration Approach

When creating a new schema version with breaking changes:

1. Implement migration functions in `migration.py` to transform old configurations to the new format
2. Create CLI commands to help users migrate their configurations
3. Document the migration process clearly with examples

## Testing Strategy

For schema changes, implement tests for:

1. **Validation**: Ensure valid configurations pass validation
2. **Migration**: Verify migrations correctly transform configurations
3. **Error Handling**: Confirm invalid configurations fail with appropriate errors

## Best Practices

1. Follow semantic versioning principles
2. Maintain backward compatibility whenever possible
3. Document all changes clearly
4. Provide migration tools for major changes
5. Test thoroughly before releasing
6. Keep the schema as simple as possible while meeting requirements
7. Include examples with each schema version
