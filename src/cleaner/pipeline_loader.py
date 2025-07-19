import yaml
from .rule_registry import registry

def load_pipeline_config(path="src/cleaner/pipeline.yml"):
    """Load pipeline configuration from YAML file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config["pipeline"]

def run_pipeline(df, config_path="src/cleaner/pipeline.yml"):
    """Run a pipeline using the specified config file."""
    rule_names = load_pipeline_config(config_path)
    return registry.apply_chain(df, *rule_names)

def list_available_pipelines():
    """List all available pipeline configurations."""
    # TODO: Scan for multiple pipeline config files
    return ["default"]

def validate_pipeline_config(config_path="src/cleaner/pipeline.yml"):
    """Validate that all rules in the pipeline config exist."""
    try:
        rule_names = load_pipeline_config(config_path)
        available_rules = registry.list_rules()
        
        missing_rules = [rule for rule in rule_names if rule not in available_rules]
        if missing_rules:
            raise ValueError(f"Missing rules: {missing_rules}")
        
        return True
    except Exception as e:
        print(f"Pipeline validation failed: {e}")
        return False 