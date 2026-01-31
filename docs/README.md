# Documentation

This directory contains comprehensive documentation for the Reflector Position Optimization project.

## 📚 Documentation Structure

### 📖 Guides (Getting Started)
User-focused guides for installation, usage, and quick reference:

- **[INSTALL.md](guides/INSTALL.md)** - Installation instructions and dependency setup
- **[USAGE.md](guides/USAGE.md)** - Detailed usage guide with examples
- **[QUICKREF.md](guides/QUICKREF.md)** - Quick reference for common tasks

### 🏗️ Architecture (Project Structure)
Technical documentation about the codebase:

- **[PROJECT_STRUCTURE.md](architecture/PROJECT_STRUCTURE.md)** - Package organization and module descriptions
- **[CHANGELOG.md](architecture/CHANGELOG.md)** - Version history and changes
- **[INTEGRATION_SUMMARY.md](architecture/INTEGRATION_SUMMARY.md)** - History of Ray architecture integration
- **[UPDATE_SUMMARY.md](architecture/UPDATE_SUMMARY.md)** - Dependency and documentation updates

### 🔬 Methodology (Research & Algorithms)
In-depth explanations of optimization approaches and research methodology:

- **[OPTIMIZATION_WORKFLOW.md](methodology/OPTIMIZATION_WORKFLOW.md)** - Ray-based distributed optimization architecture for physical object placement
- **[RAY_ARCHITECTURE.md](methodology/RAY_ARCHITECTURE.md)** - Why Ray is needed for reflector optimization (vs vectorization)
- **[BASELINES.md](methodology/BASELINES.md)** - Baseline comparison methods (GA, PSO, AO) for benchmarking
- **[FUTURE_ROADMAP.md](methodology/FUTURE_ROADMAP.md)** - Planned features and research extensions

## 🚀 Quick Navigation

### I want to...
- **Install the package** → [guides/INSTALL.md](guides/INSTALL.md)
- **Run an optimization** → [guides/USAGE.md](guides/USAGE.md)
- **Understand the code structure** → [architecture/PROJECT_STRUCTURE.md](architecture/PROJECT_STRUCTURE.md)
- **Learn about the optimization algorithm** → [methodology/OPTIMIZATION_WORKFLOW.md](methodology/OPTIMIZATION_WORKFLOW.md)
- **Compare with other methods** → [methodology/BASELINES.md](methodology/BASELINES.md)
- **See future plans** → [methodology/FUTURE_ROADMAP.md](methodology/FUTURE_ROADMAP.md)

## 📊 For Researchers

If you're using this framework for research:

1. **Start with**: [OPTIMIZATION_WORKFLOW.md](methodology/OPTIMIZATION_WORKFLOW.md) to understand the Ray-based distributed optimization approach
2. **Then read**: [BASELINES.md](methodology/BASELINES.md) for recommended comparison methods
3. **Check**: [FUTURE_ROADMAP.md](methodology/FUTURE_ROADMAP.md) for advanced features

## 🛠️ For Developers

If you're contributing to the codebase:

1. **Start with**: [PROJECT_STRUCTURE.md](architecture/PROJECT_STRUCTURE.md) to understand the architecture
2. **Then read**: [INSTALL.md](guides/INSTALL.md) for development setup
3. **Check**: [CHANGELOG.md](architecture/CHANGELOG.md) before making changes

## 📝 For Quick Users

If you just want to run optimizations:

1. **Start with**: [QUICKREF.md](guides/QUICKREF.md) for common commands
2. **Then read**: [USAGE.md](guides/USAGE.md) for detailed examples

## Main Documentation

See the main [README.md](../README.md) in the repository root for:
- Project overview and features
- Quick start guide
- Python API examples
- Current status and TODO list

## Project Status

See [STATUS.md](../STATUS.md) in the repository root for:
- Development roadmap
- Completed features
- In-progress tasks
- Known issues
- Performance benchmarks

## Examples

See the [examples/](../examples/) directory for:
- `quick_test.py` - Fast gradient descent test
- `full_comparison.py` - Compare grid search vs gradient descent
- `config_example.py` - Configuration template

## Need Help?

1. Check [QUICKREF.md](QUICKREF.md) for common commands
2. Read [USAGE.md](USAGE.md) for detailed examples
3. See [INSTALL.md](INSTALL.md) for installation issues
4. Run `reflector-optimize --help` for CLI help
