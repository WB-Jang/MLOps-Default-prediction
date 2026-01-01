# Contributing to Loan Default Prediction MLOps

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Poetry (optional for local development)
- Git

### Local Development

1. **Clone the repository**

```bash
git clone https://github.com/WB-Jang/MLOps-Default-prediction.git
cd MLOps-Default-prediction
```

2. **Install dependencies**

```bash
poetry install
```

3. **Run services**

```bash
make start
```

## Code Style

We follow Python best practices:

- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting

Format your code before committing:

```bash
make format
make lint
```

## Project Structure

```
src/
  ├── api/          # FastAPI application
  ├── data/         # Database and data utilities
  ├── models/       # Model architectures and training
  └── utils/        # Utility functions

airflow/
  └── dags/         # Airflow DAG definitions

config/           # Configuration files
tests/            # Test files
```

## Adding New Features

### Adding a New Model

1. Create model architecture in `src/models/`
2. Update training logic in `src/models/training.py`
3. Add tests in `tests/`
4. Update documentation

### Adding a New DAG

1. Create DAG file in `airflow/dags/`
2. Follow naming convention: `*_dag.py`
3. Add proper documentation
4. Test locally before committing

### Adding API Endpoints

1. Add endpoint in `src/api/main.py`
2. Create Pydantic models for request/response
3. Add tests
4. Update API documentation

## Testing

Run tests before submitting:

```bash
make test
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Format and lint code (`make format && make lint`)
5. Run tests (`make test`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your fork (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Example:
```
feat: add batch prediction endpoint
fix: correct F1-score calculation in monitoring
docs: update README with new API endpoints
```

## Questions?

Open an issue for questions or discussions.
