# Contributing to Nano-vLLM-RS

Thank you for your interest in contributing to Nano-vLLM-RS! This project is maintained by [Sai Sunkara](https://github.com/ssvgopal). We appreciate your time and effort in helping us improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/nano-vllm-rs.git
   cd nano-vllm-rs
   ```
3. Set up the development environment:
   ```bash
   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Install Rust toolchain
   rustup install stable
   rustup default stable
   
   # Install development tools
   rustup component add rustfmt clippy
   cargo install cargo-tarpaulin
   ```

## Development Workflow

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-short-description
   ```

2. Make your changes following the coding standards

3. Run tests and verify your changes:
   ```bash
   cargo test --all-features
   cargo clippy --all-targets --all-features -- -D warnings
   cargo fmt --all -- --check
   ```

4. Commit your changes with a descriptive commit message:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # or
   git commit -m "fix: resolve issue with model loading"
   ```

5. Push your changes to your fork:
   ```bash
   git push origin your-branch-name
   ```

6. Open a pull request against the main branch

## Coding Standards

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for consistent code formatting
- Run `cargo clippy` to catch common mistakes and improve code quality
- Document all public APIs with Rustdoc comments
- Write unit tests for new functionality
- Keep commits focused and atomic
- Write clear, concise commit messages following the [Conventional Commits](https://www.conventionalcommits.org/) specification

## Testing

We use a comprehensive testing strategy:

1. **Unit Tests**: Test individual functions and modules
   ```bash
   cargo test --lib
   ```

2. **Integration Tests**: Test the public API
   ```bash
   cargo test --test integration
   ```

3. **Documentation Tests**: Test code examples in documentation
   ```bash
   cargo test --doc
   ```

4. **Benchmarks**: Performance testing
   ```bash
   cargo bench
   ```

5. **Code Coverage** (requires `cargo-tarpaulin`):
   ```bash
   cargo tarpaulin --out Html
   ```

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Ensure tests pass and coverage remains high
3. Open a pull request with a clear description of changes
4. Reference any related issues or pull requests
5. You may merge the pull request once you have the sign-off of at least one maintainer

## Reporting Bugs

Please use the [GitHub issue tracker](https://github.com/ssvgopal/nano-vllm-rs/issues) to report bugs. Include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Environment details (OS, Rust version, etc.)
- Any relevant logs or error messages

## Feature Requests

We welcome feature requests! Please:

1. Check if a similar feature request already exists
2. Clearly describe the problem you're trying to solve
3. Explain why this feature would be valuable
4. Provide any additional context or examples

## License

By contributing to Nano-vLLM-RS, you agree that your contributions will be licensed under the [MIT License](LICENSE).
