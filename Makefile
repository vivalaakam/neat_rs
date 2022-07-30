test:
	cargo check
	cargo test
	cargo fmt --all -- --check
