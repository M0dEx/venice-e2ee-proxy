#[test]
fn cargo_integration_test_harness_runs() {
    assert_eq!(env!("CARGO_PKG_NAME"), "venice-e2ee-proxy");
}
