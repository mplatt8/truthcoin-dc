//! Performance benchmarks for Bitcoin Hivemind Phase 1 voting system
//!
//! These benchmarks measure the performance of critical voting operations
//! to ensure the system can handle realistic loads as specified in the
//! Bitcoin Hivemind whitepaper.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use truthcoin_dc::{
    state::{
        State,
        voting::{
            VotingSystem,
            types::{VotingPeriod, VotingPeriodId, VotingPeriodStatus, Vote, VoteValue, VoterId},
        },
        slots::SlotId,
    },
    types::Address,
    math::voting::{SparseVoteMatrix, ReputationVector, VoteAggregator, MatrixUtils},
};
use sneed::Env;
use tempfile::TempDir;
use std::collections::HashMap;

/// Benchmark environment setup
struct BenchEnv {
    _temp_dir: TempDir,
    state: State,
}

impl BenchEnv {
    fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let env = Env::new(&temp_dir, 1024 * 1024 * 500).expect("Failed to create environment");
        let mut rwtxn = env.rw_txn().expect("Failed to create transaction");
        let state = State::new(&env, &mut rwtxn).expect("Failed to create state");
        rwtxn.commit().expect("Failed to commit transaction");

        Self {
            _temp_dir: temp_dir,
            state,
        }
    }
}

// Helper functions for creating test data
fn create_test_voter_ids(count: usize) -> Vec<VoterId> {
    (0..count)
        .map(|i| {
            let mut addr_bytes = [0u8; 20];
            addr_bytes[0] = (i & 0xFF) as u8;
            addr_bytes[1] = ((i >> 8) & 0xFF) as u8;
            addr_bytes[2] = ((i >> 16) & 0xFF) as u8;
            VoterId::from_address(&Address(addr_bytes))
        })
        .collect()
}

fn create_test_decision_ids(count: usize) -> Vec<SlotId> {
    (0..count)
        .map(|i| SlotId::new(1, i as u16).unwrap())
        .collect()
}

fn create_test_tx_hash(seed: u32) -> [u8; 32] {
    let mut hash = [0u8; 32];
    hash[0] = (seed & 0xFF) as u8;
    hash[1] = ((seed >> 8) & 0xFF) as u8;
    hash[2] = ((seed >> 16) & 0xFF) as u8;
    hash[3] = ((seed >> 24) & 0xFF) as u8;
    hash
}

/// Benchmark voting period creation and management
fn bench_voting_period_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("voting_period_operations");

    // Test different numbers of decisions per period
    for num_decisions in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("create_period", num_decisions),
            num_decisions,
            |b, &num_decisions| {
                let env = BenchEnv::new();
                let decisions = create_test_decision_ids(num_decisions);

                b.iter(|| {
                    let mut rwtxn = env.state.rw_txn().unwrap();
                    let period_id = VotingPeriodId::new(rand::random::<u32>());

                    env.state.voting_system().create_voting_period(
                        &mut rwtxn,
                        period_id,
                        1000,
                        2000,
                        decisions.clone(),
                        100,
                    ).unwrap();

                    rwtxn.commit().unwrap();
                });
            },
        );
    }

    // Benchmark period status updates
    group.bench_function("period_status_updates", |b| {
        let env = BenchEnv::new();
        let decisions = create_test_decision_ids(100);

        // Pre-create periods
        let period_ids: Vec<_> = (0..100)
            .map(|i| {
                let period_id = VotingPeriodId::new(i);
                let mut rwtxn = env.state.rw_txn().unwrap();
                env.state.voting_system().create_voting_period(
                    &mut rwtxn,
                    period_id,
                    1000 + i as u64 * 10000,
                    2000 + i as u64 * 10000,
                    decisions.clone(),
                    100 + i,
                ).unwrap();
                rwtxn.commit().unwrap();
                period_id
            })
            .collect();

        b.iter(|| {
            let rotxn = env.state.ro_txn().unwrap();
            let _updates = env.state.voting_system()
                .get_periods_needing_update(&rotxn, 1500).unwrap();
            rotxn.commit().unwrap();
        });
    });

    group.finish();
}

/// Benchmark vote casting operations
fn bench_vote_casting(c: &mut Criterion) {
    let mut group = c.benchmark_group("vote_casting");

    // Test batch vote casting with different scales
    for num_votes in [100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*num_votes as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_vote_casting", num_votes),
            num_votes,
            |b, &num_votes| {
                let env = BenchEnv::new();
                let period_id = VotingPeriodId::new(1);
                let voters = create_test_voter_ids(num_votes / 10); // 10 votes per voter
                let decisions = create_test_decision_ids(10);

                // Setup period
                {
                    let mut rwtxn = env.state.rw_txn().unwrap();
                    env.state.voting_system().create_voting_period(
                        &mut rwtxn, period_id, 1000, 2000, decisions.clone(), 100
                    ).unwrap();
                    env.state.voting_system().activate_voting_period(&mut rwtxn, period_id, 1000).unwrap();
                    rwtxn.commit().unwrap();
                }

                b.iter(|| {
                    let mut rwtxn = env.state.rw_txn().unwrap();

                    for i in 0..num_votes {
                        let voter = voters[i / 10];
                        let decision = decisions[i % 10];
                        let vote_value = if i % 3 == 0 {
                            VoteValue::Binary(i % 2 == 0)
                        } else if i % 3 == 1 {
                            VoteValue::Scalar((i % 100) as f64 / 100.0)
                        } else {
                            VoteValue::Abstain
                        };

                        env.state.voting_system().cast_vote(
                            &mut rwtxn,
                            voter,
                            period_id,
                            decision,
                            vote_value,
                            1500 + i as u64,
                            200 + i as u64,
                            create_test_tx_hash(i as u32),
                        ).unwrap();
                    }

                    rwtxn.commit().unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark vote retrieval and matrix operations
fn bench_vote_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("vote_retrieval");

    // Setup test data
    let env = BenchEnv::new();
    let period_id = VotingPeriodId::new(1);
    let voters = create_test_voter_ids(1000);
    let decisions = create_test_decision_ids(100);

    // Pre-populate with votes
    {
        let mut rwtxn = env.state.rw_txn().unwrap();

        env.state.voting_system().create_voting_period(
            &mut rwtxn, period_id, 1000, 2000, decisions.clone(), 100
        ).unwrap();
        env.state.voting_system().activate_voting_period(&mut rwtxn, period_id, 1000).unwrap();

        // Create sparse voting pattern (30% density)
        let mut vote_counter = 0u32;
        for (i, voter) in voters.iter().enumerate() {
            for (j, decision) in decisions.iter().enumerate() {
                if (i + j) % 3 == 0 { // 33% participation
                    let vote_value = if (i + j) % 2 == 0 {
                        VoteValue::Binary(true)
                    } else {
                        VoteValue::Scalar((i + j) as f64 / 200.0)
                    };

                    env.state.voting_system().cast_vote(
                        &mut rwtxn,
                        *voter,
                        period_id,
                        *decision,
                        vote_value,
                        1500 + vote_counter as u64,
                        200 + vote_counter as u64,
                        create_test_tx_hash(vote_counter),
                    ).unwrap();

                    vote_counter += 1;
                }
            }
        }

        rwtxn.commit().unwrap();
    }

    // Benchmark vote retrieval for entire period
    group.bench_function("get_all_votes_for_period", |b| {
        b.iter(|| {
            let rotxn = env.state.ro_txn().unwrap();
            let _votes = env.state.voting_system().get_votes_for_period(&rotxn, period_id).unwrap();
            rotxn.commit().unwrap();
        });
    });

    // Benchmark vote matrix generation
    group.bench_function("generate_vote_matrix", |b| {
        b.iter(|| {
            let rotxn = env.state.ro_txn().unwrap();
            let _matrix = env.state.voting_system().get_vote_matrix(&rotxn, period_id).unwrap();
            rotxn.commit().unwrap();
        });
    });

    // Benchmark participation statistics
    group.bench_function("calculate_participation_stats", |b| {
        b.iter(|| {
            let rotxn = env.state.ro_txn().unwrap();
            let _stats = env.state.voting_system().get_participation_stats(&rotxn, period_id).unwrap();
            rotxn.commit().unwrap();
        });
    });

    group.finish();
}

/// Benchmark reputation management operations
fn bench_reputation_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("reputation_operations");

    // Test reputation initialization for different numbers of voters
    for num_voters in [100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*num_voters as u64));
        group.bench_with_input(
            BenchmarkId::new("reputation_initialization", num_voters),
            num_voters,
            |b, &num_voters| {
                let env = BenchEnv::new();
                let voters = create_test_voter_ids(num_voters);
                let period_id = VotingPeriodId::new(1);

                b.iter(|| {
                    let mut rwtxn = env.state.rw_txn().unwrap();

                    for (i, voter) in voters.iter().enumerate() {
                        env.state.voting_system().initialize_voter_reputation(
                            &mut rwtxn,
                            *voter,
                            0.3 + (i as f64 / num_voters as f64 * 0.4),
                            1000,
                            period_id,
                        ).unwrap();
                    }

                    rwtxn.commit().unwrap();
                });
            },
        );
    }

    // Benchmark reputation updates
    let env = BenchEnv::new();
    let voters = create_test_voter_ids(1000);
    let period_id = VotingPeriodId::new(1);

    // Pre-initialize reputations
    {
        let mut rwtxn = env.state.rw_txn().unwrap();
        for (i, voter) in voters.iter().enumerate() {
            env.state.voting_system().initialize_voter_reputation(
                &mut rwtxn, *voter, 0.5, 1000, period_id
            ).unwrap();
        }
        rwtxn.commit().unwrap();
    }

    group.bench_function("reputation_updates", |b| {
        b.iter(|| {
            let mut rwtxn = env.state.rw_txn().unwrap();

            for (i, voter) in voters.iter().enumerate() {
                env.state.voting_system().update_voter_reputation(
                    &mut rwtxn,
                    *voter,
                    i % 2 == 0, // Alternating correct/incorrect
                    2000,
                    VotingPeriodId::new(2),
                ).unwrap();
            }

            rwtxn.commit().unwrap();
        });
    });

    group.bench_function("get_reputation_weights", |b| {
        b.iter(|| {
            let rotxn = env.state.ro_txn().unwrap();
            let _weights = env.state.voting_system().get_reputation_weights(&rotxn, period_id).unwrap();
            rotxn.commit().unwrap();
        });
    });

    group.finish();
}

/// Benchmark sparse matrix operations
fn bench_sparse_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matrix_operations");

    // Test matrix creation and manipulation with different sizes
    for &(num_voters, num_decisions) in [(100, 50), (500, 100), (1000, 200), (2000, 500)].iter() {
        let voters = create_test_voter_ids(num_voters);
        let decisions = create_test_decision_ids(num_decisions);

        group.bench_with_input(
            BenchmarkId::new("matrix_creation", format!("{}x{}", num_voters, num_decisions)),
            &(voters.clone(), decisions.clone()),
            |b, (voters, decisions)| {
                b.iter(|| {
                    SparseVoteMatrix::new(voters.clone(), decisions.clone())
                });
            },
        );

        // Benchmark sparse matrix filling
        group.bench_with_input(
            BenchmarkId::new("matrix_filling", format!("{}x{}", num_voters, num_decisions)),
            &(voters.clone(), decisions.clone()),
            |b, (voters, decisions)| {
                b.iter(|| {
                    let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());

                    // Fill with 20% density
                    for (i, voter) in voters.iter().enumerate() {
                        for (j, decision) in decisions.iter().enumerate() {
                            if (i + j) % 5 == 0 {
                                let value = if (i + j) % 2 == 0 { 1.0 } else { 0.0 };
                                matrix.set_vote(*voter, *decision, value).unwrap();
                            }
                        }
                    }
                });
            },
        );

        // Benchmark dense conversion
        let mut matrix = SparseVoteMatrix::new(voters.clone(), decisions.clone());
        for (i, voter) in voters.iter().enumerate() {
            for (j, decision) in decisions.iter().enumerate() {
                if (i + j) % 5 == 0 {
                    let value = if (i + j) % 2 == 0 { 1.0 } else { 0.0 };
                    matrix.set_vote(*voter, *decision, value).unwrap();
                }
            }
        }

        group.bench_with_input(
            BenchmarkId::new("dense_conversion", format!("{}x{}", num_voters, num_decisions)),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    matrix.to_dense(0.5)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark vote aggregation algorithms
fn bench_vote_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vote_aggregation");

    // Test aggregation with different numbers of votes
    for num_votes in [100, 500, 1000, 5000].iter() {
        let voters = create_test_voter_ids(*num_votes);
        let mut votes = HashMap::new();
        let mut reputation = ReputationVector::new();

        // Setup test data
        for (i, voter) in voters.iter().enumerate() {
            let vote_value = (i as f64 / *num_votes as f64).clamp(0.0, 1.0);
            votes.insert(*voter, vote_value);

            let rep_value = 0.3 + (i as f64 / *num_votes as f64 * 0.4);
            reputation.set_reputation(*voter, rep_value);
        }

        group.throughput(Throughput::Elements(*num_votes as u64));

        group.bench_with_input(
            BenchmarkId::new("simple_majority", num_votes),
            &votes,
            |b, votes| {
                b.iter(|| {
                    VoteAggregator::simple_majority(votes)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("weighted_average", num_votes),
            &(&votes, &reputation),
            |b, (votes, reputation)| {
                b.iter(|| {
                    VoteAggregator::weighted_average(votes, reputation)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("median_vote", num_votes),
            &votes,
            |b, votes| {
                b.iter(|| {
                    VoteAggregator::median_vote(votes)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("calculate_confidence", num_votes),
            &votes,
            |b, votes| {
                b.iter(|| {
                    VoteAggregator::calculate_confidence(votes, 0.7)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark matrix utility functions
fn bench_matrix_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_utils");

    // Create large test matrix
    let num_voters = 1000;
    let num_decisions = 200;
    let voters = create_test_voter_ids(num_voters);
    let decisions = create_test_decision_ids(num_decisions);
    let mut matrix = SparseVoteMatrix::new(voters, decisions);

    // Fill matrix with realistic density (25%)
    for i in 0..num_voters {
        for j in 0..num_decisions {
            if (i + j) % 4 == 0 {
                let voter = create_test_voter_ids(1)[0];
                let decision = create_test_decision_ids(1)[0];
                let vote_value = (i + j) as f64 / (num_voters + num_decisions) as f64;
                // Use actual matrix voters/decisions instead of creating new ones
                if let (Some(&actual_voter), Some(&actual_decision)) = (
                    matrix.get_voters().get(i % matrix.get_voters().len()),
                    matrix.get_decisions().get(j % matrix.get_decisions().len())
                ) {
                    matrix.set_vote(actual_voter, actual_decision, vote_value).unwrap();
                }
            }
        }
    }

    group.bench_function("participation_rates", |b| {
        b.iter(|| {
            MatrixUtils::calculate_participation_rates(&matrix)
        });
    });

    group.bench_function("voter_activity", |b| {
        b.iter(|| {
            MatrixUtils::calculate_voter_activity(&matrix)
        });
    });

    group.bench_function("outlier_detection", |b| {
        b.iter(|| {
            MatrixUtils::find_outlier_voters(&matrix, 2.0).unwrap()
        });
    });

    group.finish();
}

/// Benchmark decision outcome resolution
fn bench_decision_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_resolution");

    // Test resolution with different scales
    for num_decisions in [10, 50, 100, 500].iter() {
        let env = BenchEnv::new();
        let period_id = VotingPeriodId::new(1);
        let voters = create_test_voter_ids(100);
        let decisions = create_test_decision_ids(*num_decisions);

        // Setup period with votes
        {
            let mut rwtxn = env.state.rw_txn().unwrap();

            env.state.voting_system().create_voting_period(
                &mut rwtxn, period_id, 1000, 2000, decisions.clone(), 100
            ).unwrap();
            env.state.voting_system().activate_voting_period(&mut rwtxn, period_id, 1000).unwrap();

            // Initialize reputations
            for (i, voter) in voters.iter().enumerate() {
                env.state.voting_system().initialize_voter_reputation(
                    &mut rwtxn, *voter, 0.3 + (i as f64 / 100.0 * 0.4), 1000, period_id
                ).unwrap();
            }

            // Cast votes
            let mut vote_counter = 0u32;
            for voter in &voters {
                for decision in &decisions {
                    if vote_counter % 3 == 0 { // 33% participation
                        let vote_value = if vote_counter % 2 == 0 {
                            VoteValue::Binary(true)
                        } else {
                            VoteValue::Scalar((vote_counter % 100) as f64 / 100.0)
                        };

                        env.state.voting_system().cast_vote(
                            &mut rwtxn,
                            *voter,
                            period_id,
                            *decision,
                            vote_value,
                            1500 + vote_counter as u64,
                            200 + vote_counter as u64,
                            create_test_tx_hash(vote_counter),
                        ).unwrap();
                    }
                    vote_counter += 1;
                }
            }

            env.state.voting_system().close_voting_period(&mut rwtxn, period_id, 2000).unwrap();
            rwtxn.commit().unwrap();
        }

        group.throughput(Throughput::Elements(*num_decisions as u64));
        group.bench_with_input(
            BenchmarkId::new("resolve_period_decisions", num_decisions),
            num_decisions,
            |b, _| {
                b.iter(|| {
                    let mut rwtxn = env.state.rw_txn().unwrap();
                    let _outcomes = env.state.voting_system().resolve_period_decisions(
                        &mut rwtxn, period_id, 2500, 300
                    ).unwrap();
                    rwtxn.abort(); // Don't commit to allow re-running
                });
            },
        );
    }

    group.finish();
}

/// Benchmark system-wide statistics calculation
fn bench_system_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("system_statistics");

    // Setup large system with multiple periods
    let env = BenchEnv::new();
    let voters = create_test_voter_ids(500);
    let decisions_per_period = 50;

    // Create multiple periods with data
    for period_num in 1..=5 {
        let period_id = VotingPeriodId::new(period_num);
        let decisions = create_test_decision_ids(decisions_per_period);

        let mut rwtxn = env.state.rw_txn().unwrap();

        env.state.voting_system().create_voting_period(
            &mut rwtxn,
            period_id,
            1000 + (period_num - 1) * 10000,
            2000 + (period_num - 1) * 10000,
            decisions.clone(),
            100 + period_num * 50,
        ).unwrap();

        if period_num <= 3 {
            env.state.voting_system().activate_voting_period(
                &mut rwtxn,
                period_id,
                1000 + (period_num - 1) * 10000,
            ).unwrap();
        }

        // Initialize reputations for first period
        if period_num == 1 {
            for (i, voter) in voters.iter().enumerate() {
                env.state.voting_system().initialize_voter_reputation(
                    &mut rwtxn, *voter, 0.3 + (i as f64 / 500.0 * 0.4), 1000, period_id
                ).unwrap();
            }
        }

        // Add some votes for active periods
        if period_num <= 3 {
            let mut vote_counter = 0u32;
            for (i, voter) in voters.iter().enumerate() {
                if i % 5 == 0 { // 20% voter participation
                    for (j, decision) in decisions.iter().enumerate() {
                        if j % 3 == 0 { // 33% decision participation
                            let vote_value = if (i + j) % 2 == 0 {
                                VoteValue::Binary(true)
                            } else {
                                VoteValue::Scalar((i + j) as f64 / 100.0)
                            };

                            env.state.voting_system().cast_vote(
                                &mut rwtxn,
                                *voter,
                                period_id,
                                *decision,
                                vote_value,
                                1500 + vote_counter as u64,
                                200 + vote_counter as u64,
                                create_test_tx_hash(vote_counter),
                            ).unwrap();

                            vote_counter += 1;
                        }
                    }
                }
            }
        }

        rwtxn.commit().unwrap();
    }

    group.bench_function("calculate_period_statistics", |b| {
        b.iter(|| {
            let rotxn = env.state.ro_txn().unwrap();
            let _stats = env.state.voting_system().calculate_period_statistics(
                &rotxn, VotingPeriodId::new(1), 2500
            ).unwrap();
            rotxn.commit().unwrap();
        });
    });

    group.bench_function("get_system_stats", |b| {
        b.iter(|| {
            let rotxn = env.state.ro_txn().unwrap();
            let _stats = env.state.voting_system().get_system_stats(&rotxn).unwrap();
            rotxn.commit().unwrap();
        });
    });

    group.bench_function("validate_consistency", |b| {
        b.iter(|| {
            let rotxn = env.state.ro_txn().unwrap();
            let _issues = env.state.voting_system().validate_consistency(&rotxn).unwrap();
            rotxn.commit().unwrap();
        });
    });

    group.finish();
}

/// Memory usage benchmark
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Test memory efficiency with large sparse matrices
    for &(voters, decisions, density) in [(1000, 500, 0.1), (2000, 1000, 0.05), (5000, 2000, 0.02)].iter() {
        group.bench_with_input(
            BenchmarkId::new("sparse_matrix_memory", format!("{}x{}@{}%", voters, decisions, (density * 100.0) as u8)),
            &(voters, decisions, density),
            |b, &(num_voters, num_decisions, density)| {
                b.iter(|| {
                    let voters = create_test_voter_ids(num_voters);
                    let decisions = create_test_decision_ids(num_decisions);
                    let mut matrix = SparseVoteMatrix::new(voters, decisions);

                    // Fill matrix with specified density
                    let total_cells = num_voters * num_decisions;
                    let filled_cells = (total_cells as f64 * density) as usize;

                    for i in 0..filled_cells {
                        let voter_idx = i % num_voters;
                        let decision_idx = (i / num_voters) % num_decisions;

                        if let (Some(&voter), Some(&decision)) = (
                            matrix.get_voters().get(voter_idx),
                            matrix.get_decisions().get(decision_idx)
                        ) {
                            matrix.set_vote(voter, decision, (i as f64 / filled_cells as f64)).unwrap();
                        }
                    }

                    matrix
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    voting_benchmarks,
    bench_voting_period_operations,
    bench_vote_casting,
    bench_vote_retrieval,
    bench_reputation_operations,
    bench_sparse_matrix_operations,
    bench_vote_aggregation,
    bench_matrix_utils,
    bench_decision_resolution,
    bench_system_statistics,
    bench_memory_usage
);

criterion_main!(voting_benchmarks);