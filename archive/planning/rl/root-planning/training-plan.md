# RL Training Plan: MarketSessionView Training Strategy

## Executive Summary

This document outlines the training strategy for the market-agnostic RL environment using `MarketSessionView` data with Stable Baselines3 (SB3) integration. The core challenge is transitioning from sequential single-episode training to effective multi-episode RL training that leverages the market-agnostic architecture.

## Current State Analysis

### Architecture Overview

- **MarketSessionView**: Pre-filtered single-market view from a session
  - One `MarketSessionView` = One complete episode
  - Contains all timesteps for one market within a session
  - Variable episode lengths (10-1000+ steps depending on market activity)
  - Market-agnostic features (no market identity exposed to agent)

- **MarketAgnosticKalshiEnv**: Gymnasium-compatible environment
  - Accepts `MarketSessionView` in constructor
  - Has `set_market_view()` method for curriculum learning
  - 52-feature observation space (market-agnostic)
  - 5-action discrete space (HOLD + 4 limit order actions)

- **Current Training**: `SimpleSessionCurriculum` class
  - Loads sessions and creates market views
  - Runs one episode per market with random actions
  - Tracks results but doesn't actually train an RL agent

### Key Constraints

1. **One View = One Episode**: Each `MarketSessionView` represents a complete episode
2. **Variable Lengths**: Episodes vary significantly in length
3. **Market-Agnostic**: Agent never sees market tickers or identity
4. **Historical Data Only**: Training uses pre-collected session data
5. **SB3 Integration**: Must work with PPO/A2C algorithms from Stable Baselines3

## Training Strategy: Multi-Episode Learning

### Core Principle

**Single episodes per market are insufficient for learning.** Effective RL training requires:
- Multiple episodes per market (10-100+ episodes)
- Proper replay buffer management
- Balanced sampling across markets
- Progressive difficulty (curriculum learning)

### Training Flow Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Data Collection                                │
│ - Load all available sessions                           │
│ - Create MarketSessionView for each market               │
│ - Filter by minimum episode length                      │
│ - Compute difficulty metrics                            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Environment Setup                             │
│ - Create MarketAgnosticKalshiEnv wrapper               │
│ - Implement view cycling logic                          │
│ - Configure SB3 environment interface                   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Multi-Episode Training                        │
│ - Cycle through MarketSessionViews                      │
│ - Run multiple episodes per view                        │
│ - Collect experience tuples                             │
│ - Update policy via SB3 algorithms                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Evaluation & Monitoring                       │
│ - Track performance per market                          │
│ - Monitor reward distributions                          │
│ - Validate generalization                               │
└─────────────────────────────────────────────────────────┘
```

## Training Approaches

### Approach 1: Sequential Multi-Episode Training (Baseline)

**Strategy**: Train multiple episodes per MarketSessionView, cycling through all views

**Implementation Concept**:
1. Collect all valid MarketSessionViews from all sessions
2. Create environment wrapper that cycles through views
3. For each view, run N episodes (e.g., 10-50 episodes)
4. SB3 handles replay buffer and policy updates internally

**Advantages**:
- Simple to implement
- Natural integration with SB3's training loop
- Ensures all markets get equal training time

**Considerations**:
- Need to handle variable episode lengths gracefully
- May require custom environment wrapper for view cycling
- Should balance episodes across markets to prevent overfitting

**Episode Distribution**:
- Fixed episodes per view: 10-50 episodes per MarketSessionView
- Or proportional: Episodes ∝ episode_length (longer markets = more episodes)

### Approach 2: Curriculum Learning with Progressive Difficulty

**Strategy**: Start with easy markets, gradually increase difficulty

**Difficulty Metrics** (from MarketSessionView):
- `avg_spread`: Lower spread = easier (more liquid)
- `volatility_score`: Lower volatility = easier (more stable)
- `market_coverage`: Higher coverage = easier (more data points)

**Implementation Concept**:
1. Score all MarketSessionViews by difficulty
2. Sort by difficulty (easy → hard)
3. Train in stages:
   - Stage 1: Easiest 20% of markets, 50 episodes each
   - Stage 2: Next 20%, 50 episodes each
   - ... continue until all markets trained
4. Optionally: Revisit easy markets periodically to prevent forgetting

**Advantages**:
- Better sample efficiency (learn basics before advanced)
- More stable training (avoid catastrophic forgetting)
- Natural progression matches human learning

**Considerations**:
- Need to define difficulty scoring function
- May require careful tuning of difficulty thresholds
- Should track performance per difficulty level

### Approach 3: Balanced Sampling Across Markets

**Strategy**: Ensure equal representation of all markets in training

**Implementation Concept**:
1. Track episode counts per market
2. Sample markets inversely proportional to their episode count
3. Markets with fewer episodes get prioritized
4. Continue until all markets reach target episode count

**Advantages**:
- Prevents overfitting to popular/easy markets
- Ensures generalization across all market types
- Better for market-agnostic learning goal

**Considerations**:
- Need to track sampling statistics
- May require dynamic rebalancing during training
- Should monitor for markets that are too difficult (always fail)

### Approach 4: Hybrid Strategy (RECOMMENDED)

**Combine Approaches 2 + 3**: Curriculum learning with balanced sampling

**Implementation Concept**:
1. **Initial Phase**: Curriculum learning (easy → hard)
   - Start with easiest markets
   - Gradually introduce harder markets
   - 20-50 episodes per market per stage

2. **Balanced Phase**: Balanced sampling
   - Once all markets introduced, switch to balanced sampling
   - Ensure equal representation
   - Continue training until convergence

3. **Refinement Phase**: Focus on difficult markets
   - Identify markets where agent struggles
   - Additional training episodes for difficult markets
   - Fine-tune performance

## SB3 Integration Strategy

### Environment Wrapper Design

**Challenge**: SB3 expects a single environment, but we have multiple MarketSessionViews

**Solution**: Create wrapper that cycles through views

**Wrapper Requirements**:
1. Maintain list of MarketSessionViews
2. Track current view index
3. On episode end, cycle to next view
4. Call `env.set_market_view()` before each reset
5. Handle edge cases (empty views, very short episodes)

**Interface Compatibility**:
- Must implement `gym.Env` interface
- Compatible with `DummyVecEnv` for vectorization
- Works with SB3's `learn()` method

### Training Loop Integration

**SB3 Training Flow**:
```python
# Conceptual flow (not implementation)
model = PPO("MlpPolicy", env, verbose=1)

# SB3 handles:
# - Episode collection
# - Replay buffer management
# - Policy updates
# - Value function learning
model.learn(total_timesteps=1_000_000)
```

**Our Integration Points**:
1. **Environment Creation**: Wrapper cycles through MarketSessionViews
2. **Episode Boundaries**: Wrapper handles view switching on episode end
3. **Observation Consistency**: Ensure 52-feature observation space maintained
4. **Action Space**: 5 discrete actions (already compatible)

### Algorithm Selection

**PPO (Proximal Policy Optimization)** - RECOMMENDED
- On-policy algorithm (good for stable learning)
- Handles variable episode lengths well
- Good sample efficiency
- Stable Baselines3 has excellent PPO implementation

**A2C (Advantage Actor-Critic)** - Alternative
- Simpler than PPO
- Faster training (no clipping)
- May be less stable
- Good for initial experiments

**Recommendation**: Start with PPO, experiment with A2C if needed

## Data Management Strategy

### MarketSessionView Collection

**Collection Process**:
1. Load all available sessions from database
2. For each session, create MarketSessionView for each market
3. Filter views by minimum episode length (e.g., ≥ 50 steps)
4. Compute difficulty metrics for each view
5. Store views in list/dict for training

**Filtering Criteria**:
- Minimum episode length: 50 steps (configurable)
- Maximum episode length: None (use all data)
- Market coverage: ≥ 10% of session duration
- Data quality: Use `data_quality_score` from SessionData

### Episode Length Distribution

**Current Distribution** (from session 6):
- 300 markets with varying episode lengths
- Some markets: 10-50 steps (short)
- Some markets: 500-1000+ steps (long)

**Training Strategy**:
- **Option A**: Use all markets regardless of length
  - Pros: Maximum data diversity
  - Cons: Very short episodes provide little signal

- **Option B**: Filter by minimum length
  - Pros: Better learning signal per episode
  - Cons: Loses some market diversity

- **Option C**: Group by length, train separately
  - Pros: Can optimize for different length distributions
  - Cons: More complex, may hurt generalization

**Recommendation**: Option B with minimum length = 50 steps

### Session and Market Diversity

**Current Data**:
- Multiple sessions collected over time
- Each session has different markets
- Markets vary in characteristics (spread, volatility, coverage)

**Training Considerations**:
- **Temporal Diversity**: Shuffle sessions to avoid temporal bias
- **Market Diversity**: Ensure representation across market types
- **Session Diversity**: Mix sessions from different time periods

**Sampling Strategy**:
- Shuffle MarketSessionViews before training
- Ensure no single session dominates training
- Balance markets across sessions

## Training Configuration

### Hyperparameters (PPO)

**Recommended Starting Values**:
- `learning_rate`: 3e-4 (standard for PPO)
- `n_steps`: 2048 (episode collection buffer)
- `batch_size`: 64 (mini-batch size)
- `n_epochs`: 10 (policy update epochs)
- `gamma`: 0.99 (discount factor)
- `gae_lambda`: 0.95 (GAE parameter)
- `clip_range`: 0.2 (PPO clipping)

**Environment-Specific**:
- `total_timesteps`: 1,000,000 - 10,000,000 (depending on data size)
- `episode_length`: Variable (handled by environment)
- `action_space`: Discrete(5) (already configured)

### Training Schedule

**Phase 1: Initial Training** (1M timesteps)
- All markets, balanced sampling
- Establish baseline performance
- Identify problematic markets

**Phase 2: Curriculum Training** (2M timesteps)
- Progressive difficulty introduction
- Focus on learning fundamentals
- Build market-agnostic patterns

**Phase 3: Refinement** (1M timesteps)
- Balanced sampling across all markets
- Fine-tune performance
- Improve generalization

**Total**: ~4M timesteps (adjust based on data availability)

## Evaluation and Monitoring

### Performance Metrics

**Per-Market Metrics**:
- Episode reward (mean, std, min, max)
- Episode length
- Success rate (positive reward episodes)
- Final portfolio value

**Aggregate Metrics**:
- Average reward across all markets
- Reward distribution (histogram)
- Learning curve (reward vs training iteration)
- Generalization score (performance on held-out markets)

### Monitoring Strategy

**During Training**:
- TensorBoard logging (SB3 built-in)
- Track reward per episode
- Monitor policy entropy (exploration)
- Track value function estimates

**Per-Market Tracking**:
- Performance per market type
- Difficulty vs performance correlation
- Episode length vs performance correlation
- Market coverage vs performance correlation

### Validation Strategy

**Hold-Out Set**:
- Reserve 20% of MarketSessionViews for validation
- Never train on validation set
- Evaluate periodically during training

**Validation Metrics**:
- Average reward on validation set
- Performance consistency across markets
- Generalization to unseen markets

## Implementation Phases

### Phase 1: Basic SB3 Integration (Week 1)

**Goals**:
- Integrate SB3 PPO with MarketAgnosticKalshiEnv
- Create environment wrapper for view cycling
- Implement basic multi-episode training loop
- Validate training runs successfully

**Deliverables**:
- Environment wrapper class
- Basic training script
- Training runs without errors
- Basic monitoring/logging

### Phase 2: Multi-Episode Training (Week 2)

**Goals**:
- Implement multi-episode training per MarketSessionView
- Add episode length filtering
- Implement balanced sampling
- Track performance per market

**Deliverables**:
- Enhanced training loop
- Market filtering logic
- Performance tracking per market
- Training metrics dashboard

### Phase 3: Curriculum Learning (Week 3)

**Goals**:
- Implement difficulty scoring
- Progressive difficulty training
- Advanced monitoring
- Validation set evaluation

**Deliverables**:
- Difficulty scoring function
- Curriculum training implementation
- Validation framework
- Comprehensive monitoring

### Phase 4: Optimization & Tuning (Week 4)

**Goals**:
- Hyperparameter tuning
- Performance optimization
- Advanced curriculum strategies
- Production-ready training pipeline

**Deliverables**:
- Optimized hyperparameters
- Production training script
- Comprehensive documentation
- Training best practices guide

## Key Design Decisions

### Decision 1: Episodes Per MarketSessionView

**Options**:
- Fixed: 10-50 episodes per view
- Proportional: Episodes ∝ episode_length
- Adaptive: More episodes for difficult markets

**Decision**: Start with fixed (20 episodes), experiment with adaptive

**Rationale**: Fixed is simplest, adaptive may improve sample efficiency

### Decision 2: Environment Wrapper vs Custom Training Loop

**Options**:
- Environment wrapper (cycles views automatically)
- Custom training loop (explicit view management)

**Decision**: Environment wrapper

**Rationale**: Better SB3 integration, cleaner code, standard pattern

### Decision 3: Curriculum Learning vs Balanced Sampling

**Options**:
- Pure curriculum (easy → hard)
- Pure balanced (equal representation)
- Hybrid (curriculum then balanced)

**Decision**: Hybrid approach

**Rationale**: Best of both worlds - sample efficiency + generalization

### Decision 4: Minimum Episode Length

**Options**:
- No filtering (use all markets)
- Filter < 50 steps
- Filter < 100 steps

**Decision**: Filter < 50 steps

**Rationale**: Very short episodes provide little learning signal, 50 is reasonable minimum

## Risks and Mitigations

### Risk 1: Overfitting to Specific Markets

**Mitigation**:
- Balanced sampling ensures equal representation
- Market-agnostic features prevent market-specific patterns
- Validation set monitors generalization

### Risk 2: Variable Episode Lengths Cause Instability

**Mitigation**:
- SB3 handles variable lengths gracefully
- Filter very short episodes
- Monitor episode length distribution

### Risk 3: Insufficient Training Data

**Mitigation**:
- Multiple episodes per market (20-50x data multiplication)
- Use all available sessions
- Data augmentation if needed (future work)

### Risk 4: Curriculum Learning Too Aggressive

**Mitigation**:
- Start conservative (easy markets first)
- Monitor performance per difficulty level
- Adjust difficulty thresholds based on results

## Success Criteria

### Training Success Metrics

1. **Learning Progress**:
   - Reward increases over training iterations
   - Policy entropy decreases (less exploration needed)
   - Value function estimates improve

2. **Generalization**:
   - Performance on validation set improves
   - Consistent performance across market types
   - Good performance on held-out markets

3. **Stability**:
   - Training doesn't diverge
   - No catastrophic forgetting
   - Smooth learning curves

### Performance Targets

**Initial Targets** (after Phase 1):
- Training runs without errors
- Agent learns to take non-random actions
- Positive average reward on some markets

**Intermediate Targets** (after Phase 2):
- Positive average reward on 50%+ of markets
- Consistent performance across market types
- Learning curves show improvement

**Final Targets** (after Phase 4):
- Positive average reward on 80%+ of markets
- Strong generalization to validation set
- Production-ready training pipeline

## Future Enhancements

### Advanced Training Strategies

1. **Multi-Task Learning**: Train on multiple markets simultaneously
2. **Transfer Learning**: Pre-train on easy markets, fine-tune on hard
3. **Meta-Learning**: Learn to quickly adapt to new markets
4. **Ensemble Methods**: Train multiple agents, combine predictions

### Data Enhancements

1. **Data Augmentation**: Synthetic market data generation
2. **Active Learning**: Focus training on informative markets
3. **Online Learning**: Continuously update with new session data

### Algorithm Enhancements

1. **Custom Rewards**: More sophisticated reward shaping
2. **Hierarchical RL**: Multi-level decision making
3. **Imitation Learning**: Learn from expert demonstrations

## Conclusion

Training from MarketSessionView data requires a multi-episode approach with proper SB3 integration. The recommended strategy combines curriculum learning (for sample efficiency) with balanced sampling (for generalization). Key implementation priorities are:

1. Environment wrapper for view cycling
2. Multi-episode training per MarketSessionView
3. Curriculum learning with difficulty scoring
4. Comprehensive monitoring and validation

This plan provides a roadmap for effective RL training that leverages the market-agnostic architecture while ensuring robust learning and generalization.
