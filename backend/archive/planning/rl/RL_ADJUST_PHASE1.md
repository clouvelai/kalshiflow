# RL Phase 1: Foundation Learning Implementation Plan

## Executive Summary

**Current Status**: Our RL system is 95% ready for Phase 1 foundation learning. All 52 current features are basic market features - NO algorithm-specific features have been implemented yet.

**Gap**: Minor configuration changes needed for session filtering, curriculum staging, and training parameters.

**Timeline**: 8-11 hours (1-2 days) to complete Phase 1 setup.

---

## Current System Analysis

### ✅ What We Have (Complete & Working)
- **Full orderbook collection system** - collecting data daily
- **MarketAgnosticKalshiEnv** with 52 basic features
- **SB3 training pipeline** (PPO/A2C) with curriculum support
- **Session-based data loading** from historical episodes
- **Simulated order management** with probabilistic fills
- **Unified reward system** tracking portfolio value
- **21-action space** (HOLD + BUY/SELL at multiple sizes)

### 📊 Current Feature Breakdown (52 Features - ALL BASIC)

**Market Features (21 per market):**
- 8 Price features: bid/ask/mid/spread for YES/NO
- 6 Volume features: total volume, imbalances, side ratios  
- 4 Book shape: depth, liquidity concentration
- 3 Arbitrage/efficiency: opportunity detection, market efficiency

**Temporal Features (14):**
- 3 Time-based: time gap, time of day, day of week
- 4 Activity momentum: current activity, changes, update frequency
- 4 Activity patterns: burst/quiet indicators, trends
- 3 Single-market: volatility regime, consistency, stability

**Portfolio Features (12):**
- 2 Composition: cash ratio, position ratio
- 6 Position characteristics: count, size, long/short ratios
- 4 Risk metrics: diversity, leverage, unrealized P&L

**Order Features (5):**
- Order state: open buy/sell, distances, time since order

### 🎯 Key Finding
**NO algorithm-specific features exist yet!** Current system is perfect for Phase 1 foundation learning.

---

## Phase 1 Implementation Requirements

### Phase 1 Goals (From Roadmap)
1. **Learn basic market patterns** without algorithm focus
2. **Stable market selection** (lower volatility sessions)  
3. **Simple feature subset** (we already have this!)
4. **Pure P&L rewards** (no pattern exploitation bonuses)

### Phase 1 Curriculum Stages
```python
# Week 1: Price Prediction 
Stage1(name="price_prediction", duration="1 week")

# Week 1: Basic Trading
Stage2(name="basic_trading", duration="1 week")

# Week 2: Position Management  
Stage3(name="position_management", duration="2 weeks")
```

---

## Implementation Tasks

### 1. Session Filtering for Stable Markets
**Time Estimate**: 2-3 hours  
**Files to Modify**: 
- `src/kalshiflow_rl/environments/curriculum_service.py`
- `src/kalshiflow_rl/environments/session_data_loader.py`

**Implementation**:

#### A. Create StabilityMetrics Class
```python
class SessionStabilityMetrics:
    """Calculate session stability for foundation learning."""
    
    @staticmethod
    def calculate_stability_score(session_data: SessionData) -> float:
        """
        Calculate stability score [0,1] where 1 = most stable.
        
        Factors:
        - Spread volatility (lower = more stable)
        - Activity consistency (consistent = more stable)  
        - Volume patterns (moderate = more stable)
        """
        pass
    
    @staticmethod
    def is_stable_session(session_data: SessionData, threshold: float = 0.6) -> bool:
        """Return True if session meets stability criteria."""
        pass
```

#### B. Add Session Filtering
```python
class SessionDataLoader:
    async def get_stable_sessions(self, stability_threshold: float = 0.6) -> List[SessionData]:
        """Filter sessions by stability score for foundation learning."""
        pass
```

#### C. Update Curriculum Service
```python
class CurriculumService:
    def select_foundation_sessions(self) -> List[int]:
        """Select stable sessions for Phase 1 foundation learning."""
        pass
```

### 2. Curriculum Staging System
**Time Estimate**: 4-5 hours  
**Files to Create/Modify**:
- `src/kalshiflow_rl/training/foundation_curriculum.py` (new)
- `src/kalshiflow_rl/training/train_sb3.py` (modify)

**Implementation**:

#### A. Create Foundation Curriculum
```python
class FoundationCurriculum:
    """3-stage foundation learning curriculum."""
    
    def __init__(self):
        self.stages = [
            PricePredictionStage(duration_days=7, max_position=10),
            BasicTradingStage(duration_days=7, max_position=50), 
            PositionManagementStage(duration_days=14, max_position=100)
        ]
    
    def get_stage_config(self, stage_id: int) -> StageConfig:
        """Get configuration for specific curriculum stage."""
        pass
    
    def get_reward_function(self, stage_id: int) -> RewardFunction:
        """Get stage-specific reward function."""
        pass
```

#### B. Stage-Specific Configurations
```python
@dataclass
class PricePredictionStage:
    """Stage 1: Focus on price prediction accuracy."""
    duration_days: int = 7
    max_position: int = 10
    allowed_actions: List[int] = field(default_factory=lambda: [0, 1, 2, 6, 7])  # HOLD + small sizes
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'price_prediction_accuracy': 0.7,
        'simple_pnl': 0.3
    })

@dataclass  
class BasicTradingStage:
    """Stage 2: Basic trading with transaction costs."""
    duration_days: int = 7
    max_position: int = 50
    allowed_actions: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 6, 7, 8])  # Medium sizes
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'simple_pnl': 0.8,
        'transaction_cost_penalty': 0.2
    })

@dataclass
class PositionManagementStage:
    """Stage 3: Full position management with risk controls."""
    duration_days: int = 14
    max_position: int = 100
    allowed_actions: List[int] = field(default_factory=lambda: list(range(21)))  # All actions
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'risk_adjusted_pnl': 0.6,
        'drawdown_penalty': 0.2, 
        'diversification_bonus': 0.2
    })
```

#### C. Stage-Specific Reward Functions
```python
class FoundationRewardCalculator:
    """Reward calculator for foundation learning stages."""
    
    def calculate_stage1_reward(self, state: TradingState) -> float:
        """Price prediction focused rewards."""
        pass
    
    def calculate_stage2_reward(self, state: TradingState) -> float:
        """Basic trading with cost awareness."""
        pass
    
    def calculate_stage3_reward(self, state: TradingState) -> float:
        """Full risk-adjusted portfolio management."""
        pass
```

### 3. Training Configuration Updates  
**Time Estimate**: 1-2 hours
**Files to Modify**:
- `src/kalshiflow_rl/training/train_sb3.py`
- `src/kalshiflow_rl/training/sb3_wrapper.py`

**Implementation**:

#### A. Add Phase 1 Training Mode
```python
def main():
    parser.add_argument('--phase1', action='store_true',
                       help='Enable Phase 1 foundation learning mode')
    parser.add_argument('--stage', type=int, choices=[1,2,3], 
                       help='Specific curriculum stage (1-3)')
    
    if args.phase1:
        print("🎯 PHASE 1 FOUNDATION LEARNING MODE")
        curriculum = FoundationCurriculum()
        # Configure foundation learning
```

#### B. Update SB3TrainingConfig
```python
@dataclass
class SB3TrainingConfig:
    # Existing fields...
    
    # Phase 1 specific configurations
    foundation_mode: bool = False
    curriculum_stage: Optional[int] = None
    stable_sessions_only: bool = False
    stability_threshold: float = 0.6
```

#### C. Integration with Training Loop
```python
def create_foundation_training_env(config: SB3TrainingConfig) -> SessionBasedEnvironment:
    """Create environment configured for foundation learning."""
    
    if config.stable_sessions_only:
        # Filter to stable sessions only
        stable_sessions = curriculum_service.select_foundation_sessions()
        config.session_ids = stable_sessions
    
    if config.curriculum_stage:
        # Apply stage-specific configuration
        stage_config = foundation_curriculum.get_stage_config(config.curriculum_stage)
        # Update environment with stage settings
    
    return create_sb3_env(config)
```

### 4. Documentation and Validation
**Time Estimate**: 1 hour
**Files to Create/Modify**:
- `tests/test_rl/training/test_foundation_curriculum.py` (new)
- Update training documentation

**Implementation**:

#### A. Validation Tests
```python
def test_foundation_curriculum_stages():
    """Test that all 3 curriculum stages are properly configured."""
    pass

def test_stable_session_filtering():
    """Test that session stability filtering works correctly."""
    pass

def test_phase1_training_integration():
    """Test that Phase 1 mode works end-to-end.""" 
    pass
```

#### B. Usage Documentation
```bash
# Phase 1 Foundation Learning Examples

# Stage 1: Price Prediction (Week 1)
python train_sb3.py --phase1 --stage 1 --algorithm ppo --total-timesteps 50000

# Stage 2: Basic Trading (Week 1) 
python train_sb3.py --phase1 --stage 2 --algorithm ppo --total-timesteps 50000

# Stage 3: Position Management (Week 2)
python train_sb3.py --phase1 --stage 3 --algorithm ppo --total-timesteps 100000

# Full foundation curriculum (auto-progression)
python train_sb3.py --phase1 --algorithm ppo --curriculum
```

---

## Expected Outcomes

### Phase 1 Success Metrics
- **Stage 1**: Price prediction accuracy >60%, minimal trading activity
- **Stage 2**: Positive P&L with transaction cost awareness  
- **Stage 3**: Risk-adjusted returns with proper position management

### Phase 1 → Phase 2 Transition
Once foundation learning shows consistent performance:
1. **Stable profitable trading** in controlled environments
2. **Proper risk management** behaviors learned  
3. **Ready for algorithm pattern features** (Phase 2)

### Timeline
- **Week 1**: Stages 1-2 (price prediction + basic trading)
- **Week 2**: Stage 3 (position management)
- **Week 3**: Evaluation and Phase 2 preparation
- **Week 4**: Begin Phase 2 (algorithm pattern detection)

---

## Implementation Priority

1. ✅ **Current system assessment** (COMPLETE - all basic features)
2. 🔄 **Session filtering** (2-3 hours) - START HERE
3. 🔄 **Curriculum staging** (4-5 hours) - CORE IMPLEMENTATION  
4. 🔄 **Training configuration** (1-2 hours) - INTEGRATION
5. 🔄 **Validation testing** (1 hour) - QUALITY ASSURANCE

**Total Effort**: 8-11 hours over 1-2 days

---

## Key Insights

### Why Phase 1 is Nearly Ready
- **No algorithm features to remove** - current features are all basic
- **Training infrastructure exists** - just needs curriculum configuration
- **Data collection working** - stable sessions available for selection

### Competitive Advantage
Starting with foundation learning ensures:
- **Solid trading fundamentals** before algorithm exploitation
- **Robust risk management** learned early
- **Transferable skills** across market conditions
- **Baseline performance** to measure algorithm pattern improvements against

Foundation learning isn't just preparation - it's the critical base layer that makes algorithm exploitation profitable and sustainable.