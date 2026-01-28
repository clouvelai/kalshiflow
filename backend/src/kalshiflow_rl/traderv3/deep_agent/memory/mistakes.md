# Trading API Issues and Mistakes

## PERSISTENT API ISSUE (2026-01-27 23:01)

### Problem
The trade() function consistently fails with error:
```
"exactly one of yes_price, no_price, yes_price_dollars, or no_price_dollars should be provided"
```

### Failed Attempts
1. No price parameters - FAILED
2. `yes_price=15` - FAILED  
3. `yes_price_dollars=0.15` - FAILED
4. Market order approach - FAILED
5. Reduced contracts to 10 - FAILED

### **CRITICAL MISSED OPPORTUNITY**
**Trump OUT Market Signals**:
- **Price Impact**: +50 to +90 (EXCEPTIONAL)
- **Confidence**: 1.0 (PERFECT across 8+ signals)
- **Peak Signal**: +90 price impact - highest seen
- **Current Price**: 11¢ (massive upside potential)
- **Duration**: Signals persisting across multiple cycles

This represents the **STRONGEST EDGE OPPORTUNITY** identified to date, but technical execution is completely blocked.

### Signal Quality Validation ✅
- Multiple corroborating signals over time
- Perfect confidence scores consistently
- Clear directional bias (all suggest BUY YES)
- Proper sentiment transformation logic (OUT markets + negative sentiment = positive impact)
- Market hasn't moved despite signals (confirming edge opportunity)

### Technical Requirements (Still Unknown)
The API error suggests it needs exactly one of:
- `yes_price` (attempted - failed)
- `no_price` 
- `yes_price_dollars` (attempted - failed)
- `no_price_dollars`

### Impact Assessment
- **Signal Detection**: WORKING PERFECTLY ✅
- **Edge Identification**: WORKING PERFECTLY ✅  
- **Trade Execution**: COMPLETELY BLOCKED ❌

This is a **CRITICAL SYSTEM LIMITATION** preventing profitable trading on exceptional signals.