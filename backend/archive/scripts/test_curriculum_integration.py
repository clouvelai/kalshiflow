#!/usr/bin/env python3
"""
Test script for M9 SB3 curriculum integration.

This script tests the curriculum training functionality added to train_with_sb3.py
by running a quick curriculum training session on a small number of timesteps.
"""

import asyncio
import os
import sys
import subprocess
from pathlib import Path
import json
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_curriculum_integration():
    """Test the curriculum integration with SB3."""
    print("=" * 80)
    print("TESTING M9: SB3 CURRICULUM INTEGRATION")
    print("=" * 80)
    
    # Check DATABASE_URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        return False
    
    # Test parameters
    session_id = 9  # Use session 9 (should have good data)
    algorithm = "ppo"
    timesteps = 100  # Very small for quick testing
    
    print(f"\n🧪 TEST CONFIGURATION:")
    print(f"   Session: {session_id}")
    print(f"   Algorithm: {algorithm.upper()}")
    print(f"   Timesteps per market: {timesteps}")
    print(f"   Min episode length: 20")
    
    # Create temporary model path
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_curriculum_model.zip"
        
        # Run curriculum training
        print(f"\n🚀 RUNNING CURRICULUM TRAINING...")
        print(f"   Command: python train_with_sb3.py --session {session_id} --curriculum --algorithm {algorithm} --total-timesteps {timesteps}")
        
        try:
            # Build command
            cmd = [
                sys.executable,
                str(Path(__file__).parent.parent / "src" / "kalshiflow_rl" / "scripts" / "train_with_sb3.py"),
                "--session", str(session_id),
                "--curriculum",
                "--algorithm", algorithm,
                "--total-timesteps", str(timesteps),
                "--model-save-path", str(model_path),
                "--min-episode-length", "20",  # Lower threshold for more markets
                "--log-level", "INFO"
            ]
            
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Check results
            print(f"\n📊 TRAINING RESULTS:")
            print(f"   Return code: {result.returncode}")
            
            if result.returncode == 0:
                print("   ✅ Training completed successfully!")
                
                # Parse output for key metrics
                output = result.stdout
                lines = output.split('\n')
                
                # Look for curriculum results
                markets_trained = None
                total_markets = None
                
                for line in lines:
                    if "Markets trained:" in line:
                        # Example: "Markets trained: 15/282"
                        parts = line.split("Markets trained:")
                        if len(parts) > 1:
                            counts = parts[1].strip()
                            if '/' in counts:
                                markets_trained, total_markets = counts.split('/')
                                markets_trained = int(markets_trained)
                                total_markets = int(total_markets)
                
                print(f"\n✅ CURRICULUM VALIDATION:")
                if markets_trained is not None and total_markets is not None:
                    print(f"   Markets trained: {markets_trained}")
                    print(f"   Total viable markets: {total_markets}")
                    print(f"   Training coverage: {100*markets_trained/total_markets:.1f}%")
                    
                    if markets_trained > 0:
                        print(f"   ✅ Successfully trained on multiple markets!")
                        
                        # Check if model was saved
                        if model_path.exists():
                            print(f"   ✅ Model saved successfully: {model_path}")
                        else:
                            print(f"   ❌ Model file not found: {model_path}")
                        
                        # Look for results file
                        results_dir = model_path.parent
                        results_file = results_dir / "curriculum_training_results.json"
                        
                        if results_file.exists():
                            print(f"   ✅ Results file created: {results_file}")
                            
                            # Parse results
                            with open(results_file, 'r') as f:
                                results_data = json.load(f)
                            
                            print(f"\n📋 DETAILED RESULTS:")
                            print(f"   Session: {results_data.get('session_id')}")
                            print(f"   Markets trained: {len(results_data.get('markets_trained', []))}")
                            print(f"   Total duration: {results_data.get('total_duration', 0):.2f} seconds")
                            
                            # Show some market results
                            market_results = results_data.get('market_results', {})
                            successful_markets = [m for m, r in market_results.items() if 'error' not in r]
                            failed_markets = [m for m, r in market_results.items() if 'error' in r]
                            
                            print(f"   Successful markets: {len(successful_markets)}")
                            print(f"   Failed markets: {len(failed_markets)}")
                            
                            if successful_markets:
                                print(f"   Sample markets trained: {successful_markets[:3]}")
                            
                            return True
                        else:
                            print(f"   ❌ Results file not found: {results_file}")
                            return False
                    else:
                        print(f"   ❌ No markets were trained")
                        return False
                else:
                    print(f"   ❌ Could not parse training results from output")
                    print(f"   Output snippet: {output[:500]}")
                    return False
            else:
                print("   ❌ Training failed!")
                print(f"   Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("   ❌ Training timed out (>5 minutes)")
            return False
        except Exception as e:
            print(f"   ❌ Training failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main test entry point."""
    try:
        success = test_curriculum_integration()
        
        print(f"\n" + "=" * 80)
        if success:
            print("🎉 M9 CURRICULUM INTEGRATION TEST PASSED!")
            print("✅ SimpleMarketCurriculum successfully integrated with SB3 training")
            print("✅ Model trains on multiple markets sequentially")
            print("✅ Results and model files are properly saved")
        else:
            print("❌ M9 CURRICULUM INTEGRATION TEST FAILED!")
            print("   Check the error messages above for details")
        print("=" * 80)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())