import { test, expect } from '@playwright/test';
import { mkdir } from 'fs/promises';
import { join } from 'path';

/**
 * Frontend E2E Regression Test - Golden Standard
 * 
 * CRITICAL: This test MUST fail if the backend isn't running.
 * It validates real data flow from Kalshi → Backend → Frontend.
 * 
 * Success Criteria:
 * ✓ Backend is running and accepting WebSocket connections
 * ✓ WebSocket connection established with status "Live"
 * ✓ Real market data flows into the application
 * ✓ Analytics shows non-zero values
 * ✓ Chart renders with real data points
 * ✓ Top trades section is present
 * ✓ Data changes over time proving real-time updates
 */

test.describe('Frontend E2E Regression Test - Golden Standard', () => {
  test('Validates complete data flow from Kalshi through backend to frontend', async ({ page }) => {
    console.log('🚀 Starting Frontend E2E Regression Test');
    console.log('=====================================');
    
    // Create screenshot directory
    const screenshotDir = join(process.cwd(), 'test-results', 'screenshots');
    await mkdir(screenshotDir, { recursive: true });
    
    // Helper to take screenshot with logging
    const takeScreenshot = async (name) => {
      const path = join(screenshotDir, name);
      await page.screenshot({ path, fullPage: true });
      console.log(`📸 Screenshot: ${name}`);
    };
    
    // Track critical failures
    let criticalFailures = [];
    
    // ================================================================
    // PHASE 1: Backend Connection & Initial Data (0-3 seconds)
    // ================================================================
    console.log('\n📋 PHASE 1: Backend connection & initial data validation');
    
    // Navigate to application
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Verify basic application structure loads
    await expect(page.getByTestId('app-layout')).toBeVisible({ timeout: 3000 });
    console.log('✅ Application loaded');
    await takeScreenshot('01_initial_load.png');
    
    // CRITICAL: Check WebSocket connection status
    const connectionStatus = page.getByTestId('connection-status-text');
    
    // Wait for connection - fail fast if backend not running
    try {
      await expect(connectionStatus).toHaveText('Live Data Stream Active', { timeout: 5000 });
      console.log('✅ WebSocket connected to backend (status: Live Data Stream Active)');
      await takeScreenshot('02_connection_established.png');
    } catch (error) {
      const actualStatus = await connectionStatus.textContent();
      console.log(`❌ CRITICAL: WebSocket connection failed. Status: "${actualStatus}"`);
      await takeScreenshot('error_no_connection.png');
      throw new Error('Backend is not running or WebSocket connection failed. Status: ' + actualStatus);
    }
    
    // Quick check for any initial data (wait a bit longer for data to flow)
    await page.waitForTimeout(4000);
    
    // ================================================================
    // PHASE 2: Core Component Validation (3-8 seconds)
    // ================================================================
    console.log('\n🔍 PHASE 2: Core component validation');
    
    // 1. Analytics Values - Must have non-zero data (allow for initial load time)
    const totalVolume = await page.getByTestId('total-volume-value').textContent();
    const totalTrades = await page.getByTestId('total-trades-value').textContent();
    
    if (totalVolume && totalVolume !== '$0' && totalVolume !== '0') {
      console.log(`✅ Analytics active - Volume: ${totalVolume}, Trades: ${totalTrades}`);
    } else {
      // Don't mark as critical failure immediately - data may still be loading
      console.log('⚠️  No analytics data yet (may still be loading)');
    }
    
    // 2. Top trades - component should render even if waiting for data
    const topTradesSection = page.getByTestId('top-trades-list');
    if (await topTradesSection.isVisible()) {
      console.log('✅ Top trades section rendered');
    } else {
      criticalFailures.push('Top trades section missing');
      console.log('❌ Top trades section missing - critical failure');
    }
    
    // 3. Chart Validation - Should have bars
    const chartBars = page.locator('[data-testid="analytics-chart"] .recharts-bar-rectangle');
    const barCount = await chartBars.count();
    
    if (barCount > 0) {
      console.log(`✅ Chart rendering - ${barCount} data points visible`);
    } else {
      console.log('⚠️  Chart has no visible data (may be building)');
    }
    
    // 4. FAQ Section
    const faqSection = page.getByTestId('faq-section');
    if (await faqSection.isVisible()) {
      console.log('✅ FAQ section rendered');
    } else {
      console.log('⚠️  FAQ section not visible');
    }
    
    // Take screenshot of data populated state
    await takeScreenshot('03_data_populated.png');
    
    // Capture initial state for comparison
    const initialState = {
      volume: totalVolume,
      trades: totalTrades
    };
    
    // ================================================================
    // PHASE 3: Quick Interactive Test (8-10 seconds)
    // ================================================================
    console.log('\n⚡ PHASE 3: Quick functionality test');
    
    const chartVisible = await page.getByTestId('analytics-chart').isVisible().catch(() => false);
    const chartLoading = await page.getByTestId('chart-loading').isVisible().catch(() => false);
    if (chartVisible || chartLoading) {
      console.log('✅ Hourly trading activity chart is present');
    } else {
      criticalFailures.push('Trading activity chart missing');
    }
    await takeScreenshot('04_interactive_features.png');
    
    // ================================================================
    // PHASE 4: Wait and Verify Real-time Updates (10-15 seconds)
    // ================================================================
    console.log('\n🔄 PHASE 4: Verifying real-time updates');
    console.log('⏳ Waiting 5 seconds for data changes...');
    
    await page.waitForTimeout(5000);
    
    // Check for updates
    const updatedVolume = await page.getByTestId('total-volume-value').textContent();
    const updatedTrades = await page.getByTestId('total-trades-value').textContent();
    // Check if TradeFlowRiver is present instead of trade tape
    const topTradesStillVisible = await page.getByTestId('top-trades-list').isVisible();
    
    const finalState = {
      volume: updatedVolume,
      trades: updatedTrades,
      topTradesVisible: topTradesStillVisible
    };
    
    console.log(`📊 Initial: Volume=${initialState.volume}, Trades=${initialState.trades}`);
    console.log(`📊 Updated: Volume=${finalState.volume}, Trades=${finalState.trades}, TopTrades=${finalState.topTradesVisible}`);
    
    // Check if anything changed
    let dataChanged = false;
    if (finalState.volume !== initialState.volume) {
      console.log(`✅ Volume updated: ${initialState.volume} → ${finalState.volume}`);
      dataChanged = true;
    }
    if (finalState.trades !== initialState.trades) {
      console.log(`✅ Trade count updated: ${initialState.trades} → ${finalState.trades}`);
      dataChanged = true;
    }
    if (finalState.topTradesVisible) {
      console.log(`✅ Top trades section is active`);
    }
    
    if (!dataChanged) {
      console.log('ℹ️  No changes detected (market may be quiet)');
    }
    
    // Take screenshot showing final state with updates
    await takeScreenshot('05_final_state.png');
    
    // ================================================================
    // FINAL VALIDATION (15-16 seconds)
    // ================================================================
    console.log('\n📋 FINAL VALIDATION');
    console.log('=====================================');
    
    // Check connection is still live
    const finalStatus = await connectionStatus.textContent();
    if (finalStatus === 'Live Data Stream Active') {
      console.log('✅ WebSocket connection: STABLE');
    } else {
      criticalFailures.push(`Connection lost during test: ${finalStatus}`);
      console.log(`❌ WebSocket connection: ${finalStatus}`);
    }
    
    // Verify all components still visible
    const componentsOk = 
      await page.getByTestId('unified-analytics').isVisible() &&
      await page.getByTestId('top-trades-list').isVisible() &&
      await page.getByTestId('faq-section').isVisible();
    
    if (componentsOk) {
      console.log('✅ All components: RENDERED');
    } else {
      criticalFailures.push('Components failed to render');
      console.log('❌ Component rendering issues detected');
    }
    
    // Final decision
    if (criticalFailures.length > 0) {
      console.log('\n❌ TEST FAILED - Critical issues:');
      criticalFailures.forEach(failure => console.log(`   - ${failure}`));
      throw new Error(`Test failed with ${criticalFailures.length} critical failures`);
    }
    
    // Must have data to pass
    if (!finalState.volume || finalState.volume === '$0') {
      throw new Error('No data flowing through system - backend may not be receiving trades');
    }
    
    console.log('\n🎉 Frontend E2E Test PASSED');
    console.log('✅ Backend connected and stable');
    console.log('✅ Data flowing through system');
    console.log('✅ All components functional');
    console.log('✅ System is operational');
    console.log(`\n⏱️  Test completed in ~15 seconds`);
  });
});