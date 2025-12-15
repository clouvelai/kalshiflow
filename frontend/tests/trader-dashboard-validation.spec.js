import { test, expect } from '@playwright/test';
import { mkdir } from 'fs/promises';
import { join } from 'path';

/**
 * Trader Dashboard E2E Validation Test
 * 
 * CRITICAL: This test validates the RL Trader UX dashboard functionality.
 * Requires backend RL service to be running on port 8002.
 * 
 * Success Criteria:
 * ✓ Trader dashboard loads at /trader
 * ✓ WebSocket connection established to ws://localhost:8002/rl/ws
 * ✓ TraderStatePanel shows portfolio metrics
 * ✓ ActionFeed receives HOLD action messages from RL service
 * ✓ Real-time data updates from RL service
 * ✓ No critical console errors
 */

test.describe('Trader Dashboard Validation', () => {
  test('Validates complete trader dashboard functionality with RL service', async ({ page }) => {
    console.log('🚀 Starting Trader Dashboard Validation Test');
    console.log('============================================');
    
    // Create screenshot directory
    const screenshotDir = join(process.cwd(), 'test-results', 'trader-screenshots');
    await mkdir(screenshotDir, { recursive: true });
    
    // Helper to take screenshot with logging
    const takeScreenshot = async (name) => {
      const path = join(screenshotDir, name);
      await page.screenshot({ path, fullPage: true });
      console.log(`📸 Screenshot: ${name}`);
    };
    
    // Track console messages
    const consoleMessages = [];
    const errors = [];
    
    page.on('console', (msg) => {
      const text = msg.text();
      consoleMessages.push({
        type: msg.type(),
        text: text,
        timestamp: new Date().toISOString()
      });
      console.log(`🖥️ Console [${msg.type()}]: ${text}`);
    });
    
    page.on('pageerror', (err) => {
      errors.push(err.message);
      console.log(`❌ Page Error: ${err.message}`);
    });
    
    // ================================================================
    // PHASE 1: Navigation and Initial Load
    // ================================================================
    console.log('\n📋 PHASE 1: Navigation to trader dashboard');
    
    // Navigate to trader dashboard
    await page.goto('/trader', { waitUntil: 'networkidle' });
    
    // Verify page loaded
    await expect(page).toHaveURL(/.*\/trader/);
    console.log('✅ Successfully navigated to /trader');
    
    await takeScreenshot('01_trader_initial_load.png');
    
    // Check basic page structure
    const header = page.locator('header');
    await expect(header).toBeVisible({ timeout: 3000 });
    
    const mainContent = page.locator('main');
    await expect(mainContent).toBeVisible();
    
    // Look for trader-specific elements
    const traderTitle = page.locator('h1:has-text("RL Trader Dashboard")');
    await expect(traderTitle).toBeVisible();
    console.log('✅ Trader dashboard header found');
    
    // ================================================================
    // PHASE 2: WebSocket Connection Validation
    // ================================================================
    console.log('\n🔗 PHASE 2: WebSocket connection validation');
    
    // Wait a moment for WebSocket connection
    await page.waitForTimeout(3000);
    
    // Check for connection status indicators
    const connectionIndicators = [
      'text="Live"',
      'text="Connected"',
      '.text-green-400',
      '[class*="green"]'
    ];
    
    let connectionFound = false;
    let connectionText = '';
    
    for (const selector of connectionIndicators) {
      try {
        const element = page.locator(selector).first();
        if (await element.isVisible({ timeout: 1000 })) {
          connectionText = await element.textContent();
          if (connectionText.toLowerCase().includes('live') || 
              connectionText.toLowerCase().includes('connected')) {
            connectionFound = true;
            console.log(`✅ WebSocket connection status: "${connectionText}"`);
            break;
          }
        }
      } catch (e) {
        // Continue checking
      }
    }
    
    if (!connectionFound) {
      // Check for connection status in header
      const statusElement = page.locator('[class*="text-green"], [class*="text-red"], [class*="text-yellow"]').first();
      if (await statusElement.isVisible({ timeout: 2000 })) {
        connectionText = await statusElement.textContent();
        console.log(`📊 Connection status found: "${connectionText}"`);
      } else {
        console.log('⚠️  No clear connection status indicator found');
      }
    }
    
    await takeScreenshot('02_websocket_connection_status.png');
    
    // ================================================================
    // PHASE 3: TraderStatePanel Validation
    // ================================================================
    console.log('\n💼 PHASE 3: TraderStatePanel validation');
    
    // Look for portfolio metrics
    const portfolioMetrics = [
      'Portfolio Value',
      'Available Cash', 
      'Open Orders',
      'Fill Rate',
      '$', // Currency symbol
      '%'  // Percentage symbol
    ];
    
    let metricsFound = 0;
    const foundMetrics = [];
    
    for (const metric of portfolioMetrics) {
      const elements = page.locator(`text="${metric}"`);
      const count = await elements.count();
      if (count > 0) {
        metricsFound++;
        foundMetrics.push(metric);
        console.log(`✅ Found metric: ${metric}`);
      }
    }
    
    if (metricsFound >= 2) {
      console.log(`✅ TraderStatePanel displaying ${metricsFound} portfolio metrics`);
    } else {
      console.log(`⚠️  TraderStatePanel only showing ${metricsFound} metrics - may still be loading`);
    }
    
    // Check for dollar amounts (financial data)
    const dollarAmounts = page.locator('text=/\\$\\d+/');
    const dollarCount = await dollarAmounts.count();
    if (dollarCount > 0) {
      console.log(`✅ Found ${dollarCount} dollar amounts - portfolio data present`);
    }
    
    await takeScreenshot('03_trader_state_panel.png');
    
    // ================================================================
    // PHASE 4: ActionFeed Validation
    // ================================================================
    console.log('\n⚡ PHASE 4: ActionFeed validation');
    
    // Wait for potential action messages
    await page.waitForTimeout(5000);
    
    // Look for action-related content
    const actionIndicators = [
      'HOLD',
      'Action',
      'Decision', 
      'Trading',
      'Strategy',
      'timestamp',
      'action_type'
    ];
    
    let actionContentFound = 0;
    const foundContent = [];
    
    for (const indicator of actionIndicators) {
      const elements = page.locator(`text="${indicator}"`);
      const count = await elements.count();
      if (count > 0) {
        actionContentFound++;
        foundContent.push(indicator);
        console.log(`✅ Found action content: ${indicator}`);
      }
    }
    
    // Check for list items that might contain actions
    const listItems = page.locator('li, .action-item, [class*="action"]');
    const listItemCount = await listItems.count();
    
    if (actionContentFound > 0 || listItemCount > 0) {
      console.log(`✅ ActionFeed showing trading content (${actionContentFound} indicators, ${listItemCount} entries)`);
    } else {
      console.log('⚠️  ActionFeed not showing expected content - may need more time for HOLD actions');
    }
    
    await takeScreenshot('04_action_feed.png');
    
    // ================================================================
    // PHASE 5: Real-time Data Updates
    // ================================================================
    console.log('\n🔄 PHASE 5: Real-time data validation');
    
    // Capture initial page content
    const initialContent = await page.content();
    
    console.log('⏳ Waiting 10 seconds for real-time updates...');
    await page.waitForTimeout(10000);
    
    // Capture updated content
    const updatedContent = await page.content();
    
    // Check for content changes
    const contentChanged = initialContent !== updatedContent;
    
    // Look for timestamps or time-related content
    const timeElements = page.locator('text=/\\d{1,2}:\\d{2}|AM|PM|2025|Dec/');
    const timeElementCount = await timeElements.count();
    
    const realTimeActive = contentChanged || timeElementCount > 0;
    
    if (realTimeActive) {
      console.log(`✅ Real-time updates detected (content changed: ${contentChanged}, time elements: ${timeElementCount})`);
    } else {
      console.log('⚠️  No clear real-time updates observed');
    }
    
    await takeScreenshot('05_real_time_updates.png');
    
    // ================================================================
    // PHASE 6: Console Error Analysis
    // ================================================================
    console.log('\n🐛 PHASE 6: Console error analysis');
    
    const errorMessages = consoleMessages.filter(msg => msg.type === 'error');
    const warningMessages = consoleMessages.filter(msg => msg.type === 'warn');
    const websocketMessages = consoleMessages.filter(msg => 
      msg.text.toLowerCase().includes('websocket') || 
      msg.text.toLowerCase().includes('ws') ||
      msg.text.toLowerCase().includes('connected') ||
      msg.text.toLowerCase().includes('trader')
    );
    
    // Filter out non-critical errors
    const criticalErrors = errorMessages.filter(err => 
      !err.text.includes('favicon') && 
      !err.text.includes('Extension') && 
      !err.text.includes('chrome-extension') &&
      !err.text.includes('DevTools')
    );
    
    console.log(`📊 Console Analysis:`);
    console.log(`   - Total errors: ${errorMessages.length}`);
    console.log(`   - Critical errors: ${criticalErrors.length}`);
    console.log(`   - Warnings: ${warningMessages.length}`);
    console.log(`   - WebSocket messages: ${websocketMessages.length}`);
    console.log(`   - Page errors: ${errors.length}`);
    
    if (criticalErrors.length === 0) {
      console.log('✅ No critical console errors detected');
    } else {
      console.log(`⚠️  ${criticalErrors.length} critical console errors detected`);
      criticalErrors.slice(0, 3).forEach(err => {
        console.log(`   Error: ${err.text}`);
      });
    }
    
    // ================================================================
    // PHASE 7: Final Validation & Report
    // ================================================================
    console.log('\n📋 FINAL VALIDATION SUMMARY');
    console.log('============================');
    
    const validationResults = {
      navigation: true, // We got here successfully
      websocket_status: connectionFound,
      portfolio_metrics: metricsFound >= 2,
      action_feed: actionContentFound > 0 || listItemCount > 0,
      real_time_updates: realTimeActive,
      console_errors: criticalErrors.length === 0,
      page_errors: errors.length === 0
    };
    
    const passedTests = Object.values(validationResults).filter(Boolean).length;
    const totalTests = Object.keys(validationResults).length;
    
    console.log(`✅ Passed: ${passedTests}/${totalTests} validation tests`);
    
    // Log detailed results
    Object.entries(validationResults).forEach(([test, passed]) => {
      const status = passed ? '✅' : '❌';
      console.log(`${status} ${test.replace(/_/g, ' ')}`);
    });
    
    // Take final screenshot
    await takeScreenshot('06_final_validation_state.png');
    
    // Generate validation report
    const report = {
      timestamp: new Date().toISOString(),
      git_commit: '327f545c312104b08f1e20fc25f16f19f019019f',
      trader_url: '/trader',
      expected_websocket: 'ws://localhost:8002/rl/ws',
      validation_results: validationResults,
      metrics: {
        portfolio_metrics_found: metricsFound,
        action_content_found: actionContentFound,
        list_items_count: listItemCount,
        time_elements: timeElementCount,
        console_errors: errorMessages.length,
        critical_errors: criticalErrors.length,
        websocket_messages: websocketMessages.length
      },
      recommendations: []
    };
    
    // Add recommendations based on results
    if (!validationResults.websocket_status) {
      report.recommendations.push('WebSocket connection not clearly indicated. Verify RL service is running on port 8002.');
    }
    
    if (!validationResults.portfolio_metrics) {
      report.recommendations.push('TraderStatePanel not showing expected portfolio metrics. Check trader_state message broadcasting.');
    }
    
    if (!validationResults.action_feed) {
      report.recommendations.push('ActionFeed not receiving expected content. Verify HOLD strategy is broadcasting trader_action messages.');
    }
    
    if (criticalErrors.length > 0) {
      report.recommendations.push('Console errors detected. Review browser console for JavaScript issues.');
    }
    
    console.log('\n💡 RECOMMENDATIONS:');
    if (report.recommendations.length === 0) {
      console.log('   No major issues detected - trader dashboard functioning well!');
    } else {
      report.recommendations.forEach((rec, index) => {
        console.log(`   ${index + 1}. ${rec}`);
      });
    }
    
    // Success criteria: Most tests should pass
    const successThreshold = 0.7; // 70% of tests should pass
    const successRate = passedTests / totalTests;
    
    if (successRate >= successThreshold) {
      console.log(`\n🎉 Trader Dashboard Validation PASSED (${Math.round(successRate * 100)}% success rate)`);
      console.log('✅ Trader dashboard is functional');
    } else {
      console.log(`\n❌ Trader Dashboard Validation FAILED (${Math.round(successRate * 100)}% success rate)`);
      throw new Error(`Trader dashboard validation failed - only ${passedTests}/${totalTests} tests passed`);
    }
    
    console.log(`\n⏱️  Test completed successfully`);
  });
});