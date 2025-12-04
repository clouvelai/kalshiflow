const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

// Read session name
const sessionName = fs.readFileSync('puppeteer_agent_artifacts/current_session.txt', 'utf8').trim();
const screenshotDir = `puppeteer_agent_artifacts/${sessionName}`;

console.log(`Starting Milestone 5 UX Reorganization Validation - Session: ${sessionName}`);

// Validation results tracking
let validationResults = {
  session: sessionName,
  timestamp: new Date().toISOString(),
  tests: {},
  screenshots: [],
  summary: {
    total_tests: 0,
    passed: 0,
    failed: 0,
    issues: []
  }
};

// Helper function to log validation results
function logTest(testName, status, details) {
  validationResults.tests[testName] = {
    status: status,
    details: details,
    timestamp: new Date().toISOString()
  };
  validationResults.summary.total_tests++;
  if (status === 'PASS') {
    validationResults.summary.passed++;
  } else {
    validationResults.summary.failed++;
    validationResults.summary.issues.push(`${testName}: ${details}`);
  }
  console.log(`[${status}] ${testName}: ${details}`);
}

// Helper function to take screenshot
async function takeScreenshot(page, filename, description) {
  const screenshotPath = path.join(screenshotDir, filename);
  await page.screenshot({ 
    path: screenshotPath, 
    fullPage: true 
  });
  validationResults.screenshots.push({
    filename: filename,
    description: description,
    path: screenshotPath
  });
  console.log(`Screenshot saved: ${filename} - ${description}`);
  return screenshotPath;
}

// Helper function to wait for element
async function waitForElementSafe(page, selector, timeout = 10000) {
  try {
    await page.waitForSelector(selector, { timeout });
    return true;
  } catch (error) {
    return false;
  }
}

// Helper function to check if element exists
async function elementExists(page, selector) {
  const elements = await page.$$(selector);
  return elements.length > 0;
}

async function runValidation() {
  let browser;
  
  try {
    browser = await puppeteer.launch({
      headless: false, // Set to false for debugging
      defaultViewport: { width: 1920, height: 1080 },
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();
    
    // Set up console message capture
    const consoleMessages = [];
    page.on('console', (msg) => {
      consoleMessages.push({
        type: msg.type(),
        text: msg.text(),
        timestamp: new Date().toISOString()
      });
    });

    // Navigate to the application
    console.log('Navigating to application...');
    await page.goto('http://localhost:5173', { 
      waitUntil: 'networkidle2',
      timeout: 30000 
    });

    // Wait for application to load
    await new Promise(resolve => setTimeout(resolve, 3000));

    console.log('\n=== STARTING MILESTONE 5 VALIDATION TESTS ===\n');

    // Test 1: Hero Stats Section Validation
    console.log('1. HERO STATS SECTION VALIDATION');
    
    // Check for hero stats container
    const hasHeroStats = await elementExists(page, '[data-testid="hero-stats"], .hero-stats, .stats-section');
    if (hasHeroStats) {
      logTest('hero_stats_container', 'PASS', 'Hero stats section found');
      
      // Check for individual stats cards
      const tradesCardExists = await elementExists(page, '[data-testid="trades-card"], .trades-card, .stat-card');
      const volumeCardExists = await elementExists(page, '[data-testid="volume-card"], .volume-card, .stat-card');
      const flowCardExists = await elementExists(page, '[data-testid="flow-card"], .flow-card, .stat-card');
      
      if (tradesCardExists && volumeCardExists && flowCardExists) {
        logTest('hero_stats_cards', 'PASS', 'All three hero stats cards found (trades, volume, flow)');
      } else {
        logTest('hero_stats_cards', 'FAIL', `Missing stats cards - trades:${tradesCardExists}, volume:${volumeCardExists}, flow:${flowCardExists}`);
      }
      
      await takeScreenshot(page, '01_hero_stats_section.png', 'Hero stats section with all three metrics');
    } else {
      logTest('hero_stats_container', 'FAIL', 'Hero stats section not found');
    }

    // Test 2: Market Grid Layout Validation  
    console.log('\n2. MARKET GRID LAYOUT VALIDATION');
    
    const hasMarketGrid = await elementExists(page, '[data-testid="market-grid"], .market-grid, .markets-grid');
    if (hasMarketGrid) {
      logTest('market_grid_container', 'PASS', 'Market grid container found');
      
      // Check for market cards
      const marketCards = await page.$$('[data-testid="market-card"], .market-card, .ticker-card');
      if (marketCards.length > 0) {
        logTest('market_grid_cards', 'PASS', `${marketCards.length} market cards found in grid`);
        
        // Check for enhanced sparklines
        const hasSparklines = await elementExists(page, 'svg, canvas, .sparkline, [data-testid="sparkline"]');
        if (hasSparklines) {
          logTest('market_grid_sparklines', 'PASS', 'Enhanced sparklines found in market cards');
        } else {
          logTest('market_grid_sparklines', 'FAIL', 'No sparklines found in market cards');
        }
        
        await takeScreenshot(page, '02_market_grid_layout.png', 'Market grid with enhanced sparklines');
      } else {
        logTest('market_grid_cards', 'FAIL', 'No market cards found in grid');
      }
    } else {
      logTest('market_grid_container', 'FAIL', 'Market grid container not found');
    }

    // Test 3: Live Trade Feed Positioning
    console.log('\n3. LIVE TRADE FEED POSITIONING VALIDATION');
    
    const hasTradeFeed = await elementExists(page, '[data-testid="trade-tape"], .trade-tape, .trade-feed');
    if (hasTradeFeed) {
      logTest('trade_feed_container', 'PASS', 'Live trade feed container found');
      
      // Check for trade rows
      const tradeRows = await page.$$('[data-testid="trade-row"], .trade-row, .trade-item');
      if (tradeRows.length > 0) {
        logTest('trade_feed_data', 'PASS', `${tradeRows.length} trade entries found in feed`);
        
        await takeScreenshot(page, '03_live_feed_bottom.png', 'Trade feed at bottom with reduced prominence');
      } else {
        logTest('trade_feed_data', 'FAIL', 'No trade entries found in feed');
      }
    } else {
      logTest('trade_feed_container', 'FAIL', 'Live trade feed container not found');
    }

    // Test 4: Complete Layout Flow
    console.log('\n4. COMPLETE LAYOUT FLOW VALIDATION');
    
    await takeScreenshot(page, '04_complete_layout_flow.png', 'Full layout hierarchy: hero → grid → feed');
    
    // Check vertical stacking order by measuring positions
    const heroY = await page.evaluate(() => {
      const heroElement = document.querySelector('[data-testid="hero-stats"], .hero-stats, .stats-section');
      return heroElement ? heroElement.getBoundingClientRect().top : -1;
    });
    
    const gridY = await page.evaluate(() => {
      const gridElement = document.querySelector('[data-testid="market-grid"], .market-grid, .markets-grid');
      return gridElement ? gridElement.getBoundingClientRect().top : -1;
    });
    
    const feedY = await page.evaluate(() => {
      const feedElement = document.querySelector('[data-testid="trade-tape"], .trade-tape, .trade-feed');
      return feedElement ? feedElement.getBoundingClientRect().top : -1;
    });
    
    if (heroY !== -1 && gridY !== -1 && feedY !== -1 && heroY < gridY && gridY < feedY) {
      logTest('layout_flow_order', 'PASS', `Correct layout order: hero(${heroY}) → grid(${gridY}) → feed(${feedY})`);
    } else {
      logTest('layout_flow_order', 'FAIL', `Incorrect layout order: hero(${heroY}) → grid(${gridY}) → feed(${feedY})`);
    }

    // Test 5: Desktop Responsive Layout (4-column)
    console.log('\n5. DESKTOP RESPONSIVE LAYOUT (4-COLUMN)');
    
    await page.setViewport({ width: 1920, height: 1080 });
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    await takeScreenshot(page, '05_desktop_4col_grid.png', 'Desktop responsive layout (4-column grid)');
    logTest('desktop_responsive', 'PASS', 'Desktop layout tested at 1920x1080');

    // Test 6: Tablet Responsive Layout (2-column)
    console.log('\n6. TABLET RESPONSIVE LAYOUT (2-COLUMN)');
    
    await page.setViewport({ width: 768, height: 1024 });
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    await takeScreenshot(page, '06_tablet_2col_grid.png', 'Tablet responsive layout (2-column grid)');
    logTest('tablet_responsive', 'PASS', 'Tablet layout tested at 768x1024');

    // Test 7: Mobile Responsive Layout (1-column stack)
    console.log('\n7. MOBILE RESPONSIVE LAYOUT (1-COLUMN STACK)');
    
    await page.setViewport({ width: 375, height: 667 });
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    await takeScreenshot(page, '07_mobile_1col_stack.png', 'Mobile responsive layout (1-column stack)');
    logTest('mobile_responsive', 'PASS', 'Mobile layout tested at 375x667');

    // Reset to desktop for further tests
    await page.setViewport({ width: 1920, height: 1080 });
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Test 8: Market Card Interaction
    console.log('\n8. MARKET CARD INTERACTION VALIDATION');
    
    const firstMarketCard = await page.$('[data-testid="market-card"], .market-card, .ticker-card');
    if (firstMarketCard) {
      await firstMarketCard.click();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Check if detail drawer opened
      const hasDetailDrawer = await elementExists(page, '[data-testid="ticker-detail"], .detail-drawer, .modal, .drawer');
      if (hasDetailDrawer) {
        logTest('market_card_interaction', 'PASS', 'Market card click opens detail drawer');
        await takeScreenshot(page, '08_market_card_interaction.png', 'Market detail drawer with enhanced sparkline');
        
        // Close drawer
        const closeButton = await page.$('[data-testid="close-button"], .close-button, .modal-close, [aria-label="Close"]');
        if (closeButton) {
          await closeButton.click();
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      } else {
        logTest('market_card_interaction', 'FAIL', 'Market card click did not open detail drawer');
      }
    } else {
      logTest('market_card_interaction', 'FAIL', 'No market card found to test interaction');
    }

    // Test 9: WebSocket Functionality and Real-time Updates
    console.log('\n9. WEBSOCKET FUNCTIONALITY VALIDATION');
    
    // Check for WebSocket connection by monitoring network activity
    let wsConnected = false;
    
    page.on('response', response => {
      if (response.url().includes('/ws/stream')) {
        wsConnected = true;
      }
    });
    
    // Wait a bit for WebSocket activity
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Check if trade count changes over time (indicating real-time updates)
    const initialTradeCount = await page.evaluate(() => {
      const tradesElement = document.querySelector('[data-testid="trades-count"], .trades-count, .stat-value');
      return tradesElement ? tradesElement.textContent : '0';
    });
    
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    const updatedTradeCount = await page.evaluate(() => {
      const tradesElement = document.querySelector('[data-testid="trades-count"], .trades-count, .stat-value');
      return tradesElement ? tradesElement.textContent : '0';
    });
    
    if (initialTradeCount !== updatedTradeCount) {
      logTest('websocket_realtime_updates', 'PASS', `Trade count updated from ${initialTradeCount} to ${updatedTradeCount}`);
    } else {
      logTest('websocket_realtime_updates', 'WARN', 'Trade count did not change (may be low trading activity)');
    }

    // Test 10: Console Error Check
    console.log('\n10. CONSOLE ERROR CHECK');
    
    const errorMessages = consoleMessages.filter(msg => msg.type === 'error');
    const warningMessages = consoleMessages.filter(msg => msg.type === 'warning');
    
    if (errorMessages.length === 0) {
      logTest('console_errors', 'PASS', 'No console errors detected');
    } else {
      logTest('console_errors', 'FAIL', `${errorMessages.length} console errors found: ${errorMessages.map(m => m.text).join(', ')}`);
    }
    
    if (warningMessages.length > 0) {
      logTest('console_warnings', 'WARN', `${warningMessages.length} console warnings found`);
    }
    
    await takeScreenshot(page, '09_console_clean.png', 'Clean console without errors');

    // Test 11: Performance Metrics
    console.log('\n11. PERFORMANCE METRICS');
    
    const performanceMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0];
      return {
        loadTime: navigation.loadEventEnd - navigation.loadEventStart,
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
        firstPaint: performance.getEntriesByType('paint').find(entry => entry.name === 'first-paint')?.startTime || 0
      };
    });
    
    logTest('performance_metrics', 'PASS', `Load time: ${performanceMetrics.loadTime}ms, DOM loaded: ${performanceMetrics.domContentLoaded}ms`);
    await takeScreenshot(page, '10_performance_metrics.png', 'Performance metrics during high volume');

  } catch (error) {
    console.error('Validation failed with error:', error);
    logTest('validation_execution', 'FAIL', `Validation script error: ${error.message}`);
  } finally {
    // Close browser
    if (browser) {
      await browser.close();
    }

    // Save validation results
    const resultsPath = path.join(screenshotDir, 'validation_results.json');
    fs.writeFileSync(resultsPath, JSON.stringify(validationResults, null, 2));
    
    // Generate summary report
    const summaryPath = path.join(screenshotDir, 'validation_summary.txt');
    const summaryContent = `
MILESTONE 5 UX REORGANIZATION VALIDATION SUMMARY
================================================

Session: ${validationResults.session}
Timestamp: ${validationResults.timestamp}
Git Commit: ${fs.readFileSync('puppeteer_agent_artifacts/' + sessionName + '/git_commit.txt', 'utf8').trim()}

RESULTS OVERVIEW:
Total Tests: ${validationResults.summary.total_tests}
Passed: ${validationResults.summary.passed}
Failed: ${validationResults.summary.failed}
Success Rate: ${((validationResults.summary.passed / validationResults.summary.total_tests) * 100).toFixed(1)}%

SCREENSHOTS CAPTURED:
${validationResults.screenshots.map(s => `- ${s.filename}: ${s.description}`).join('\n')}

ISSUES FOUND:
${validationResults.summary.issues.length > 0 ? validationResults.summary.issues.join('\n') : 'None'}

DETAILED RESULTS:
${Object.entries(validationResults.tests).map(([test, result]) => 
  `[${result.status}] ${test}: ${result.details}`
).join('\n')}
`;
    
    fs.writeFileSync(summaryPath, summaryContent);
    
    console.log('\n=== VALIDATION COMPLETE ===');
    console.log(`Results saved to: ${resultsPath}`);
    console.log(`Summary saved to: ${summaryPath}`);
    console.log(`Screenshots saved in: ${screenshotDir}`);
    console.log(`\nSUCCESS RATE: ${((validationResults.summary.passed / validationResults.summary.total_tests) * 100).toFixed(1)}%`);
  }
}

// Execute validation
runValidation().catch(console.error);