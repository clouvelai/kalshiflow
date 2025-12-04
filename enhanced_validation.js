const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

// Read session name
const sessionName = fs.readFileSync('puppeteer_agent_artifacts/current_session.txt', 'utf8').trim();
const screenshotDir = `puppeteer_agent_artifacts/${sessionName}`;

console.log('Enhanced Milestone 5 Validation with Correct Selectors');

async function runEnhancedValidation() {
  let browser;
  
  try {
    browser = await puppeteer.launch({
      headless: false,
      defaultViewport: { width: 1920, height: 1080 },
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();
    
    // Capture console messages with better error details
    const consoleMessages = [];
    page.on('console', (msg) => {
      const message = {
        type: msg.type(),
        text: msg.text(),
        location: msg.location(),
        timestamp: new Date().toISOString()
      };
      consoleMessages.push(message);
      console.log(`[CONSOLE ${msg.type().toUpperCase()}] ${msg.text()}`);
    });

    // Capture network errors
    const networkErrors = [];
    page.on('requestfailed', request => {
      networkErrors.push({
        url: request.url(),
        error: request.failure().errorText,
        timestamp: new Date().toISOString()
      });
      console.log(`[NETWORK ERROR] ${request.url()}: ${request.failure().errorText}`);
    });

    console.log('Navigating to application...');
    await page.goto('http://localhost:5173', { 
      waitUntil: 'networkidle2',
      timeout: 30000 
    });

    await new Promise(resolve => setTimeout(resolve, 5000));

    console.log('\n=== ENHANCED MILESTONE 5 VALIDATION ===\n');

    // Test 1: Hero Stats with Correct Selectors
    console.log('1. ENHANCED HERO STATS VALIDATION');
    
    // Wait for hero stats to load
    await page.waitForSelector('h2', { timeout: 10000 });
    
    const heroStatsText = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      const results = {
        hasMarketOverview: false,
        tradesToday: null,
        totalVolume: null,
        netFlow: null,
        statsCards: []
      };
      
      // Look for Market Overview header
      for (const element of elements) {
        if (element.textContent && element.textContent.includes('Market Overview')) {
          results.hasMarketOverview = true;
        }
        if (element.textContent && element.textContent.includes('Trades Today')) {
          results.tradesToday = element.textContent;
        }
        if (element.textContent && element.textContent.includes('Total Volume')) {
          results.totalVolume = element.textContent;
        }
        if (element.textContent && element.textContent.includes('Net Flow')) {
          results.netFlow = element.textContent;
        }
      }
      
      return results;
    });
    
    console.log('Hero Stats Found:', heroStatsText);
    
    if (heroStatsText.hasMarketOverview) {
      console.log('✅ Market Overview section found');
    }
    if (heroStatsText.tradesToday) {
      console.log('✅ Trades Today metric found');
    }
    if (heroStatsText.totalVolume) {
      console.log('✅ Total Volume metric found');
    }
    if (heroStatsText.netFlow) {
      console.log('✅ Net Flow metric found');
    }

    // Test 2: Market Cards Validation
    console.log('\n2. ENHANCED MARKET GRID VALIDATION');
    
    const marketData = await page.evaluate(() => {
      const results = {
        hotMarketsTitle: false,
        marketCount: 0,
        sparklines: 0,
        marketTickers: [],
        hasPrices: false,
        hasVolumes: false
      };
      
      // Look for Hot Markets title
      const elements = document.querySelectorAll('*');
      for (const element of elements) {
        if (element.textContent && element.textContent.includes('Hot Markets')) {
          results.hotMarketsTitle = true;
        }
      }
      
      // Count SVG sparklines (enhanced sparklines)
      results.sparklines = document.querySelectorAll('svg').length;
      
      // Look for market tickers and data
      const textElements = Array.from(elements)
        .map(el => el.textContent)
        .filter(text => text && text.length > 0);
        
      // Count potential market tickers (short uppercase codes)
      const tickerPattern = /^[A-Z0-9]{3,15}$/;
      results.marketTickers = textElements.filter(text => 
        tickerPattern.test(text.trim()) && text.length < 20
      );
      
      // Check for price indicators
      results.hasPrices = textElements.some(text => 
        text.includes('¢') || text.includes('$') || /\d+¢/.test(text)
      );
      
      // Check for volume indicators  
      results.hasVolumes = textElements.some(text => 
        text.includes('Volume') || /\$[\d,]+/.test(text)
      );
      
      return results;
    });
    
    console.log('Market Grid Data:', marketData);
    
    if (marketData.hotMarketsTitle) {
      console.log('✅ Hot Markets section found');
    }
    if (marketData.sparklines > 0) {
      console.log(`✅ ${marketData.sparklines} SVG sparklines found`);
    }
    if (marketData.marketTickers.length > 0) {
      console.log(`✅ ${marketData.marketTickers.length} market tickers found`);
    }

    // Test 3: Live Trade Feed Validation
    console.log('\n3. ENHANCED TRADE FEED VALIDATION');
    
    const tradeFeedData = await page.evaluate(() => {
      const results = {
        tradeFeedTitle: false,
        tradeEntries: 0,
        hasTimestamps: false,
        hasMarketNames: false,
        hasPrices: false,
        hasSides: false
      };
      
      const elements = document.querySelectorAll('*');
      const textContent = Array.from(elements).map(el => el.textContent).join(' ');
      
      // Look for trade feed indicators
      if (textContent.includes('Live Trade Feed') || textContent.includes('Recent Trades')) {
        results.tradeFeedTitle = true;
      }
      
      // Look for time patterns (HH:MM:SS)
      const timePattern = /\d{2}:\d{2}:\d{2}/g;
      const timeMatches = textContent.match(timePattern);
      results.hasTimestamps = timeMatches && timeMatches.length > 0;
      
      // Look for YES/NO indicators
      results.hasSides = textContent.includes('YES') || textContent.includes('NO');
      
      // Count potential trade entries
      if (timeMatches) {
        results.tradeEntries = timeMatches.length;
      }
      
      return results;
    });
    
    console.log('Trade Feed Data:', tradeFeedData);
    
    if (tradeFeedData.tradeFeedTitle) {
      console.log('✅ Live Trade Feed section found');
    }
    if (tradeFeedData.tradeEntries > 0) {
      console.log(`✅ ${tradeFeedData.tradeEntries} trade entries found`);
    }
    if (tradeFeedData.hasTimestamps) {
      console.log('✅ Timestamps found in trade feed');
    }

    // Test 4: Market Card Click Test
    console.log('\n4. ENHANCED MARKET INTERACTION TEST');
    
    try {
      // Find and click a market card area
      const marketCard = await page.evaluateHandle(() => {
        const elements = document.querySelectorAll('*');
        for (const element of elements) {
          // Look for elements containing market ticker patterns
          if (element.textContent && /^[A-Z0-9]{3,15}$/.test(element.textContent.trim())) {
            // Find clickable parent
            let parent = element.parentElement;
            while (parent && parent !== document.body) {
              const style = window.getComputedStyle(parent);
              if (style.cursor === 'pointer' || parent.onclick || parent.getAttribute('role') === 'button') {
                return parent;
              }
              parent = parent.parentElement;
            }
          }
        }
        return null;
      });
      
      if (marketCard) {
        await marketCard.click();
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Check if modal/drawer opened
        const modalOpened = await page.evaluate(() => {
          const modals = document.querySelectorAll('[role="dialog"], .modal, .drawer');
          return modals.length > 0;
        });
        
        if (modalOpened) {
          console.log('✅ Market detail drawer opened successfully');
          await page.screenshot({ 
            path: path.join(screenshotDir, '11_market_interaction_success.png'),
            fullPage: true 
          });
          
          // Close modal
          await page.keyboard.press('Escape');
          await new Promise(resolve => setTimeout(resolve, 1000));
        } else {
          console.log('❌ Market click did not open drawer');
        }
      } else {
        console.log('❌ No clickable market card found');
      }
    } catch (error) {
      console.log(`❌ Market interaction error: ${error.message}`);
    }

    // Test 5: Real-time Updates Test
    console.log('\n5. REAL-TIME UPDATES TEST');
    
    const initialData = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      const textContent = Array.from(elements).map(el => el.textContent).join(' ');
      
      // Extract trade count
      const tradeMatch = textContent.match(/(\d{1,3}(,\d{3})*)\s*Trades\s*Today/);
      const volumeMatch = textContent.match(/\$(\d+\.\d{2}M)/);
      
      return {
        tradeCount: tradeMatch ? tradeMatch[1] : null,
        volume: volumeMatch ? volumeMatch[1] : null,
        timestamp: new Date().toISOString()
      };
    });
    
    console.log('Initial Data:', initialData);
    
    // Wait for updates
    console.log('Waiting 10 seconds for real-time updates...');
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    const updatedData = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      const textContent = Array.from(elements).map(el => el.textContent).join(' ');
      
      const tradeMatch = textContent.match(/(\d{1,3}(,\d{3})*)\s*Trades\s*Today/);
      const volumeMatch = textContent.match(/\$(\d+\.\d{2}M)/);
      
      return {
        tradeCount: tradeMatch ? tradeMatch[1] : null,
        volume: volumeMatch ? volumeMatch[1] : null,
        timestamp: new Date().toISOString()
      };
    });
    
    console.log('Updated Data:', updatedData);
    
    if (initialData.tradeCount !== updatedData.tradeCount) {
      console.log('✅ Real-time trade count updates confirmed');
    } else {
      console.log('⚠️  Trade count unchanged (may be low activity period)');
    }

    // Final screenshot
    await page.screenshot({ 
      path: path.join(screenshotDir, '12_final_enhanced_validation.png'),
      fullPage: true 
    });

    // Console Error Analysis
    console.log('\n6. CONSOLE ERROR ANALYSIS');
    
    const errorMessages = consoleMessages.filter(msg => msg.type === 'error');
    const warningMessages = consoleMessages.filter(msg => msg.type === 'warning');
    
    console.log(`Console Errors: ${errorMessages.length}`);
    console.log(`Console Warnings: ${warningMessages.length}`);
    
    errorMessages.forEach(err => {
      console.log(`❌ ERROR: ${err.text}`);
      if (err.location) {
        console.log(`   Location: ${err.location.url}:${err.location.lineNumber}`);
      }
    });
    
    warningMessages.forEach(warn => {
      console.log(`⚠️  WARNING: ${warn.text}`);
    });
    
    // Network Error Analysis
    console.log(`\nNetwork Errors: ${networkErrors.length}`);
    networkErrors.forEach(err => {
      console.log(`❌ NETWORK: ${err.url} - ${err.error}`);
    });
    
    // Save detailed results
    const enhancedResults = {
      session: sessionName,
      timestamp: new Date().toISOString(),
      validation_type: 'enhanced',
      results: {
        hero_stats: heroStatsText,
        market_grid: marketData,
        trade_feed: tradeFeedData,
        initial_data: initialData,
        updated_data: updatedData,
        console_errors: errorMessages,
        console_warnings: warningMessages,
        network_errors: networkErrors
      }
    };
    
    fs.writeFileSync(
      path.join(screenshotDir, 'enhanced_validation_results.json'),
      JSON.stringify(enhancedResults, null, 2)
    );
    
    console.log('\n=== ENHANCED VALIDATION COMPLETE ===');
    console.log('Results saved to enhanced_validation_results.json');

  } catch (error) {
    console.error('Enhanced validation failed:', error);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

runEnhancedValidation().catch(console.error);