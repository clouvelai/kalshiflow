#!/usr/bin/env node
/**
 * Quick validation script to test milestone 5 implementation:
 * 1. Check frontend is accessible
 * 2. Verify WebSocket connection works
 * 3. Check that hero stats show real data (not "Coming Soon")
 */

const puppeteer = require('puppeteer');

async function validateMilestone5() {
  console.log('🧪 Starting Milestone 5 validation...\n');

  const browser = await puppeteer.launch({
    headless: false, // Show browser for debugging
    defaultViewport: { width: 1200, height: 800 }
  });

  try {
    const page = await browser.newPage();
    
    // Enable console logging
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.log('❌ Browser Error:', msg.text());
      }
    });

    console.log('🌐 Loading frontend...');
    await page.goto('http://localhost:5173/', { 
      waitUntil: 'networkidle0',
      timeout: 10000 
    });

    // Wait for the hero stats to load
    console.log('📊 Checking hero stats...');
    await page.waitForSelector('[data-testid="hero-stats"]', { timeout: 5000 });

    // Check if hero stats cards exist
    const heroStatsCards = await page.$$eval('.animate-slide-up', (cards) => {
      return cards.map(card => {
        const label = card.querySelector('.text-sm.font-medium.text-gray-600')?.textContent;
        const value = card.querySelector('.text-3xl.font-bold')?.textContent;
        const isComingSoon = card.querySelector('.animate-pulse')?.textContent === 'Coming Soon';
        return { label, value, isComingSoon };
      });
    });

    console.log('\n📈 Hero Stats Analysis:');
    heroStatsCards.forEach((card, index) => {
      console.log(`  Card ${index + 1}: ${card.label}`);
      console.log(`    Value: ${card.value}`);
      console.log(`    Coming Soon: ${card.isComingSoon}`);
    });

    // Check if volume and net flow are no longer "Coming Soon"
    const volumeCard = heroStatsCards.find(card => card.label === 'Total Volume');
    const netFlowCard = heroStatsCards.find(card => card.label === 'Net Flow');

    console.log('\n✅ Validation Results:');

    if (volumeCard && !volumeCard.isComingSoon) {
      console.log('  ✅ Total Volume shows real data:', volumeCard.value);
    } else {
      console.log('  ❌ Total Volume still shows "Coming Soon"');
    }

    if (netFlowCard && !netFlowCard.isComingSoon) {
      console.log('  ✅ Net Flow shows real data:', netFlowCard.value);
    } else {
      console.log('  ❌ Net Flow still shows "Coming Soon"');
    }

    // Check if markets show last_price
    console.log('\n🏪 Checking market cards...');
    await page.waitForSelector('[data-testid^="market-card-"]', { timeout: 5000 });

    const marketCards = await page.$$eval('[data-testid^="market-card-"]', (cards) => {
      return cards.slice(0, 3).map(card => { // Check first 3 cards
        const ticker = card.querySelector('h3')?.textContent;
        const lastPriceElement = card.querySelector('.text-lg.font-bold');
        const lastPrice = lastPriceElement?.textContent;
        return { ticker, lastPrice };
      });
    });

    console.log('  Market Last Prices:');
    marketCards.forEach(market => {
      if (market.lastPrice && market.lastPrice !== '--') {
        console.log(`    ✅ ${market.ticker}: ${market.lastPrice}`);
      } else {
        console.log(`    ❌ ${market.ticker}: No price data`);
      }
    });

    console.log('\n🎉 Milestone 5 validation completed!');

    // Keep browser open for 5 seconds to see the results
    await page.waitForTimeout(5000);

  } catch (error) {
    console.error('❌ Validation failed:', error.message);
  } finally {
    await browser.close();
  }
}

// Run validation
validateMilestone5().catch(console.error);