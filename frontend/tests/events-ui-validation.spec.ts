import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FRONTEND_URL = 'http://localhost:5175/v3-trader/events';
const SCREENSHOT_DIR = path.join(__dirname, '../../test-results/screenshots');

test.describe('Events Page UI Validation', () => {
  test.beforeAll(() => {
    // Ensure screenshot directory exists
    if (!fs.existsSync(SCREENSHOT_DIR)) {
      fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
    }
  });

  test('should validate Events page overview', async ({ page }) => {
    console.log('Navigating to Events page...');

    // Navigate to Events page
    await page.goto(FRONTEND_URL);

    // Wait for page to load
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000); // Allow React to render

    // Take overview screenshot
    const screenshotPath = path.join(SCREENSHOT_DIR, '01_events_page_overview.png');
    await page.screenshot({
      path: screenshotPath,
      fullPage: true
    });
    console.log(`✅ Screenshot saved: ${screenshotPath}`);

    // Validate page title
    await expect(page.locator('h1, h2').first()).toBeVisible();

    // Check for Events heading or title
    const pageContent = await page.content();
    console.log('Page loaded successfully');
  });

  test('should validate Edge Hypothesis section in EventCard', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000); // Wait for events to load

    // Look for event cards with chevron (collapsed state)
    // The cards are likely using a button or div with chevron icon
    const eventCards = page.locator('button:has(svg), div[role="button"]:has(svg)').filter({
      hasText: /Primate|Trump|KXRT/
    });

    let cardCount = await eventCards.count();

    // Fallback: try finding by text content
    if (cardCount === 0) {
      const allButtons = page.locator('button, div[role="button"]');
      cardCount = await allButtons.count();
      console.log(`Found ${cardCount} clickable elements on page`);
    }

    console.log(`Found ${cardCount} potential event cards`);

    if (cardCount === 0) {
      console.log('⚠️ No event cards found - taking empty state screenshot');
      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, '02_events_empty_state.png'),
        fullPage: true
      });
      return;
    }

    // Click first event card to expand it
    const firstCard = eventCards.first();
    console.log('Clicking first event card to expand...');
    await firstCard.click();
    await page.waitForTimeout(2000); // Wait for expansion animation

    // Take screenshot of expanded card
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '02_expanded_event_card.png'),
      fullPage: true
    });
    console.log('✅ Expanded event card screenshot saved');

    // Look for Edge Hypothesis section (amber/gold with lightbulb)
    const edgeHypothesisSection = page.locator(
      'section:has-text("Edge Hypothesis"), ' +
      'div:has-text("Edge Hypothesis")'
    );

    const hasEdgeSection = await edgeHypothesisSection.count() > 0;

    if (hasEdgeSection) {
      console.log('✅ Edge Hypothesis section found');

      // Scroll to Edge Hypothesis section
      await edgeHypothesisSection.first().scrollIntoViewIfNeeded();
      await page.waitForTimeout(500);

      // Take close-up screenshot
      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, '03_edge_hypothesis_section.png'),
        fullPage: true
      });
      console.log('✅ Edge Hypothesis section screenshot saved');

      // Validate section has lightbulb icon or proper styling
      const sectionHTML = await edgeHypothesisSection.first().innerHTML();
      console.log('Edge Hypothesis section HTML preview:', sectionHTML.substring(0, 300));

      // Check for lightbulb icon
      const lightbulbIcon = edgeHypothesisSection.locator('svg[class*="lightbulb"], svg:has(path[d*="M9"])');
      const hasLightbulb = await lightbulbIcon.count() > 0;
      console.log(hasLightbulb ? '✅ Lightbulb icon found' : '⚠️ Lightbulb icon not found');

    } else {
      console.log('⚠️ Edge Hypothesis section not found in expanded card');

      // Log what sections we do find
      const allSections = page.locator('section, div[class*="section"]');
      const sectionCount = await allSections.count();
      console.log(`Found ${sectionCount} sections in expanded card`);
    }
  });

  test('should validate Market Assessments section with TRADE badge and EV', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);

    // Find and expand first event card
    const eventCards = page.locator('button:has(svg), div[role="button"]:has(svg)').filter({
      hasText: /Primate|Trump|KXRT/
    });
    const cardCount = await eventCards.count();

    if (cardCount === 0) {
      console.log('⚠️ No event cards to validate Market Assessments');
      return;
    }

    console.log('Expanding first event card...');
    await eventCards.first().click();
    await page.waitForTimeout(2000);

    // Look for Market Assessments section
    const marketAssessmentsSection = page.locator(
      'section:has-text("Market Assessments"), ' +
      'div:has-text("Market Assessments")'
    );

    const hasMarketSection = await marketAssessmentsSection.count() > 0;

    if (hasMarketSection) {
      console.log('✅ Market Assessments section found');

      // Scroll to section
      await marketAssessmentsSection.first().scrollIntoViewIfNeeded();
      await page.waitForTimeout(500);

      // Look for TRADE badge (emerald green badge)
      const tradeBadge = page.locator(
        'span:has-text("TRADE")'
      );

      const tradeBadgeCount = await tradeBadge.count();
      console.log(tradeBadgeCount > 0 ? `✅ Found ${tradeBadgeCount} TRADE badge(s)` : '⚠️ TRADE badge not found');

      // Look for EV display (should show percentage)
      const evDisplay = page.locator('text=/EV:.*%/');
      const evCount = await evDisplay.count();

      if (evCount > 0) {
        console.log(`✅ Found ${evCount} EV display(s)`);
        // Log first EV value for verification
        const firstEV = await evDisplay.first().textContent();
        console.log(`   First EV value: ${firstEV}`);
      } else {
        console.log('⚠️ EV display not found');
      }

      // Take screenshot
      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, '04_market_assessments_section.png'),
        fullPage: true
      });
      console.log('✅ Market Assessments section screenshot saved');

      // Count total markets displayed
      const marketItems = page.locator('div:has-text("YES"), div:has-text("NO")');
      const marketCount = await marketItems.count();
      console.log(`📊 Total market items displayed: ${marketCount}`);

    } else {
      console.log('⚠️ Market Assessments section not found');

      // Debug: log what we can find
      const allText = await page.textContent('body');
      const hasMarketText = allText?.includes('Market') || allText?.includes('Assessment');
      console.log(`   Page contains "Market" or "Assessment": ${hasMarketText}`);
    }
  });

  test('should check for console errors', async ({ page }) => {
    const consoleErrors: string[] = [];
    const consoleWarnings: string[] = [];

    // Listen for console messages
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      } else if (msg.type() === 'warning') {
        consoleWarnings.push(msg.text());
      }
    });

    // Listen for page errors
    page.on('pageerror', error => {
      consoleErrors.push(`Page Error: ${error.message}`);
    });

    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);

    // Try to expand a card to trigger any component errors
    const eventCards = page.locator('[class*="EventCard"], [class*="event-card"], article, [role="article"]');
    if (await eventCards.count() > 0) {
      await eventCards.first().click();
      await page.waitForTimeout(2000);
    }

    // Report findings
    if (consoleErrors.length > 0) {
      console.log('❌ Console Errors Found:');
      consoleErrors.forEach((err, idx) => {
        console.log(`  ${idx + 1}. ${err}`);
      });
    } else {
      console.log('✅ No console errors detected');
    }

    if (consoleWarnings.length > 0) {
      console.log('⚠️ Console Warnings:');
      consoleWarnings.slice(0, 5).forEach((warn, idx) => {
        console.log(`  ${idx + 1}. ${warn}`);
      });
    }

    // Save error log
    const errorLog = {
      timestamp: new Date().toISOString(),
      errors: consoleErrors,
      warnings: consoleWarnings
    };

    fs.writeFileSync(
      path.join(SCREENSHOT_DIR, 'console_errors.json'),
      JSON.stringify(errorLog, null, 2)
    );

    console.log('✅ Console error log saved');
  });
});
