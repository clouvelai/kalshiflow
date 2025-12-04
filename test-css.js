import puppeteer from 'puppeteer';

(async () => {
  console.log('Starting browser to check Kalshi Flowboard CSS...');
  
  const browser = await puppeteer.launch({ 
    headless: false,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });
  
  console.log('Navigating to http://localhost:5173...');
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle2', timeout: 30000 });
  
  // Wait for content to load
  await page.waitForTimeout(3000);
  
  // Check if Tailwind CSS classes are being applied
  const styleCheck = await page.evaluate(() => {
    const body = document.body;
    const computedStyle = window.getComputedStyle(body);
    
    // Get all computed styles
    const bgColor = computedStyle.backgroundColor;
    const textColor = computedStyle.color;
    
    // Check for elements with Tailwind classes
    const elementsWithBg = document.querySelectorAll('[class*="bg-"]');
    const elementsWithText = document.querySelectorAll('[class*="text-"]');
    const elementsWithFlex = document.querySelectorAll('[class*="flex"]');
    const elementsWithGrid = document.querySelectorAll('[class*="grid"]');
    
    // Get the first few elements with classes for debugging
    const sampleElements = Array.from(document.querySelectorAll('*'))
      .filter(el => el.className && el.className.length > 0)
      .slice(0, 5)
      .map(el => ({
        tag: el.tagName.toLowerCase(),
        classes: el.className
      }));
    
    return {
      bodyBgColor: bgColor,
      bodyTextColor: textColor,
      elementsWithBg: elementsWithBg.length,
      elementsWithText: elementsWithText.length,
      elementsWithFlex: elementsWithFlex.length,
      elementsWithGrid: elementsWithGrid.length,
      sampleElements,
      hasContent: document.body.innerHTML.length > 100
    };
  });
  
  console.log('\n=== CSS Verification Results ===');
  console.log('Body Background Color:', styleCheck.bodyBgColor);
  console.log('Body Text Color:', styleCheck.bodyTextColor);
  console.log('Elements with bg- classes:', styleCheck.elementsWithBg);
  console.log('Elements with text- classes:', styleCheck.elementsWithText);
  console.log('Elements with flex classes:', styleCheck.elementsWithFlex);
  console.log('Elements with grid classes:', styleCheck.elementsWithGrid);
  console.log('Has Content:', styleCheck.hasContent);
  console.log('\nSample elements with classes:');
  styleCheck.sampleElements.forEach(el => {
    console.log(`  <${el.tag}>:`, el.classes);
  });
  
  // Take a screenshot
  const screenshotPath = '/tmp/kalshi-css-check.png';
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log('\nScreenshot saved to:', screenshotPath);
  
  // Check specific component rendering
  const componentCheck = await page.evaluate(() => {
    const header = document.querySelector('header');
    const main = document.querySelector('main');
    
    // Look for specific text content
    const hasKalshiText = document.body.textContent.includes('Kalshi');
    const hasConnectionText = document.body.textContent.includes('Connect') || 
                              document.body.textContent.includes('connect');
    
    return {
      hasHeader: header !== null,
      hasMain: main !== null,
      headerClasses: header ? header.className : 'no header',
      mainClasses: main ? main.className : 'no main',
      hasKalshiText,
      hasConnectionText,
      pageTitle: document.title
    };
  });
  
  console.log('\n=== Component Check ===');
  console.log('Has Header:', componentCheck.hasHeader);
  console.log('Header Classes:', componentCheck.headerClasses);
  console.log('Has Main:', componentCheck.hasMain);
  console.log('Main Classes:', componentCheck.mainClasses);
  console.log('Has Kalshi Text:', componentCheck.hasKalshiText);
  console.log('Has Connection Text:', componentCheck.hasConnectionText);
  console.log('Page Title:', componentCheck.pageTitle);
  
  // Keep browser open for 10 seconds to visually verify
  console.log('\nKeeping browser open for 10 seconds for visual verification...');
  await page.waitForTimeout(10000);
  
  await browser.close();
  console.log('\nBrowser closed. CSS check complete!');
})();