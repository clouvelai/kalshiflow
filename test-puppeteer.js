const puppeteer = require('puppeteer');

(async () => {
  console.log('🚀 Starting Puppeteer test...');
  
  const browser = await puppeteer.launch({ 
    headless: false,
    defaultViewport: { width: 1280, height: 720 }
  });
  
  const page = await browser.newPage();
  
  try {
    console.log('📱 Navigating to Kalshi Flowboard...');
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle0', timeout: 10000 });
    
    console.log('✅ Page loaded successfully');
    
    // Wait a moment to see the app
    await page.waitForTimeout(3000);
    
    // Take a screenshot
    await page.screenshot({ path: 'kalshi-flowboard-test.png', fullPage: true });
    console.log('📸 Screenshot saved as kalshi-flowboard-test.png');
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  } finally {
    await browser.close();
    console.log('🏁 Test complete');
  }
})();