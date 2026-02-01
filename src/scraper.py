import os
import pandas as pd
import time
import random
import logging
import traceback
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
    UnexpectedAlertPresentException,
    InvalidSessionIdException,
    WebDriverException
)


class JumiaRetailScraper:
    """
    A fully self-healing, stealth-optimized, headless scraper for Jumia Nigeria.
    Handles invalid WebDriver sessions, auto-restarts, popups, and exports reviews by category.
    """

    def __init__(self, driver_path, user_agent):
        # Save initialization parameters
        self.driver_path = driver_path
        self.user_agent = user_agent

        # Temporary buffer for storing scraped reviews
        self.results = []

        # Browser and WebDriverWait objects will be initialized later
        self.browser = None
        self.wait = None

        # ---------- LOGGING CONFIGURATION ----------
        # Create a folder for logs if not present
        os.makedirs("logs", exist_ok=True)

        # Configure logging to file + console
        logging.basicConfig(
            filename="logs/jumia_scraper.log",   # Log file name
            filemode="a",                        # Append mode
            level=logging.INFO,                  # Logging level
            format="%(asctime)s [%(levelname)s]: %(message)s"
        )
        self.logger = logging.getLogger()

        # Initialize the browser immediately when scraper is created
        self._init_browser()

    def _init_browser(self):
        """Initialize Edge WebDriver in stealth + headless mode."""
        try:
            # Close existing browser instance if already running
            if self.browser:
                try:
                    self.browser.quit()
                except Exception:
                    pass

            self.logger.info("Starting new Edge WebDriver session...")

            # ---------- EDGE OPTIONS ----------
            edge_options = Options()
            edge_options.add_argument(f"user-agent={self.user_agent}")  # Custom user-agent
            edge_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            edge_options.add_argument("--disable-blink-features=AutomationControlled")  # Hide automation flag
            edge_options.add_argument("--disable-features=RendererCodeIntegrity")       # Avoid rendering crashes
            edge_options.add_argument("--disable-infobars")
            edge_options.add_argument("--no-sandbox")
            edge_options.add_argument("--disable-dev-shm-usage")
            edge_options.add_argument("--disable-gpu")
            edge_options.add_argument("--disable-extensions")
            edge_options.add_argument("--disable-notifications")
            edge_options.add_argument("--log-level=3")               # Suppress ChromeDriver logs
            edge_options.add_argument("--headless=new")              # Run in headless mode (no GUI)
            edge_options.add_argument("--window-size=1920,1080")     # Ensure full page loads

            # Disable SmartScreen and safe browsing popups
            prefs = {
                "safebrowsing.enabled": False,
                "safebrowsing.disable_download_protection": True,
                "profile.block_third_party_cookies": True
            }
            edge_options.add_experimental_option("prefs", prefs)

            # ---------- SERVICE ----------
            # Direct EdgeDriver logs to null to avoid console spam
            service = Service(executable_path=self.driver_path, log_path=os.devnull)

            # Launch the browser
            self.browser = webdriver.Edge(service=service, options=edge_options)

            # Inject JavaScript to spoof navigator properties (anti-detection)
            self.browser.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                    Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4]});
                """
            })

            # Explicit wait for dynamic elements
            self.wait = WebDriverWait(self.browser, 15)
            self.logger.info("Edge WebDriver initialized successfully.")

        except WebDriverException as e:
            self.logger.error(f"Failed to start WebDriver: {e}")
            raise

    def _random_delay(self, low=1.2, high=3.5):
        """Add randomized sleep intervals to mimic human browsing."""
        time.sleep(random.uniform(low, high))

    def _session_guard(self, func, *args, **kwargs):
        """
        Wraps Selenium operations to catch 'InvalidSessionIdException'.
        If browser crashes, re-initialize and retry once.
        """
        try:
            return func(*args, **kwargs)
        except InvalidSessionIdException:
            self.logger.error("WebDriver session lost. Restarting browser...")
            self._init_browser()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Retry failed after reinit: {e}")
                return None

    def navigate_home_and_clear_popups(self):
        """Open Jumia home page and close popups like newsletter and cookie consent."""
        def _inner():
            self.logger.info("Navigating to Jumia homepage...")
            self.browser.get("https://www.jumia.com.ng")
            self._random_delay(2, 4)

            # Try to dismiss browser alerts (if any)
            try:
                alert = self.browser.switch_to.alert
                self.logger.warning(f"Unexpected alert detected: {alert.text}")
                alert.accept()
            except Exception:
                pass

            # Close the newsletter modal popup if present
            try:
                newsletter_close = self.wait.until(EC.element_to_be_clickable(
                    (By.XPATH, "//button[@aria-label='newsletter_popup_close-cta'] | //div[contains(@class, 'cls')]")
                ))
                newsletter_close.click()
                self.logger.info("Newsletter popup closed successfully.")
            except TimeoutException:
                self.logger.info("No newsletter popup detected.")

            # Accept cookie consent if banner is visible
            try:
                cookie_accept = self.wait.until(EC.element_to_be_clickable(
                    (By.XPATH, "//button[@id='cookies-accept-all'] | //button[contains(text(), 'Accept')]")
                ))
                cookie_accept.click()
                self.logger.info("Cookie consent accepted.")
            except TimeoutException:
                self.logger.info("Cookie consent not found or already dismissed.")

        # Run inside session guard for auto-retry safety
        self._session_guard(_inner)

    def discover_products(self, cat_name):
        """Extracts product URLs from a category page."""
        found_links = []

        def _inner():
            # Wait until product grid loads
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "prd")))
            soup = BeautifulSoup(self.browser.page_source, "html.parser")

            # Extract product <a> tags
            for art in soup.find_all("article", class_="prd"):
                tag = art.find("a", class_="core")
                if tag and tag.get("href"):
                    url = tag["href"]
                    # Convert relative path to absolute URL
                    if url.startswith("/"):
                        url = "https://www.jumia.com.ng" + url
                    found_links.append(url)

            self.logger.info(f"Found {len(found_links)} product URLs in category '{cat_name}'")

        # Run within session guard to handle browser crashes
        self._session_guard(_inner)
        return found_links

    def extract_reviews(self, product_url, category):
        """Scrape all customer reviews for a single product."""
        self.logger.info(f"[SCRAPE] Product: {product_url}")

        def _inner():
            # Open product page
            self.browser.get(product_url)
            self._random_delay(1.5, 3)

            # Try to open the "See All Reviews" section
            try:
                see_all = self.wait.until(EC.element_to_be_clickable(
                    (By.XPATH, "//a[contains(@href, '/reviews/')]")
                ))
                see_all.click()
                self._random_delay(1.5, 3)
            except TimeoutException:
                self.logger.info("No 'See All Reviews' button found for this product.")
                return

            # Loop through all review pages (pagination)
            while True:
                self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))
                soup = BeautifulSoup(self.browser.page_source, "html.parser")
                reviews = soup.find_all("article", class_="-pvs -border _pld")

                # Extract individual reviews
                for rev in reviews:
                    # Default rating is 0
                    rating = 0
                    stars_div = rev.find("div", class_="stars")
                    if stars_div:
                        for cls in stars_div.get("class", []):
                            if cls.startswith("_") and cls[1:].isdigit():
                                rating = int(cls[1:])

                    # Check for "Verified Purchase" badge
                    verified = bool(rev.find(text="Verified Purchase") or rev.find("div", class_="bdg _vrf"))

                    # Extract username (if available)
                    user_name = "Anonymous"
                    meta_info = rev.find("div", class_="-hr -pvs")
                    if meta_info:
                        spans = meta_info.find_all("span")
                        if spans:
                            user_name = spans[0].get_text(strip=True).replace("by ", "")

                    # Extract timestamp and review text
                    date_val = rev.find("span", class_="-df -i-ctr -fs12 -pts")
                    timestamp = date_val.get_text(strip=True) if date_val else "N/A"
                    text_body = rev.find("p", class_="-pvs")
                    review_text = text_body.get_text(strip=True) if text_body else ""

                    # Append to results
                    self.results.append({
                        "Category": category,
                        "Product_URL": product_url,
                        "User_Name": user_name,
                        "Rating": rating,
                        "Timestamp": timestamp,
                        "Verified_Badge": verified,
                        "Review_Text": review_text
                    })

                # Try to move to next page of reviews
                try:
                    next_btn = self.browser.find_element(By.CSS_SELECTOR, "a.pg[aria-label='Next Page']")
                    self.browser.execute_script("arguments[0].scrollIntoView(true);", next_btn)
                    self._random_delay(1, 2)
                    next_btn.click()
                    self._random_delay(1.5, 3)
                except (NoSuchElementException, ElementClickInterceptedException):
                    break  # Exit loop if no next page found

        # Run with session guard to auto-restart browser if needed
        self._session_guard(_inner)

    def export_data(self, category):
        """Save scraped reviews to a CSV file per category."""
        if not self.results:
            self.logger.warning(f"No reviews collected for category '{category}'")
            return

        os.makedirs("data", exist_ok=True)
        filename = f"data/jumia_reviews_{category.lower().replace(' ', '_')}.csv"

        # Convert to DataFrame and export
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        self.logger.info(f"Exported {len(df)} reviews to {filename}")

        # Clear results after export to avoid duplicates
        self.results.clear()

    def shutdown(self):
        """Safely close the WebDriver instance."""
        try:
            if self.browser:
                self.browser.quit()
        except InvalidSessionIdException:
            self.logger.warning("Browser session already terminated.")
        self.logger.info("Edge browser closed successfully.")


def run_scraper():
    """Main entry point for the scraper with auto-restart logic."""
    #  USER CONFIG 
    PATH_TO_DRIVER = r"C:\Users\OLALERE\Desktop\Books\edgedriver_win64\msedgedriver.exe"
    MY_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0"

    # Define categories to scrape
    TARGET_CATEGORIES = {
        "Mobile Phones": "https://www.jumia.com.ng/mobile-phones/",
        "Computing": "https://www.jumia.com.ng/computing/",
        "Electronics": "https://www.jumia.com.ng/electronics/"
    }

    # AUTO-RESTART LOOP 
    while True:
        try:
            # Initialize scraper
            jumia_bot = JumiaRetailScraper(PATH_TO_DRIVER, MY_USER_AGENT)
            jumia_bot.navigate_home_and_clear_popups()

            # Loop through each category
            for cat_name, cat_url in TARGET_CATEGORIES.items():
                jumia_bot.logger.info(f"--- Starting category: {cat_name} ---")
                jumia_bot._session_guard(jumia_bot.browser.get, cat_url)

                # Discover product links
                links = jumia_bot.discover_products(cat_name)

                # Iterate over products
                for i, link in enumerate(links):
                    jumia_bot.extract_reviews(link, cat_name)

                    # Refresh browser every 20 products for stability
                    if i % 20 == 0:
                        jumia_bot.logger.info("Refreshing Edge session to prevent timeouts...")
                        jumia_bot._init_browser()

                # Export category data
                jumia_bot.export_data(cat_name)

            # Graceful shutdown when all categories complete
            jumia_bot.shutdown()
            break  # Exit loop

        except Exception as e:
            # Catch any unexpected error and restart after a delay
            logging.error(f"Fatal error, restarting scraper: {e}")
            traceback.print_exc()
            if 'bot' in locals():
                jumia_bot.shutdown()

            # Wait 1–3 minutes before restarting
            delay = random.randint(60, 180)
            logging.info(f"Restarting after {delay} seconds...")
            time.sleep(delay)


if __name__ == "__main__":
    run_scraper()
