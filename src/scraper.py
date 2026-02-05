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
    InvalidSessionIdException,
    WebDriverException,
    StaleElementReferenceException
)

class JumiaRetailScraper:
    """
    Enterprise-grade scraper for Jumia Nigeria.
    
    Architectural Highlights:
    - Namespace-Agnostic Pagination: Uses ARIA labels instead of SVG attributes.
    - SPA Awareness: Handles dynamic DOM updates and stale elements.
    - Anti-Bot Stealth: Uses CDP injection and randomized human-like delays.
    - Robust I/O: Atomic write operations for CSV data safety.
    """

    def __init__(self, driver_path, user_agent):
        self.driver_path = driver_path
        self.user_agent = user_agent
        self.results = []   # Storage for scraped review data
        self.browser = None
        self.wait = None

        # --- Setup Logging ---
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename="logs/jumia_scraper.log",
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s"
        )
        self.logger = logging.getLogger()
        
        # --- Initialize the browser ---
        self._init_browser()

    def _init_browser(self):
        """
        Initializes Microsoft Edge WebDriver with stealth configuration
        and human-like behaviors.
        """
        try:
            if self.browser:
                try:
                    self.browser.quit()
                except Exception:
                    pass

            self.logger.info("Initializing Stealth Edge WebDriver session...")

            edge_options = Options()
            edge_options.add_argument(f"user-agent={self.user_agent}")
            edge_options.add_argument("--disable-blink-features=AutomationControlled")  # Stealth
            edge_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            edge_options.add_argument("--headless=new")  # Run headless
            edge_options.add_argument("--disable-gpu")
            edge_options.add_argument("--window-size=1920,1080")
            edge_options.add_argument("--disable-extensions")

            # Initialize WebDriver service
            service = Service(executable_path=self.driver_path, log_path=os.devnull)
            self.browser = webdriver.Edge(service=service, options=edge_options)

            # CDP injection: hides the 'navigator.webdriver' property
            self.browser.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            })

            self.wait = WebDriverWait(self.browser, 20)  # Increased timeout
            self.logger.info("WebDriver initialized successfully.")

        except WebDriverException as e:
            self.logger.critical(f"Fatal WebDriver Initialization Error: {e}")
            raise

    def _random_delay(self, low=2.0, high=5.0):
        """Adds randomized human-like delays to avoid bot detection."""
        time.sleep(random.uniform(low, high))

    def _session_guard(self, func, *args, **kwargs):
        """
        Auto-retries function calls if WebDriver session crashes.
        """
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (InvalidSessionIdException, WebDriverException) as e:
                if attempt < max_retries:
                    self.logger.warning(f"Session crash detected ({e}). Re-initializing (Attempt {attempt+1})...")
                    self._init_browser()
                    time.sleep(5)
                else:
                    self.logger.error(f"Operation failed after {max_retries} retries: {e}")
                    return None
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                traceback.print_exc()
                return None

    def navigate_home_and_clear_popups(self):
        """Navigates to homepage and closes popups like cookies and newsletters."""
        def _inner():
            self.logger.info("Navigating to Jumia homepage...")
            self.browser.get("https://www.jumia.com.ng")
            self._random_delay(3, 5)

            # --- Close Newsletter Popups ---
            try:
                close_btn = self.wait.until(EC.element_to_be_clickable((
                    By.XPATH, "//button[contains(@aria-label, 'close') or contains(@class, 'cls')]"
                )))
                close_btn.click()
                self.logger.info("Popup dismissed.")
            except TimeoutException:
                self.logger.debug("No newsletter popup found.")

            # --- Accept Cookies ---
            try:
                cookie_btn = self.wait.until(EC.element_to_be_clickable((
                    By.XPATH, "//button[contains(text(), 'Accept') or @id='cookies-accept-all']"
                )))
                cookie_btn.click()
                self.logger.info("Cookies accepted.")
            except TimeoutException:
                pass

        self._session_guard(_inner)

    def discover_products(self, cat_name):
        """Scrapes product URLs from a category listing page."""
        found_links = []
        def _inner():
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "prd")))

            # Execute JS for faster extraction
            links = self.browser.execute_script("""
                var links = [];
                document.querySelectorAll('article.prd a.core').forEach(a => links.push(a.href));
                return links;
            """)
            found_links.extend(links)
            self.logger.info(f"Found {len(links)} products in '{cat_name}'")

        self._session_guard(_inner)
        return found_links

    def extract_reviews(self, product_url, category):
        """
        Main review extraction logic.
        Fixes NamespaceError by using ARIA labels for pagination.
        """
        self.logger.info(f"[Processing] {product_url}")

        def _inner():
            self.browser.get(product_url)
            self._random_delay(2, 4)

            # --- Click 'See All Reviews' ---
            try:
                see_all = self.wait.until(EC.element_to_be_clickable((
                    By.PARTIAL_LINK_TEXT, "See All"
                )))
                # Scroll into view & click safely
                self.browser.execute_script("arguments[0].scrollIntoView({block: 'center'});", see_all)
                self._random_delay(1, 1.5)
                try:
                    see_all.click()
                except ElementClickInterceptedException:
                    self.browser.execute_script("arguments[0].click();", see_all)
            except TimeoutException:
                self.logger.warning("No 'See All Reviews' link found. Skipping.")
                return

            page_num = 1
            total_collected = 0

            while True:
                try:
                    self.wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "article")))
                    soup = BeautifulSoup(self.browser.page_source, "html.parser")
                    reviews = soup.find_all("article")
                    if not reviews:
                        break

                    # --- Extract Review Data ---
                    for rev in reviews:
                        # Rating extraction
                        rating = "N/A"
                        stars_div = rev.find("div", class_="stars")
                        if stars_div:
                            rating = stars_div.get_text(strip=True).split(" ")

                        # Title & body
                        review_title = rev.find("h3").get_text(strip=True) if rev.find("h3") else ""
                        review_text = rev.find("p", class_="-pvs").get_text(strip=True) if rev.find("p", class_="-pvs") else ""

                        # Meta info: date & username
                        date_val, user_name = "N/A", "Anonymous"
                        meta_section = rev.find("div", class_="-pvs")
                        if meta_section:
                            spans = meta_section.find_all("span")
                            if len(spans) >= 2:
                                date_val = spans[0].get_text(strip=True)
                                user_name = spans[1].get_text(strip=True).replace("by ", "")
                            elif len(spans) == 1:
                                date_val = spans[0].get_text(strip=True)

                        verified = "Verified Purchase" in rev.get_text()

                        # Append to results
                        self.results.append({
                            "Category": category,
                            "Product_URL": product_url,
                            "User_Name": user_name,
                            "Rating": rating,
                            "Review_Title": review_title,
                            "Review_Text": review_text,
                            "Timestamp": date_val,
                            "Verified_Badge": verified
                        })
                        total_collected += 1

                    self.logger.info(f"Page {page_num}: Extracted {len(reviews)} reviews.")

                    # Autosave every 50 reviews
                    if total_collected % 50 == 0:
                        self._autosave(category, total_collected)

                    # --- Pagination using ARIA label ---
                    try:
                        next_btn = self.browser.find_element(By.CSS_SELECTOR, "a[aria-label='Next Page']")
                        if not next_btn.is_displayed():
                            self.logger.info("Next button hidden. Pagination complete.")
                            break

                        self.browser.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_btn)
                        self._random_delay(1, 2)
                        self.browser.execute_script("arguments[0].click();", next_btn)

                        # Wait for old articles to become stale
                        try:
                            old_elem = self.browser.find_elements(By.TAG_NAME, "article")
                            self.wait.until(EC.staleness_of(old_elem[0]))
                        except Exception:
                            time.sleep(3)

                        page_num += 1
                        self._random_delay(2, 4)

                    except NoSuchElementException:
                        self.logger.info("No 'Next Page' button found. Pagination complete.")
                        break

                except StaleElementReferenceException:
                    self.logger.warning(f"Stale element on page {page_num}. Retrying loop...")
                    continue
                except Exception as e:
                    self.logger.error(f"Error on page {page_num}: {e}")
                    break

            # Final save
            self._autosave(category, total_collected)

        self._session_guard(_inner)

    def _autosave(self, category, count):
        """Safely saves results to CSV."""
        try:
            os.makedirs("data", exist_ok=True)
            filename = f"data/jumia_reviews_{category.replace(' ', '_')}.csv"
            df = pd.DataFrame(self.results)
            temp_filename = filename + ".tmp"
            df.to_csv(temp_filename, index=False)
            if os.path.exists(filename):
                os.remove(filename)
            os.rename(temp_filename, filename)
            self.logger.info(f"Autosaved {count} reviews.")
        except Exception as e:
            self.logger.error(f"Save failed: {e}")

    def shutdown(self):
        """Gracefully quits the browser."""
        if self.browser:
            self.browser.quit()
        self.logger.info("Scraper Shutdown Complete.")


# ---------------- Main Orchestration ----------------
def run_scraper():
    PATH_TO_DRIVER = r"C:\Users\OLALERE\Desktop\Books\edgedriver_win64\msedgedriver.exe"
    MY_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0"
    )

    TARGET_CATEGORIES = {
        "Mobile Phones": "https://www.jumia.com.ng/mobile-phones/",
        "Computing": "https://www.jumia.com.ng/computing/",
        "Electronics": "https://www.jumia.com.ng/electronics/"
    }

    while True:
        try:
            jumia_bot = JumiaRetailScraper(PATH_TO_DRIVER, MY_USER_AGENT)
            jumia_bot.navigate_home_and_clear_popups()

            for cat_name, cat_url in TARGET_CATEGORIES.items():
                jumia_bot.logger.info(f"--- Scraping category: {cat_name} ---")
                jumia_bot._session_guard(jumia_bot.browser.get, cat_url)
                product_links = jumia_bot.discover_products(cat_name)

                for i, link in enumerate(product_links):
                    jumia_bot.extract_reviews(link, cat_name)

                    # Re-initialize browser every 20 products to avoid timeouts
                    if i % 20 == 0 and i != 0:
                        jumia_bot.logger.info("Refreshing browser session for stability...")
                        jumia_bot._init_browser()

            jumia_bot.shutdown()
            break

        except Exception as e:
            logging.error(f"Fatal error, restarting scraper: {e}")
            traceback.print_exc()
            try:
                jumia_bot.shutdown()
            except Exception:
                pass
            delay = random.randint(60, 120)
            logging.info(f"Restarting after {delay} seconds...")
            time.sleep(delay)


if __name__ == "__main__":
    run_scraper()
