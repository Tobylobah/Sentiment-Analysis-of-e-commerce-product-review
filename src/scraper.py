import os
import re
import time
import random
import logging
import hashlib
import traceback

import pandas as pd
import dotenv

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options

from selenium.webdriver.common.by import By

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.common.exceptions import(
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
    InvalidSessionIdException,
    WebDriverException,
    StaleElementReferenceException
)

dotenv.load_dotenv()


class JumiaRetailScraper:

    """
    Production review scraper.

    Improvements:
    • Stable rating extraction
    • Deterministic User_ID generation
    • Product name capture
    • Clean schema for ML pipeline
    • Crash recovery
    """

    def __init__(self, driver_path, user_agent):

        self.driver_path = driver_path
        self.user_agent = user_agent

        self.results = []

        self.browser = None
        self.wait = None

        os.makedirs("logs",exist_ok=True)

        logging.basicConfig(
            filename="logs/jumia_scraper.log",
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s"
        )

        self.logger=logging.getLogger()

        self._init_browser()


    def _init_browser(self):

        try:

            if self.browser:

                try:
                    self.browser.quit()
                except:
                    pass

            edge_options=Options()

            edge_options.add_argument(f"user-agent={self.user_agent}")

            edge_options.add_argument("--disable-blink-features=AutomationControlled")

            edge_options.add_experimental_option(
                'excludeSwitches',
                ['enable-logging']
            )

            edge_options.add_argument("--headless=new")

            edge_options.add_argument("--disable-gpu")

            edge_options.add_argument("--window-size=1920,1080")

            service=Service(
                executable_path=self.driver_path,
                log_path=os.devnull
            )

            self.browser=webdriver.Edge(
                service=service,
                options=edge_options
            )

            self.browser.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source":
                    "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
                }
            )

            self.wait=WebDriverWait(self.browser,20)

            self.logger.info("Browser ready")

        except Exception as e:

            self.logger.critical(e)
            raise


    def _delay(self,a=2,b=5):

        time.sleep(random.uniform(a,b))


    def _user_id(self,user,product):

        base=f"{user}|{product}"

        return hashlib.md5(
            base.encode()
        ).hexdigest()[:12]


    def navigate_home(self):

        self.browser.get(
            "https://www.jumia.com.ng"
        )

        self._delay(3,5)

        try:

            btn=self.wait.until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//button[contains(@aria-label,'close')]"
                    )
                )
            )

            btn.click()

        except:
            pass


    def discover_products(self):

        links=self.browser.execute_script("""

        var arr=[];

        document.querySelectorAll(
        'article.prd a.core'
        ).forEach(x=>arr.push(x.href));

        return arr;

        """)

        return links


    def extract_reviews(self,url,category):

        self.logger.info(url)

        self.browser.get(url)

        self._delay(2,4)

        product_name=self.browser.title

        try:

            see_all=self.wait.until(
                EC.element_to_be_clickable(
                    (By.PARTIAL_LINK_TEXT,"See All")
                )
            )

            self.browser.execute_script(
                "arguments[0].click();",
                see_all
            )

        except:

            return

        page=1

        while True:

            try:

                self.wait.until(
                    EC.presence_of_all_elements_located(
                        (By.TAG_NAME,"article")
                    )
                )

                soup=BeautifulSoup(
                    self.browser.page_source,
                    "html.parser"
                )

                reviews=soup.find_all("article")

                if not reviews:

                    break


                for r in reviews:

                    rating=0

                    stars=r.find(
                        "div",
                        class_="stars"
                    )

                    if stars:

                        m=re.search(
                            r'(\d+)',
                            stars.get_text()
                        )

                        if m:

                            rating=int(m.group(1))


                    title=""

                    if r.find("h3"):

                        title=r.find("h3").get_text(strip=True)


                    body=""

                    if r.find("p",class_="-pvs"):

                        body=r.find(
                            "p",
                            class_="-pvs"
                        ).get_text(strip=True)


                    user="Anonymous"

                    date="N/A"

                    meta=r.find(
                        "div",
                        class_="-pvs"
                    )

                    if meta:

                        spans=meta.find_all("span")

                        if len(spans)>=2:

                            date=spans[0].get_text(strip=True)

                            user=spans[1].get_text(
                                strip=True
                            ).replace("by ","")


                    verified=False

                    if "Verified Purchase" in r.get_text():

                        verified=True


                    user_id=self._user_id(
                        user,
                        url
                    )


                    self.results.append({

                        "Category":category,

                        "Product_Name":product_name,

                        "Product_URL":url,

                        "User_Name":user,

                        "User_ID":user_id,

                        "Rating":rating,

                        "Review_Title":title,

                        "Review_Text":body,

                        "Timestamp":date,

                        "Verified_Badge":verified

                    })


                try:

                    nxt=self.browser.find_element(

                        By.CSS_SELECTOR,
                        "a[aria-label='Next Page']"

                    )

                    self.browser.execute_script(
                        "arguments[0].click();",
                        nxt
                    )

                    page+=1

                    self._delay(2,4)

                except:

                    break


            except StaleElementReferenceException:

                continue


    def save(self,category):

        os.makedirs("data/raw",exist_ok=True)

        file=f"data/raw/jumia_reviews_{category}.csv"

        pd.DataFrame(
            self.results
        ).to_csv(
            file,
            index=False
        )

        self.logger.info(
            f"Saved {len(self.results)}"
        )


    def shutdown(self):

        if self.browser:

            self.browser.quit()



def run():

    DRIVER=os.getenv("DRIVER")

    AGENT=os.getenv("AGENT")

    TARGET={

        "mobile":

        "https://www.jumia.com.ng/mobile-phones/",

        "computing":

        "https://www.jumia.com.ng/computing/",

        "electronics":

        "https://www.jumia.com.ng/electronics/"

    }

    bot=JumiaRetailScraper(
        DRIVER,
        AGENT
    )

    bot.navigate_home()

    for cat,url in TARGET.items():

        bot.browser.get(url)

        links=bot.discover_products()

        for link in links:

            bot.extract_reviews(
                link,
                cat
            )

        bot.save(cat)

    bot.shutdown()


if __name__=="__main__":

    run()