# prod_assistant/etl/data_scrapper.py

"""
================================================================================
 PulseFlow ETL – Flipkart Product Review Scraper
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : etl.data_scrapper
- Version     : 1.0.0
- Created on  : 2025-10-10
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | Selenium | BeautifulSoup | Undetected-Chrome
================================================================================

This module defines the **FlipkartScraper** class responsible for extracting
product metadata and top customer reviews from Flipkart. It forms the *Extract*
phase of the ETL pipeline that feeds PulseFlow’s vectorized review intelligence
system (AstraDB).

Core Responsibilities
---------------------
- Launch undetected Chrome browser instance for stealth scraping.
- Search Flipkart for target product queries.
- Extract product details (title, price, rating, total reviews).
- Retrieve top user reviews for each product detail page.
- Persist all extracted data into structured CSV format for ingestion.

Workflow Topology
-----------------
    Flipkart → Selenium → HTML → BeautifulSoup → CSV → DataIngestion (AstraDB)

External Integrations
---------------------
- **undetected_chromedriver** — Stealth browser automation for Flipkart scraping.
- **BeautifulSoup** — HTML parsing and review extraction.
- **pandas / AstraDBVectorStore** — Downstream ingestion in next pipeline stage.
- **core.globals.LOGGER** — Structured logging for traceable ETL operations.

Changelog
---------
v1.0.0 (2025-10-10)
    • Added stealth scraping using undetected_chromedriver.
    • Implemented robust CSV writer and top-review extraction.
    • Integrated structured logging via global PulseFlow logger.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
"""

import csv
import time
import re
import os
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from prod_assistant.core.globals import LOGGER


class FlipkartScraper:
    """
    Handles scraping of product information and top reviews from Flipkart.

    Steps:
        1. Launches a stealth Chrome instance.
        2. Searches Flipkart with user query.
        3. Extracts product metadata.
        4. Retrieves top N user reviews per product.
        5. Exports results to structured CSV.
    """

    def __init__(self, output_dir: str = "data"):
        """Initialize output directory and prepare for scraping."""
        try:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
            LOGGER.info("FlipkartScraper initialized", output_dir=self.output_dir)
        except Exception as e:
            LOGGER.error("Failed to initialize FlipkartScraper", error=str(e))
            raise

    # ------------------------------------------------------------------
    def get_top_reviews(self, product_url: str, count: int = 2) -> str:
        """
        Retrieve the top N reviews for a given product.

        Parameters
        ----------
        product_url : str
            URL of the Flipkart product page.
        count : int, optional
            Number of reviews to extract (default = 2).

        Returns
        -------
        str
            Concatenated review text separated by `||` or
            `"No reviews found"` if none were retrieved.
        """
        if not product_url.startswith("http"):
            return "No reviews found"

        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")

        try:
            driver = uc.Chrome(options=options, use_subprocess=True)
            driver.get(product_url)
            time.sleep(4)

            # Close popup if exists
            try:
                driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
                time.sleep(1)
            except Exception:
                pass

            # Scroll to load more reviews
            for _ in range(4):
                ActionChains(driver).send_keys(Keys.END).perform()
                time.sleep(1.5)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            review_blocks = soup.select("div._27M-vq, div.col.EPCmJX, div._6K-7Co")

            seen, reviews = set(), []
            for block in review_blocks:
                text = block.get_text(separator=" ", strip=True)
                if text and text not in seen:
                    reviews.append(text)
                    seen.add(text)
                if len(reviews) >= count:
                    break

            driver.quit()
            result = " || ".join(reviews) if reviews else "No reviews found"
            LOGGER.info("Top reviews extracted", count=len(reviews))
            return result

        except Exception as e:
            LOGGER.error("Failed to extract top reviews", url=product_url, error=str(e))
            return "No reviews found"

    # ------------------------------------------------------------------
    def scrape_flipkart_products(self, query: str, max_products: int = 1, review_count: int = 2):
        """
        Search Flipkart for products and extract details along with top reviews.

        Parameters
        ----------
        query : str
            Search keyword for Flipkart (e.g., "iPhone 15 Plus").
        max_products : int, optional
            Number of product listings to scrape (default = 1).
        review_count : int, optional
            Number of top reviews to extract for each product.

        Returns
        -------
        list
            List of products containing metadata and review text.
        """
        options = uc.ChromeOptions()
        driver = uc.Chrome(options=options, use_subprocess=True)

        try:
            search_url = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
            driver.get(search_url)
            time.sleep(4)

            # Close popup if exists
            try:
                driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
            except Exception:
                pass

            time.sleep(2)
            products = []

            items = driver.find_elements(By.CSS_SELECTOR, "div[data-id]")[:max_products]
            for item in items:
                try:
                    title = item.find_element(By.CSS_SELECTOR, "div.KzDlHZ").text.strip()
                    price = item.find_element(By.CSS_SELECTOR, "div.Nx9bqj").text.strip()
                    rating = item.find_element(By.CSS_SELECTOR, "div.XQDdHH").text.strip()
                    reviews_text = item.find_element(By.CSS_SELECTOR, "span.Wphh3N").text.strip()
                    match = re.search(r"\d+(,\d+)?(?=\s+Reviews)", reviews_text)
                    total_reviews = match.group(0) if match else "N/A"

                    link_el = item.find_element(By.CSS_SELECTOR, "a[href*='/p/']")
                    href = link_el.get_attribute("href")
                    product_link = href if href.startswith("http") else "https://www.flipkart.com" + href
                    match = re.findall(r"/p/(itm[0-9A-Za-z]+)", href)
                    product_id = match[0] if match else "N/A"
                except Exception as e:
                    LOGGER.warning("Failed to parse product item", error=str(e))
                    continue

                top_reviews = (
                    self.get_top_reviews(product_link, count=review_count)
                    if "flipkart.com" in product_link
                    else "Invalid product URL"
                )

                products.append([product_id, title, rating, total_reviews, price, top_reviews])

            LOGGER.info("Product scraping completed", total=len(products))
            return products

        except Exception as e:
            LOGGER.error("Flipkart scraping failed", query=query, error=str(e))
            return []

        finally:
            driver.quit()

    # ------------------------------------------------------------------
    def save_to_csv(self, data, filename: str = "product_reviews.csv"):
        """
        Save scraped product data to a structured CSV file.

        Parameters
        ----------
        data : list
            List of product rows `[product_id, title, rating, reviews, price, top_reviews]`.
        filename : str, optional
            Output CSV file name (default = "product_reviews.csv").
        """
        try:
            if os.path.isabs(filename):
                path = filename
            elif os.path.dirname(filename):
                path = filename
                os.makedirs(os.path.dirname(path), exist_ok=True)
            else:
                path = os.path.join(self.output_dir, filename)

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"])
                writer.writerows(data)

            LOGGER.info("Scraped data saved to CSV", path=path, records=len(data))

        except Exception as e:
            LOGGER.error("Failed to save scraped data to CSV", error=str(e))
            raise
