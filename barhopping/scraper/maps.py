import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
from barhopping.config import MAX_BARS, MAX_PHOTOS, MAX_REVS
from barhopping.logger import logger

# Single browser instance
def _init_browser() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

browser = _init_browser()

def get_bars(city: str, nums: int = MAX_BARS) -> list[dict]:
    url = f"https://www.google.com/maps/search/bars+in+{city}"
    browser.get(url)
    
    elems = browser.find_elements(By.CLASS_NAME, "hfpxzc")
    while len(elems) < nums:
        prev = len(elems)
        ActionChains(browser).scroll_from_origin(
            ScrollOrigin.from_element(elems[-1]), 0, 1000
        ).perform()
        time.sleep(2)
        elems = browser.find_elements(By.CLASS_NAME, "hfpxzc")
        if len(elems) <= prev:
            break

    soup = BeautifulSoup(browser.page_source, "lxml")
    bar_links = soup.find_all("a", class_="hfpxzc")
    ratings = soup.find_all("span", class_="MW4etd")

    bars = []
    for link, rating in zip(bar_links, ratings):
        bars.append({
            "name": link["aria-label"],
            "rating": rating.text,
            "url": link["href"]
        })

    return bars


def get_addr_reviews(url: str, min_char: int = MAX_REVS) -> tuple[str, list[str]]:
    browser.get(url)

    try:
        address = browser.find_element(By.CLASS_NAME, "Io6YTe").text
    except:
        address = "Address not found"
        
    # Open reviews
    btns = browser.find_elements(By.CLASS_NAME, "hh2c6")
    if len(btns) > 1:
        btns[1].click()
        time.sleep(2)

    reviews, char_count = [], 0
    elems = browser.find_elements(By.CLASS_NAME, "MyEned")

    while char_count < min_char:
        prev_len = len(elems)
        ActionChains(browser).scroll_from_origin(
            ScrollOrigin.from_element(elems[-1]), 0, 1000
        ).perform()
        time.sleep(2)
        elems = browser.find_elements(By.CLASS_NAME, "MyEned")

        for more_btn in browser.find_elements(By.CLASS_NAME, "w8nwRe"):
            try:
                more_btn.click()
            except Exception:
                continue

        for e in elems[len(reviews):]:
            txt = re.sub(r"\s+", " ", e.text)
            reviews.append(txt)
            char_count += len(txt)
            if char_count >= min_char:
                break

        if len(elems) == prev_len:
            break

    return address, reviews


def get_photos(url: str, nums: int = MAX_PHOTOS) -> list[str]:
    browser.get(url)

    try:
        browser.find_element(By.CLASS_NAME, "Dx2nRe").click()
        time.sleep(1)

        # Click on "Vibe" tab if present
        for btn in browser.find_elements(By.CLASS_NAME, "hh2c6"):
            if btn.text == "Vibe":
                btn.click()
                time.sleep(1)
                break

        photos = browser.find_elements(By.CLASS_NAME, "Uf0tqf")
        photo_urls = []

        for photo in photos[:nums]:
            style = photo.get_attribute("style")
            start = style.find("http")
            end = style.rfind(")") if ")" in style else len(style)
            photo_urls.append(style[start:end].strip("\"')"))

        return photo_urls

    except Exception as e:
        logger.error(f"Error getting photos: {e}")
        return []