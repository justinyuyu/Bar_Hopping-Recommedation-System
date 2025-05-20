import numpy as np
import re
from itertools import combinations
from typing import List, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from barhopping.logger import logger

class PathFinder:
    def __init__(self):
        self.browser = None
        
    def _init_browser(self):
        """Initialize the browser if not already initialized."""
        if self.browser is None:
            try:
                options = webdriver.ChromeOptions()
                options.add_argument("--headless")  # Run in headless mode
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                self.browser = webdriver.Chrome(options=options)
                logger.info("Browser initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize browser: {str(e)}")
                raise
    
    def _close_browser(self):
        """Closes the browser session if open."""
        if self.browser:
            try:
                self.browser.quit()
                logger.info("Browser closed successfully.")
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")
            finally:
                self.browser = None
    
    def _get_distance(self, addr1: str, addr2: str, unit: str="m") -> float:
        """Fetches walking distance between two addresses using Google Maps."""
        try:
            self._init_browser()
            self.browser.get("https://www.google.com/maps/dir/")

            # Click walking mode - wait for the button to be clickable
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "m6Uuef"))
            )
            travel_btn = self.browser.find_elements(By.CLASS_NAME, "m6Uuef")
            for btn in travel_btn:
                if btn.get_attribute("data-tooltip") == "Walking":
                    btn.click()
                    break

            # Add two addresses
            inputs = self.browser.find_elements(By.CLASS_NAME, "tactile-searchbox-input")
            inputs[0].send_keys(addr1)
            inputs[1].send_keys(addr2)
            inputs[1].send_keys(Keys.ENTER)

            # Wait for the distance info to be present
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "ivN21e"))
            )
            dist = self.browser.find_element(By.CLASS_NAME, "ivN21e")

            conversion = {'km':1000, 'm':1, 'mile':1609.344, 'ft':0.3048}
            convert_unit = lambda s: (
                lambda m: float(m.group(1)) * conversion[m.group(2)] / conversion[unit] if m else 0
            )(re.match(r'([\d.]+)\s*(mile|ft|km|m)', s))

            return convert_unit(dist.text)
        
        except Exception as e:
            logger.error(f"Error getting distance between '{addr1}' and '{addr2}': {str(e)}")
            return float("inf")

    def _get_distance_matrix(self, addresses: List[str]) -> np.ndarray:
        """Builds a symmetric distance matrix between all bar addresses."""
        n = len(addresses)
        matrix = np.zeros((n, n))

        for i, j in combinations(range(n), 2):
            dist = self._get_distance(addresses[i], addresses[j])
            matrix[i][j] = matrix[j][i] = dist
            logger.info(f"Distance between {addresses[i]} and {addresses[j]}: {dist} meters")

        return matrix
        
    def _hamiltonian_path(self, dist_matrix: np.ndarray, start: int = 0) -> Tuple[List[int], List[float]]:
        """Computes the shortest Hamiltonian path using dynamic programming."""
        n = len(dist_matrix)
        dp = {(1 << start, start): (0, -1)}

        for subset_size in range(2, n + 1):
            for subset in combinations(range(n), subset_size):
                if start not in subset:
                    continue
                mask = sum(1 << i for i in subset)
                for curr in subset:
                    prev_mask = mask & ~(1 << curr)
                    candidates = [
                        (dp[(prev_mask, k)][0] + dist_matrix[k][curr], k)
                        for k in subset if k != curr and (prev_mask, k) in dp
                    ]
                    if candidates:
                        dp[(mask, curr)] = min(candidates)

        full_mask = (1 << n) - 1
        candidates = [(dp[(full_mask, i)][0], i) for i in range(n) if (full_mask, i) in dp]
        if not candidates:
            logger.error("No valid Hamiltonian path found.")
            return [], []

        _, last = min(candidates)

        # Backtrack to find the optimal path
        path = [last]
        mask = full_mask
        while True:
            _, prev = dp.get((mask, last), (None, None))
            if prev == -1:
                break
            path.append(prev)
            mask &= ~(1 << last)
            last = prev
        path.reverse()

        distances = [dist_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)]
        return path, distances
        
    def find_optimal_path(self, bar_ids: List[int], addresses: List[str]) -> Tuple[List[int], List[float]]:
        """Finds the optimal order to visit bars based on walking distance."""
        try:
            dist_matrix = self._get_distance_matrix(addresses)
            path, distances = self._hamiltonian_path(dist_matrix)
            return path, distances
        finally:
            self._close_browser()