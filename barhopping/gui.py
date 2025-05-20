import gradio as gr
import asyncio
from typing import List
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
from barhopping.retriever.vector_search import get_vector_search
from barhopping.path_finder import PathFinder
from barhopping.logger import logger

class BarHoppingGUI:
    def __init__(self):
        self.vector_search = get_vector_search()
        self.path_finder = PathFinder()
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
        
    def _cleanup_browser(self):
        """Clean up browser resources."""
        if self.browser is not None:
            try:
                self.browser.quit()
                self.browser = None
                logger.info("Browser closed successfully")
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")
        
    def _bar_html(self, name: str, url: str, address: str, photo: str, summary: str) -> str:
        """Generate HTML for a bar card."""
        if photo:
            img_html = f'<img src="{photo}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 12px; margin: 10px 0;" />'
        else:
            img_html = '<div style="width: 100%; height: 200px; background-color: #2a2a2a; border-radius: 12px; margin: 10px 0; display: flex; align-items: center; justify-content: center;"><p style="color: #666; font-size: 14px;">No image available</p></div>'

        return f"""
        <div style="padding: 10px; font-family: 'Segoe UI', sans-serif; background-color: #1e1e1e; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.2); margin-bottom: 20px;">
            <p style="font-size: 20px; font-weight: bold; color: white;">{name}
                <a href="{url}" target="_blank" style="margin-left: 8px; font-size: 14px; text-decoration: none; color: #fbbf24">
                    Google Maps »
                </a>
            </p>
            <p style="font-size: 14px; color: #bbb; text-align: left; margin-top: -4px">📍{address}</p>
            {img_html}
            <p style="font-size: 14px; color: #bbb; text-align: left;">{summary}</p>
        </div>
        """

    def _path_html(self, distance: float) -> str:
        return f"""
        <div style="padding: 10px; font-family:'Segoe UI', sans-serif; text-align:center; margin-bottom: 20px;">
            <p style="font-size: 16px; font-weight: bold; color: #fbbf24;">🏃 Run Distance: {int(distance)} m</p>
        </div>
        """
       
    def _map_html(self, url: str) -> str:
        """Render route map preview HTML."""
        return f"""
        <div style="padding: 10px; font-family:'Segoe UI', sans-serif; background:#1e1e1e; 
                    border-radius:16px; box-shadow:0 8px 24px rgba(0,0,0,0.2); margin-bottom:20px;">
            <p style="font-size: 20px; font-weight: bold; color: white;">🗺️ Night Crawl Route
                <a href="{url}" target="_blank" style="margin-left: 8px; font-size: 14px; text-decoration: none; color: #fbbf24;">
                    View Route on Google Maps »
                </a>
            </p>
        </div>
        """
        
    async def _get_route_url(self, addresses: List[str]) -> str:
        """Create a Google Maps walking route for the bar path."""
        try:
            self._init_browser()
            self.browser.get("https://www.google.com/maps/dir/")
            self.browser.maximize_window()

            # Select walking mode
            WebDriverWait(self.browser, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "m6Uuef")))
            for btn in self.browser.find_elements(By.CLASS_NAME, "m6Uuef"):
                if btn.get_attribute("data-tooltip") == "Walking":
                    btn.click()
                    break

            # Add all addresses
            inputs = self.browser.find_elements(By.CLASS_NAME, "tactile-searchbox-input")
            inputs[0].send_keys(addresses[0])
            inputs[1].send_keys(addresses[1])
            inputs[1].send_keys(Keys.ENTER)
            await asyncio.sleep(2)

            for addr in addresses[2:]:
                self.browser.find_elements(By.CLASS_NAME, "fC7rrc")[-1].click()
                await asyncio.sleep(2)
                input_field = self.browser.find_elements(By.CLASS_NAME, "tactile-searchbox-input")[-1]
                input_field.send_keys(addr)
                input_field.send_keys(Keys.ENTER)
                await asyncio.sleep(2)

            return self.browser.current_url

        except WebDriverException as e:
            logger.error(f"Browser error: {e}")
            self._cleanup_browser()
            self._init_browser()
            raise
        except Exception as e:
            logger.error(f"Error generating route: {e}")
            raise
        
    async def bar_recommendation(self, message: str, history):
        """Generate bar recommendations and route from user query."""
        try:
            response = []
            bars = self.vector_search.search(message)
            bar_ids = [bar["id"] for bar in bars]
            bar_addrs = [f"{bar['name']}, {bar['address']}" for bar in bars]

            path, distances = self.path_finder.find_optimal_path(bar_ids, bar_addrs)
            path_addrs = [bar_addrs[i] for i in path]
            route_task = asyncio.create_task(self._get_route_url(path_addrs))

            for i, path_idx in enumerate(path):
                bar = bars[path_idx]
                response.append(self._bar_html(
                    bar["name"], bar["URL"], bar["address"],
                    bar.get("photo", ""), bar.get("summary", "No description available")
                ))
                if i < len(distances):
                    response.append(self._path_html(distances[i]))
                yield response

            route_url = await route_task
            response.append(self._map_html(route_url))
            yield response

        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            yield ["Sorry, an error occurred while processing your request."]
        
    def launch(self) -> None:
        """Launch the Gradio chatbot interface."""
        css = """
            .chatbox { flex: 1; height: 100%; overflow-y: auto !important; padding: 20px; }
            .avatar-container { width: 50px !important; height: 50px !important; border-radius: 50% !important; }
            .gradio-container { height: 100vh; background-color: #0e0e0e !important; }
            .contain { overflow-y: auto !important; max-height: 100vh !important; }
            footer { display: none !important; }
            .message { max-width: 100% !important; }
            .message img { max-width: 100% !important; height: auto !important; }
            .description { color: white !important; }
        """
        try:
            with gr.Blocks(fill_height=True, css=css) as demo:
                gr.ChatInterface(
                    fn=self.bar_recommendation,
                    description="<strong><span style='color:#fbbf24;'>RunTini</span></strong> <span style='color:white;'>Bar Hopping Route Recommender</span>",
                    textbox=gr.Textbox(
                        placeholder="Think aesthetics, music, drinks, and crowd...",
                        submit_btn=True
                    ),
                    chatbot=gr.Chatbot(
                        elem_classes=["chatbox"],
                        placeholder="Let's map out your perfect night — pick a vibe or tell me yours! 🍸✨",
                        bubble_full_width=False,
                        avatar_images=["./images/user_avatar.png", "./images/bot_avatar.png"],
                        show_label=False,
                        type="messages"
                    ),
                    examples=[
                        "Bars with retro arcade vibes and playful, neon-lit interiors",
                        "Cozy bars with dim lighting and jazz music for a relaxed evening",
                        "Trendy rooftop bars with great views and photogenic cocktails",
                        "Speakeasy-style spots with hidden entrances and vintage aesthetics"
                    ],
                    type="messages"
                )
            demo.launch(share=True)
        finally:
            self._cleanup_browser()
