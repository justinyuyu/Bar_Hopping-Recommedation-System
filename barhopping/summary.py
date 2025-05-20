import json
from barhopping.config import CITY, MAX_BARS
from barhopping.scraper.maps import get_bars, get_addr_reviews, get_photos
from barhopping.summarizer.gemma import summarize_bar
from barhopping.embedding.granite import get_embedding
from barhopping.database.sqlite import init_bars, insert_bar
from barhopping.logger import logger

def dataPreparation():
    init_bars()
    bars = get_bars(CITY, MAX_BARS)
    logger.info(f"Retrieved {len(bars)} bars for {CITY}")

    for b in bars:
        try:
            addr, revs = get_addr_reviews(b["url"])
            photos = get_photos(b["url"])
            summary = summarize_bar(revs, photos)
            emb_list = get_embedding(summary).squeeze(0).tolist()

            bar = {
                "name": b["name"],
                "url": b["url"],
                "city": CITY,
                "address": addr,
                "rating": b["rating"],
                "photo": photos[0] if photos else "",
                "summary": summary,
                "embedding": json.dumps(emb_list),
            }

            insert_bar(bar)
            logger.info(f"Inserted {b['name']}")
        except Exception as e:
            logger.error(f"Failed {b['name']}: {e}")

if __name__ == "__main__":
    dataPreparation()