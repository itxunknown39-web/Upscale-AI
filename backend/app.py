import os
import sys

# Ensure torchvision functional_tensor backward compatibility for basicsr
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except Exception:
    pass

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(title="Adobe Stock AI Upscaler API", version="1.0.0")

# Setup CORS for development and Colab environment access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include API routers
from backend.routes import router
app.include_router(router)

# Mount the frontend directory as a static resource folder under /static
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve the HTML dashboard at the root URL /
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = "frontend/index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return HTMLResponse(
            content="<h3>Error: frontend/index.html not found! Please check repository files.</h3>",
            status_code=404
        )

if __name__ == "__main__":
    import uvicorn
    # Start web server on port 8000
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
