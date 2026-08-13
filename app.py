import os
from fastapi import FastAPI , Form 
from fastapi.responses import HTMLResponse , FileResponse
import uvicorn
from engine import search_video_by_text
app=FastAPI(title="AI vector engine web interface")
video_path= "sample.mp4"
@app.get("/videos/{video_path}")
async def get_video(video_path : str):
  
    """serves the target video directly to HTML5 video player"""
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    return HTMLResponse("VIDEO FILE NOT FOUND" , status_code=404)

@app.get("/" , response_class =HTMLResponse)
async def get_homepage():
    """builds and serves the main interace layout"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Semantic Search Hub</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background-color: #f3f2f1; padding: 40px; text-align: center; }
            .card { max-width: 650px; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin: 0 auto; }
            h2 { color: #0078d4; margin-top: 0; }
            input[type="text"] { width: 100%; padding: 12px; font-size: 16px; border: 1px solid #bab8b7; border-radius: 4px; box-sizing: border-box; }
            button { background-color: #0078d4; color: white; padding: 12px; font-size: 16px; border: none; border-radius: 4px; cursor: pointer; width: 100%; margin-top: 15px; font-weight: bold; }
            button:hover { background-color: #106ebe; }
        </style>
    </head>
    <body>
        <div class="card">
        <h2>AI Video Search Engine</h2>
        <p>Search for specific actions, objects, or scenes inside the indexed video:</p>
            <form action="/search" method="post">
                <input type="text" name="query" placeholder="Example: 'red car driving' or 'man smiling'" required>
                <button type="submit">Search Video Database</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/search" ,response_class=HTMLResponse)
async def handle_search(query: str= Form(...)):
    """execute vector query and rendors video playback with timestamp controls"""
    db_hits=search_video_by_text(query , num_result=3)
    results = "" #empty string
    if db_hits and 'ids' in db_hits and len(db_hits['ids'])>0: #Prevents IndexError crashes if the database search returns empty or keyless results.
        for i in range(len(db_hits['ids'][0])):
            metadata= db_hits['metadatas'][0][i]
            distance= db_hits['distances'][0][i]
            #extract metadata values (fallback to defaults if missing)
            timestamp =metadata.get('timestamp' ,0) #default value 0 if timestamp is missing
            video_filename = metadata.get('video_path' , 'sample.mp4')
            #formatted time display 'MM:SS'
            minutes , seconds= divmod(int(timestamp) , 60) #do time (t/60,t%60)
            formatted_time= f"{minutes :02d} :{seconds:02d}" #its a string and it changes time into 2 digits , if answer is inone it adds one 0
            results += f"""
                <div style="background:#faf9f8; padding:20px; margin-top:20px; border-left: 5px solid #0078d4; text-align:left; border-radius:0 8px 8px 0;">
                    <div style="margin-bottom:10px;">
                        <strong>Match Timestamp:</strong> {formatted_time} ({timestamp}s)<br>
                        <small style="color:#605e5c;">Vector Distance Score: {round(distance, 4)}</small>
                    </div>
                    
                    <!-- HTML5 Video element auto-seeking to timestamp using '#t=seconds' -->
                    <video controls width="100%" style="border-radius:4px; max-height:300px; background:#000;">
                        <source src="/videos/{video_filename}#t={timestamp}" type="video/mp4">
                        Your browser does not support HTML5 video streaming.
                    </video>
                </div>
            """
            
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f3f2f1; padding: 50px; }}
            .card {{ max-width: 650px; background: white; padding: 40px; border-radius: 8px; margin: 0 auto; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Search Results for: "{query}"</h2>
            {results if results else "<p>No matches located inside database indexes.</p>"}
            <br><br>
            <a href="/" style="color:#0078d4; text-decoration:none; font-weight:bold;">&larr; Search Again</a>
        </div>
    </body>
    </html>
    """
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
