import cv2
import numpy as np
import chromadb
from transformers import CLIPProcessor , CLIPModel
from PIL import Image
import os
import torch

"""Database configuration
initializes local persistent storage for vectors"""

client = chromadb.PersistentClient(path="./chroma_db_storage")
#create or import video
video=client.get_or_create_collection(
   name="movie_frame",
   metadata ={"hnsw:space":"cosine"}
)
#AI MODEL INITIALIZATION
print("[INFO] Loading OpenAI CLIP model from Hugging face... ")
ai_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
ai_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("[success] ai engine is ready")
#core processing functions
def process_and_index_video(video_path , sampling_rate_fps : int =1) -> bool:
   """extracts video frame, transforms pixels to 512 -D cLIP 
   embeddings and indexes vectors into ChromaDB with timestamp metadata."""
   cap= cv2.VideoCapture(video_path)
   if not cap.isOpened():
      print("[ERROR] Cannot open file:" , video_path)
      return False
   source_fps= cap.get(cv2.CAP_PROP_FPS)
   sample_interval= max(1, int ( source_fps / sampling_rate_fps))
   frame_count =0
   indexed_count=0
   """The long way (2 steps)
   result = cap.read()
   success = result[0]   /True or False
   frame = result[1]     /The image pixel matrix"""
   while( True):
      success ,frame =cap.read()
      if not success:
         break
      if frame_count %  sample_interval==0: #so that it only gives multiples of sample_frame like interval ex - 0, 30 , 60
         timestamp_sec=round( frame_count / source_fps , 2) #correct to 2 decimal place
         rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB) #computer vision step convert OpenCV default from BGR TO RGB format
         pil_image=Image.fromarray(rgb)
      #neutral vectorization step
         inputs =ai_processor(images=pil_image , return_tensors ="pt" , padding=True)
         with torch.no_grad():
      #extract image features
            image_features = ai_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
      #flatten the matrix into 1D and convert into python list
            vector = image_features.cpu().numpy().flatten().tolist()
      #Persistent storage step
      #store a vector coordinate , a unique string ID, and metadata properties into chromdb
         video.add(
            embeddings=[vector],
            ids=["frame_" + str(timestamp_sec)],
            metadatas= [{"timestamp" : timestamp_sec , "source" : os.path.basename(video_path)}])
         print("timestamp" , timestamp_sec , "|" , "vector dimensions :" , len(vector))
         indexed_count+=1
      frame_count+=1
   cap.release()
   print("[SUCCESS] Complete Database indexed "  ,indexed_count  , "video  frmae vectors")
   return True

def search_video_by_text(query_text , num_result=3):
   """converts a human prompt into an AI vector and queries 
   the database using mathematical  cosine similarity to find 
  the similiar frame position """
   inputs = ai_processor(text=[query_text] , return_tensors = "pt" , padding =True)
   with torch.no_grad():
      text_features=ai_model.get_text_features(**inputs)
      text_features = text_features / text_features.norm(dim=-1, keepdim=True)
      query_vector = text_features.cpu().numpy().flatten().tolist()
      results= video.query( query_embeddings=[query_vector] , n_results= num_result)
      return results

if __name__ == "__main__":
#local terminal testing profile
   video_file = "sample.mp4"
   if os.path.exists(video_file):
      process_and_index_video(video_file, sampling_rate_fps=1)
      print("Running a baseline vector quantity test " )
      print(search_video_by_text("a person walking ", num_result=2))
   else:
      print("[WARNING] Put a video file named " , video_file ," here to seed data.")
