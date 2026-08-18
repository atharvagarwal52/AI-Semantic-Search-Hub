import os
import cv2
from transformers import CLIPModel , CLIPProcessor
import torch
from PIL import Image
import numpy as np
import chromadb

client = chromadb.PersistentClient(path= "./chroma_db_storage")
video = client.get_or_create_collection(name="movie_frame" , metadata={"hnsw:space": "cosine"})
print("[INFO] Loading.....")
ai_model =CLIPModel.from_pretrained("openai/clip-vit-base-patch32" )
ai_processor =CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("[SUCCESS] engine is ready")
def process_and_index_video(video_path , sampling_fps_rate : int =1) -> bool:
    cap =cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR cannot open file" , video_path) 
        return True
    source_fps =cap.get(cv2.CAP_PROP_FPS)
    sample_interval =max(1, int(source_fps /sampling_fps_rate))
    frame_count=0
    indexed_count=0
    while( True):
        success , frame =cap.read()
        if not success:
            break
        if(frame_count % sample_interval==0):
            timestamp_sec= round(frame_count/ source_fps , 2)
            rgb=cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
            pil_image=Image.fromarray(rgb)
            inputs = ai_processor(images="pil_image" , padding =True , return_tensors= "pt")
            with torch.no_grad():
                image_feature= ai_model.get_image_features()
                image_feature= image_feature / image_feature.norm(dim=1 , keepdim = True )
                vector = image_feature.cpu().numpy().flatten().list()
                video.add(
                    embeddings=[vector],
                    id= ["frame_" + str(timestamp_sec)],
                    metadatas=[{ "timeframe" :timestamp_sec , "source" : os.basename(video_path)}])
                print("timeframe" , timestamp_sec , " | vector dimension" , len(vector) )
                indexed_count+=1
            frame_count+=1
        cap.release()
        print ("[SUCCESS] complete database indexed" , indexed_count , "video frame vectors")
        return True
def search_video_by_text(query_text , num_result= 3):
    inputs = ai_processor(query = "query_text" , padding =True, return_tensors = "pt")
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
    
            