import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from moviepy.editor import VideoFileClip
from PIL import Image, ImageFont, ImageDraw
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from models.blip import blip_decoder
from moviepy.editor import VideoFileClip
import pandas as pd
from googletrans import Translator
os.getcwd()


def get_frames(video_dir, sec_per):
    vidcap = cv2.VideoCapture(video_dir)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = VideoFileClip(video_dir).duration
    frames_per_second = int(total_frames/duration)
    denominator = frames_per_second*sec_per
    # iterator
    _,image = vidcap.read()
    
    image_numpy_list = list()
    frames_list = list()
    for frame in tqdm(range(total_frames)):
        if frame % denominator == 0:
            image_numpy_list.append(image)  
            frames_list.append(frame)   # save frame as JPEG file
        _, image = vidcap.read()
    PILs = [Image.fromarray(np.uint8(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))).convert('RGB') for i in image_numpy_list[:-1]]
    return PILs, frames_list[:-1], total_frames







class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_size, image_PIL_list):
        self.image_size = image_size
        self.image_PIL_list = image_PIL_list
        self.transformation = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 

    def __len__(self):
        return len(self.image_PIL_list)

    def __getitem__(self, idx):
        return self.transformation(self.image_PIL_list[idx])

def loader(image_size, image_PIL_list, batch_size, num_workers):
    image_loader = torch.utils.data.DataLoader(
        Dataset(image_size, image_PIL_list), batch_size = batch_size, shuffle = False,
        num_workers = num_workers, pin_memory = True
    )
    return image_loader
    



def get_text_batch(image_batch, device):
    image_size = image_batch.shape[2]
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    caption_batch = model.generate(image_batch, sample=False, num_beams=3, max_length=20, min_length=5) 
    return caption_batch



def get_text_and_image(video_dir, sec_per, image_size, batch_size, num_workers, device):
    PIL_list, frames_list, total_frames = get_frames(video_dir, sec_per)
    image_loader = loader(image_size = image_size, image_PIL_list = PIL_list, batch_size = batch_size, num_workers = num_workers)
    result_list = list()
    with torch.no_grad():
        for frames_batch in tqdm(image_loader):
            image_batch = frames_batch.to(device)
            generated_text = get_text_batch(image_batch, device)
            result_list += generated_text
    result_dict = dict()
    for frame, text in zip(frames_list, result_list):
        result_dict[frame] = text
    return result_dict, total_frames


result_dict, total_frames = get_text_and_image( video_dir = '/home/mskang/Vlog/hj/image_captioning/BLIP/y2.mp4',
                                                sec_per = 1, 
                                                image_size = 300, 
                                                batch_size = 256, 
                                                num_workers = 20, 
                                                device = 'cuda:1')



def make_pandas(result_dict, total_frames):
    translator = Translator()
    sentence_per_frame = list()
    sentence_per_frame_ko = list()
    sentence = 'dummy'
    for frame in range(total_frames):
        if frame in result_dict.keys():
            sentence = result_dict[frame]
            korean = translator.translate(sentence, dest="ko").text
        sentence_per_frame.append(sentence)
        sentence_per_frame_ko.append(korean)
    dataframe = pd.DataFrame({'frame': [i for i in range(total_frames)], 'sentence': sentence_per_frame, 'korean' : sentence_per_frame_ko})
    return dataframe





def put_text(image_numpy, text, location):
    image_PIL = Image.fromarray(image_numpy)
    draw = ImageDraw.Draw(image_PIL)
    draw.text(xy = location, text = str(text), font=ImageFont.truetype("./batang.ttc", 20), fill=(255,255,255))
    return np.array(image_PIL)
    

def make_video(video_dir, out_dir, result_dict, total_frames):
    dataframe = make_pandas(result_dict, total_frames).iterrows()
    def pipeline(frame):
        try:
            frame = put_text(image_numpy = frame, text = str(next(dataframe)[1].sentence), location = (80, 600))
            frame = put_text(image_numpy = frame, text = str(next(dataframe)[1].korean), location = (80, 700))
            #cv2.putText(frame, str(next(dataframe)[1].sentence), (80, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_8, False)

        except StopIteration:
            pass
        # additional frame manipulation
        return frame

    video = VideoFileClip(video_dir)
    out_video = video.fl_image(pipeline)
    out_video.write_videofile(out_dir, audio=True)


make_video( video_dir = '/home/mskang/Vlog/hj/image_captioning/BLIP/y2.mp4',
            out_dir = '/home/mskang/Vlog/hj/image_captioning/BLIP/y2_text.mp4',
            result_dict = result_dict, 
            total_frames = total_frames)








