import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from models.blip import blip_decoder
import pandas as pd
from googletrans import Translator


from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from tokenizers import Tokenizer
from typing import Dict, List, Optional
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import pipeline
from IPython.display import display
from typing import Dict

from transformers import logging
logging.set_verbosity_error()



style_map = {
    'formal': '문어체',
    'informal': '구어체',
    'android': '안드로이드',
    'azae': '아재',
    'chat': '채팅',
    'choding': '초등학생',
    'emoticon': '이모티콘',
    'enfp': 'enfp',
    'gentle': '신사',
    'halbae': '할아버지',
    'halmae': '할머니',
    'joongding': '중학생',
    'king': '왕',
    'naruto': '나루토',
    'seonbi': '선비',
    'sosim': '소심한',
    'translator': '번역기'
}

model_name = "gogamza/kobart-base-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = "/home/mskang/Vlog/hj/image_captioning/text-transfer-smilegate-bart"
nlg_pipeline = pipeline('text2text-generation',model=model_path, tokenizer=model_name, device = 0)














class Dataset(torch.utils.data.Dataset):
    '''
    batch loader for images captioning model
    Normalize as ImageNet
    outout : image loader for decoder
    '''
    def __init__(self, image_size, image_PIL_list, frame_size):
        self.image_size = image_size
        self.image_PIL_list = image_PIL_list
        self.transformation = transforms.Compose([
        transforms.CenterCrop(max(frame_size)), 
        transforms.Resize((image_size, image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 

    def __len__(self):
        return len(self.image_PIL_list)

    def __getitem__(self, idx):
        return self.transformation(self.image_PIL_list[idx])

def set_loader(image_size, image_PIL_list, batch_size, num_workers, frame_size):
    print('load data loader .....        ')
    image_loader = torch.utils.data.DataLoader(
        Dataset(image_size, image_PIL_list, frame_size), batch_size = batch_size, shuffle = False,
        num_workers = num_workers, pin_memory = True
    )
    print('done')
    return image_loader
    




def get_frames(video_dir = './videos/ny/', per_sec = 1):
    '''
    input : directory of vidoes 
    directory should including at least one video with .mp4
    
    if per_sce = 1 then outputs are 1 frame for 1 second
    if per_sec = 0.5 then outputs are 2 frames for 1 second
    it't same with 1/fps
    '''
    video_list = os.listdir(video_dir)
    concat = concatenate_videoclips([VideoFileClip(video_dir + '/' + i) for i in video_list])
    concat.write_videofile("temp.mp4") 
    vidcap = cv2.VideoCapture('temp.mp4')
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'total_frames : {total_frames}')
    duration = VideoFileClip("temp.mp4").duration
    print(f'duration : {duration} seconds')
    frames_per_second = int(total_frames/duration)
    print(f'frames_per_second : {frames_per_second} fps')
    frequency = frames_per_second*per_sec
    print(f'sampling frequency : one image per {frequency} frames')
    print('.'*30)
    # iterator
    _,image = vidcap.read()
    
    image_numpy_list = list()
    frames_list = list()
    for frame in tqdm(range(total_frames)):
        if frame % frequency == 0:
            image_numpy_list.append(image)  
            frames_list.append(frame)   # save frame as JPEG file
        _, image = vidcap.read()
    PILs = [Image.fromarray(np.uint8(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))).convert('RGB') for i in image_numpy_list[:-1]]
    print('done')
    return PILs, frames_list[:-1], total_frames, PILs[0].size





def get_text_batch(image_batch, device):
    '''
    go through the model 
    input : image
    output : caption (sentence)
    one can change captioning model here !
    we used blip model
    '''
    image_size = image_batch.shape[2]
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    caption_batch = model.generate(image_batch, sample=False, num_beams=3, max_length=20, min_length=5) 
    return caption_batch






def get_text_and_image(video_dir, per_sec, image_size, batch_size, num_workers, device):
    '''
    unification of get_frames, set_loader, get_text_batch
    Thus,
    input : video directory
    output : captions
    : dictionary keys : frame index / values : captions
    '''
    PIL_list, frames_list, total_frames, frame_size = get_frames(video_dir, per_sec)
    image_loader = set_loader(image_size = image_size, image_PIL_list = PIL_list, batch_size = batch_size, num_workers = num_workers, frame_size = frame_size)
    result_list = list()
    with torch.no_grad():
        print('extracting text ....      ', end = '')
        for frames_batch in tqdm(image_loader):
            image_batch = frames_batch.to(device)
            generated_text = get_text_batch(image_batch, device)
            result_list += generated_text
    result_dict = dict()
    for frame, text in zip(frames_list, result_list):
        result_dict[frame] = text
    print('done')
    return result_dict, total_frames, frame_size, PIL_list

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as f

from transformers import BertModel, BertTokenizer

def refine(result_dict):
    '''
    before calculating similarities using this function can reduce comuptational costs.
    first it removes all the neighbor-same senetences
    and we found that some generated senetences having less then 4 words are useless,
    so we suggest that elminate all of them
    '''
    ## delete duplicate for cumputation loss
    new_dict_no_du = dict()
    old = None
    for key in list(result_dict.keys()):
        if result_dict[key] != old:
            new_dict_no_du[key] = result_dict[key]
            old = result_dict[key]
    
    ## delete some words
    '''
    for key in list(new_dict_no_du.keys()):
        replace = new_dict_no_du[key].replace('a woman ', '')
        new_dict_no_du[key] = replace
    '''


    for key in list(new_dict_no_du.keys()):
        if ('coronaviruss' or 'coronavirus') in new_dict_no_du[key].split():
            del new_dict_no_du[key]

    
    ## delete short
    for key in list(new_dict_no_du.keys()):
        if len(new_dict_no_du[key].split()) < 4:
            del new_dict_no_du[key]


    from collections import Counter
    for key in list(new_dict_no_du.keys()):
        if sum([(i>3) for i in Counter(new_dict_no_du[key][0].split()).values()]) > 0:
            del new_dict_no_du[key]
    

    return new_dict_no_du


def similarity(result_dict, method ):
    '''
    calculate the similarity
    input : dict[frame index, caption]
    output : dict[frame index, [caption, similarity]]
    for instance if frame is 10 then similarity means sim(caption_9, caption_10)

    and there are two options for get similarity
    refer : https://www.kaggle.com/code/eriknovak/pytorch-bert-sentence-similarity/notebook

    the similarity is calculated as matrix.
    we, in the end, means the matrix.
    This procedure make enable calculate similarity one-many sentences.
    And we found that rather than use only one element which is one-one, this method performed well.

    Because we used it for grouping sentence.
    '''
    bert_version = 'bert-large-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    model = BertModel.from_pretrained(bert_version)
    model.eval()
    model.to('cuda:0')
    sim_dict = dict()
    sim_dict[0] = [result_dict[0], 0.0]

    num_to_see = 2
    for i in range(num_to_see, len(result_dict.keys())):

        
        encodings = tokenizer(
            [result_dict[list(result_dict.keys())[i-j]] for j in range(num_to_see)]
            , # the texts to be tokenized
            padding=True, # pad the texts to the maximum length (so that all outputs have the same length)
            return_tensors='pt' # return the tensors (not lists)
        ).to('cuda:0')
        with torch.no_grad():
            embeddings = model(**encodings)
            embeddings = embeddings['last_hidden_state']
            if method == 'cls':
                class_token = embeddings[:, 0, :]
                normalized = f.normalize(class_token, p = 2, dim = 1)
                similarity = normalized.matmul(normalized.T).mean()
            elif method == 'mean':
                MEANS = embeddings.mean(dim=1)
                normalized = f.normalize(MEANS, p=2, dim=1)
                mean_dist = normalized.matmul(normalized.T)
                similarity = mean_dist.mean()

            sim_dict[list(result_dict.keys())[i]] = [result_dict[list(result_dict.keys())[i]], float(similarity.cpu())]
    
    return sim_dict

def similarity_for_thumbnail(result_dict, method, refer):
    import torch
    import torch.nn.functional as f

    from transformers import BertModel, BertTokenizer
    bert_version = 'bert-large-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    model = BertModel.from_pretrained(bert_version)
    model.eval()
    model.to('cuda:0')
    sim_dict = dict()
    sim_dict[0] = [result_dict[0], 0.0]
    sim_list = list()
    num_to_see = 4
    for i in range(num_to_see, len(result_dict.keys())):

        
        encodings = tokenizer([refer, result_dict[list(result_dict.keys())[i]]],
            padding=True, 
            return_tensors='pt' 
        ).to('cuda:0')
        with torch.no_grad():
            embeddings = model(**encodings)
            embeddings = embeddings['last_hidden_state']
            if method == 'cls':
                class_token = embeddings[:, 0, :]
                normalized = f.normalize(class_token, p = 2, dim = 1)
                similarity = normalized.matmul(normalized.T).mean()
            elif method == 'mean':
                MEANS = embeddings.mean(dim=1)
                normalized = f.normalize(MEANS, p=2, dim=1)
                mean_dist = normalized.matmul(normalized.T)
                similarity = mean_dist.mean()

            sim_list.append(float(similarity.cpu()))
    
    return sim_list

def similarity_for_thumbnail(result_dict, method, refer):

    bert_version = 'bert-large-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    model = BertModel.from_pretrained(bert_version)
    model.eval()
    model.to('cuda:0')
    sim_dict = dict()
    sim_dict[0] = [result_dict[0], 0.0]
    sim_list = list()
    num_to_see = 4
    for i in range(num_to_see, len(result_dict.keys())):

        
        encodings = tokenizer([refer, result_dict[list(result_dict.keys())[i]]],
            padding=True, 
            return_tensors='pt' 
        ).to('cuda:0')
        with torch.no_grad():
            embeddings = model(**encodings)
            embeddings = embeddings['last_hidden_state']
            if method == 'cls':
                class_token = embeddings[:, 0, :]
                normalized = f.normalize(class_token, p = 2, dim = 1)
                similarity = normalized.matmul(normalized.T).mean()
            elif method == 'mean':
                MEANS = embeddings.mean(dim=1)
                normalized = f.normalize(MEANS, p=2, dim=1)
                mean_dist = normalized.matmul(normalized.T)
                similarity = mean_dist.mean()

            sim_list.append(float(similarity.cpu()))
    
    return sim_list




def eliminate_threshold(sim_dict, threshold):
    '''
    After calculating similarities, this function eliminate by threshold of similarity
    '''
    eliminated = dict()
    for key in list(sim_dict.keys()):
        if sim_dict[key][-1] < threshold:
            eliminated[key] = sim_dict[key]
    return eliminated


def eliminate_by_peaks(sim_dict, peaks):
    '''
    After calculating similarities, this function eliminate by threshold of similarity
    '''
    eliminated = dict()
    for key in list(sim_dict.keys()):
        
        if (key in peaks) or key == 0:
            eliminated[key] = sim_dict[key]
    return eliminated

def translate(eliminated):
    '''
    translate eng to other languages
    '''
    translator = Translator()
    sentence = 'dummy'

    for key in list(eliminated.keys()):
        en = eliminated[key][0]
        ko = translator.translate(en, dest="ko").text
        eliminated[key].append(ko)
        
    return eliminated  


import pandas as pd
import numpy as np

import warnings
warnings.simplefilter("ignore")
import konlpy
konlpy.__version__
from konlpy.tag import *
okt = Okt()

#불완전문장 추출 함수
def complete(tagging):
  if tagging[len(tagging)-1][1] == 'Noun':
    return False
  
#받침 확인 함수
def has_last(word):
  return (ord(word[-1])-44032) % 28 != 0

#동사 추출 함수
def change_order(tagging):
  for i in range(len(tagging)):
    if tagging[i][1] == 'Verb':
      a = tagging[i][0]
      del tagging[i]
      tagging.append((a,'Verb'))
      #print(tagging)

#주어에 조사 추가 함수
def make_word(word):
  if has_last(word) == True:
    word += '이다.'
  else :
    word += "가 있다"
  return word

#동사 만들기 함수
def make_verb(word):
  word = word[:-1]
  word+='다.'
  return word

#명사구를 문장으로 변환하는 함수
def make_new_sentence(tagging):
  new_sentence = ''
  new_sentence += make_word(tagging[-2][0])
  for i in range(len(tagging)-2):
    if tagging[i][1] == 'Josa':
      new_sentence+=tagging[i][0]
    else:
      new_sentence+=' '
      new_sentence+=tagging[i][0]
  new_sentence += ' '
  #new_sentence += make_word(tagging[-2][0])
  #new_sentence += ' '
  new_sentence += make_verb(tagging[-1][0])
  #new_sentence = new_sentence[1:]
  return new_sentence


# 1인칭 변환 함수
def clean_sentence(sentence):
  pronoun_list = ['여자가','여성이','남자가','남성이']
  sentence_split = sentence.split()
  drop = 0
  for word in sentence_split:
    if word in pronoun_list:
      drop += 1
      word_pronoun = word
  if drop == 0:
    return sentence
  else:
    cleaned_sentence = sentence.replace(word_pronoun,'')
    return cleaned_sentence
  


def rule_base(sentence):
    tagging = okt.pos(sentence)
    people = ['여자','남자','사람','여성','남성']
    if complete(tagging) == True:
        cleaned_sent = clean_sentence(sentence)
    else:
        if sentence.split(' ')[-1] in people:
            sentence = sentence[:-3]
            cleaned_sent = make_verb(sentence)
        else:
            sentence = make_word(sentence)
            cleaned_sent = clean_sentence(sentence)
    return cleaned_sent



def generate_text(text, target_style, num_return_sequences=1, max_length=60, rule_based = True):
    if rule_based:
        text = rule_base(text)
    target_style_name = style_map[target_style]
    text = f"{target_style_name} 말투로 변환:{text}"
    out = nlg_pipeline(text, num_return_sequences=num_return_sequences, max_length=max_length)
    return [x['generated_text'] for x in out][0]




def style_transfer(eliminated, styles):
    for key in list(eliminated.keys()):
        ko = eliminated[key][-1]
        for style in styles:
            styled = generate_text(ko, style)
            eliminated[key].append(styled)
    return eliminated






'''
make datafram by dict[frame index, [caption, similarity]] for subtitle
and also translation
factor means if similarity si lower then factor the latter captions will be replace with former

'''
def make_dataframe(styled_dict, total_frames):
    styles = ['formal','android','azae','chat','choding' ,'emoticon' ,'enfp' ,'gentle','halbae' ,'halmae','joongding','king','naruto','seonbi','sosim']
    list_dict = dict(en = list(), similarity = list(), ko = list())
    for style in styles:
        list_dict[style] = list()

    for frame in range(total_frames):
        if frame in styled_dict.keys():
            fix_frame = frame
            for idx, style in enumerate(['en', 'similarity', 'ko'] + styles):
                list_dict[style].append(styled_dict[frame][idx])
        else:
            for idx, style in enumerate(['en', 'similarity', 'ko'] + styles):
                list_dict[style].append(styled_dict[fix_frame][idx])
    list_dict['frame'] = [i for i in range(total_frames)]
    
    return pd.DataFrame.from_dict(list_dict)





def put_text_with_box(image_numpy, text, location, font_dir = "./batang.ttc", font_size = 30, color = (255,255,255)):
    # location : w,h
    # image_numpy : HWC
    # location : start point left-upper w,h

    font_box = ImageFont.FreeTypeFont(font_dir, size=font_size)
    box_size = font_box.getsize(text) # w,h
    image_numpy = np.array(image_numpy) # seems ridiculous but since by using asarray write flag can't be changed

    box = image_numpy[location[1] : location[1] + box_size[1], location[0] : location[0] + box_size[0], ...]*0.3
    box.astype('uint8')

    image_numpy[location[1] : location[1] + box_size[1], location[0] : location[0] + box_size[0], ...] = box

    image_PIL = Image.fromarray(image_numpy)
    draw = ImageDraw.Draw(image_PIL)
    draw.text(location, text, font = ImageFont.truetype(font_dir, font_size), fill = color)
    return np.array(image_PIL)
    





def make_video(video_dir, out_dir, dataframe, frame_size, style):
    print('making video ...', end = '')
    def pipeline(frame):
        try:
            texts = next(dataframe)[1]
            frame = put_text_with_box(image_numpy = frame, text = str(texts.en), location = (80, frame_size[1]-150))
            frame = put_text_with_box(image_numpy = frame, text = str(texts.style), location = (80, frame_size[1]-50))
            #cv2.putText(frame, str(next(dataframe)[1].sentence), (80, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_8, False)

        except StopIteration:
            pass
        # additional frame manipulation
        return frame

    video = VideoFileClip(video_dir)
    out_video = video.fl_image(pipeline)
    out_video.write_videofile(out_dir, audio=True)
    print(f'done, see {out_dir}')


from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration


















