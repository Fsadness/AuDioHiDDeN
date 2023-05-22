import numpy as np
import librosa
import librosa.display 
import datetime
import matplotlib.pyplot as plt
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torchvision import datasets, transforms
from IPython.display import clear_output
import torchvision
#from torchvision.datasets.vision import VisionDataset
from torch.optim import Adam
from tqdm import tqdm
import torch
import os.path
import os
import gc
import sys
from PIL import ImageFile, Image
from torchaudio import transforms as audiotransforms
import torchaudio
import soundfile 
from IPython.display import Audio
import random
import skimage
from skimage import transform
from collections import Counter
import soundfile

from plot import *
from SSIM import *
from critic import *
from decoder import *
from encoder import *
from loader import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


from torch.utils.tensorboard import SummaryWriter
# %load_ext tensorboard 

# to watch previous logs please change the path to the folder logs are stored into

current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('results/runs', current_time )
os.makedirs(log_dir, exist_ok=True)
# %tensorboard --logdir {log_dir}


epochs = 32
data_depth = 4
hidden_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOAD_MODEL = True
# PATH='audio_models\DenseEncoder_DenseDecoder_+0.774_2020-07-28_15_51_10.dat'
PATH='results\model\DenseEncoder_DenseDecoder_DenseDecoder_+0.610_+0.611_2023-01-12_17-54-07.dat'



def make_payload(width, height, depth, text):
    """
    This takes a piece of text and encodes it into a bit vector. It then
    fills a matrix of size (width, height) with copies of the bit vector.
    """
    message = text_to_bits(text) + [0] * 32

    payload = message
    while len(payload) < width * height * depth:
        payload += message

    payload = payload[:width * height * depth]

    return torch.FloatTensor(payload).view(1, depth, height, width)

def make_message(image, decoder):
    image = image.to(device)
    
    image = decoder(image).view(-1) > 0
    image=torch.tensor(image, dtype=torch.uint8)

    # split and decode messages
    candidates = Counter()
    bits = image.data.cpu().numpy().tolist()
    for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
      #print(candidate)
      candidate = bytearray_to_text(bytearray(candidate))
      if candidate:
          candidates[candidate] += 1

    # choose most common message
    if len(candidates) == 0:
      return "###!No data found!###"
      raise ValueError('Failed to find message.')

    candidate, count = candidates.most_common(1)[0]
    return candidate


def test_audio(encoder,decoder1,decoder2,data_depth,train_epoch,cover,payload1,payload2):
  # %matplotlib inline
  generated = encoder.forward(cover, payload1, payload2)

  decoded1 = decoder1.forward(generated)
  decoder_loss1 = binary_cross_entropy_with_logits(decoded1, payload1)
  decoder_acc1 = (decoded1 >= 0.0).eq(
    payload1 >= 0.5).sum().float() / payload1.numel() # .numel() calculate the number of element in a tensor
  decoded2 = decoder2.forward(generated)
  decoder_loss2 = binary_cross_entropy_with_logits(decoded2, payload2)
  decoder_acc2 = (decoded2 >= 0.0).eq(
    payload2 >= 0.5).sum().float() / payload2.numel() # .numel() calculate the number of element in a tensor

  print("Decoder loss: %.3f, %.3f"% (decoder_loss1.item(), decoder_loss2.item()))
  print("Decoder acc: %.3f, %.3f"% (decoder_acc1.item(), decoder_acc2.item()))

  f, ax = plt.subplots(1, 3,figsize=(16,5))
  f.suptitle("%s_%s_%s"%(encoder.name,decoder1.name,decoder2.name), fontsize=16)
  f.tight_layout(pad=4.0)
  if len(cover.shape)==4:
    cover_=cover.squeeze(0).cpu().detach().numpy()
  else:
    cover_=cover.cpu().detach().numpy()
  cover_spec=cover_[0]+1j*cover_[1]
  librosa.display.specshow(cover_spec, x_axis='time', fmin=0,fmax=22050,
                          y_axis='mel', sr=22050, ax=ax[0])
  ax[0].set_title('Cover image')
  if len(generated.shape)==4:
    generated_=generated.squeeze(0).cpu().detach().numpy()
  else:
    generated_=generated.cpu().detach().numpy()
  generated_spec=generated_[0]+1j*generated_[1]
  librosa.display.specshow(generated_spec, x_axis='time', fmin=0,fmax=22050,
                          y_axis='mel', sr=22050, ax=ax[1])
  ax[1].set_title('Generated image')
  payload_=cover_spec-generated_spec
  img=librosa.display.specshow(payload_, x_axis='time', y_axis='mel', fmin=0,fmax=22050,
                           sr=22050, ax=ax[2])
  ax[2].set_title('Generated payloads')

  return generated


def aud_to_melspectro(wav):
   wav=wav[0]
   len_=wav.shape[0]
   hop_length_=int(len_/(360-1))
   n_fft_=int((360-1)*2)
   mel=librosa.stft(np.asarray(wav),n_fft=n_fft_, hop_length=hop_length_)
   H_,W_=mel.shape
   mel_real=np.real(mel)
   mel_imag=np.imag(mel)
   if not mel_real.shape == (360,360):
      mel_real=skimage.transform.resize(image=mel_real,output_shape=(360,360),order=1)
   if not mel_imag.shape == (360,360):
      mel_imag=skimage.transform.resize(image=mel_imag,output_shape=(360,360),order=1)
   return torch.tensor([mel_real,mel_imag]).float(), hop_length_


if __name__ == '__main__':
  for func in [
            lambda: os.mkdir(os.path.join('.', 'results')),
            lambda: os.mkdir(os.path.join('.', 'results/model')),
            lambda: os.mkdir(os.path.join('.', 'results/plots'))]:  # create directories
    try:
      func()
    except Exception as error:
      print(error)
      continue

  METRIC_FIELDS = [
        'val.encoder_mse',
        'val.decoder_loss',
        'val.decoder_acc1',
        'val.decoder_acc2',
        'val.cover_score',
        'val.generated_score',
        'val.ssim',
        'val.psnr',
        'val.bpp',
        'train.encoder_mse',
        'train.decoder_loss',
        'train.decoder_acc1',
        'train.decoder_acc2',
        'train.cover_score',
        'train.generated_score',
  ]

  writer = SummaryWriter(log_dir)
  print('Tensorboard logs stored in: ',writer.get_logdir())

  # load datasets
  data_dir="FSDKaggle2018"#"genres"#"music_speech" # directory to audio
  channels_size=2
  transform = transforms.Compose([transforms.Lambda(lambda wav: aud_to_melspectro(wav))])
  train_set = AudioToImageFolder(os.path.join(
        data_dir, "train\\"), transform=transform)
  part_train_set = torch.utils.data.random_split(train_set, [800, len(train_set)-800])[0]
  train_loader = torch.utils.data.DataLoader(
        part_train_set, batch_size=4, shuffle=True,)
  valid_set = AudioToImageFolder(os.path.join( 
        data_dir, "val\\"), transform=transform)
  part_valid_set = torch.utils.data.random_split(valid_set, [100, len(valid_set)-100])[0]
  valid_loader = torch.utils.data.DataLoader(
        part_valid_set, batch_size=4, shuffle=True)
  

  encoder = DenseEncoder(data_depth, hidden_size,channels_size).to(device)
  decoder1 = DenseDecoder(data_depth, hidden_size,channels_size).to(device)
  decoder2 = DenseDecoder(data_depth, hidden_size,channels_size).to(device)
  critic = BasicCritic(hidden_size,channels_size).to(device)
  cr_optimizer = Adam(critic.parameters(), lr=1e-4)
  en_de_optimizer = Adam(list(decoder1.parameters()) + list(decoder2.parameters()) + list(encoder.parameters()), lr=1e-4)
  metrics = {field: list() for field in METRIC_FIELDS}

  # load module
  if torch.cuda.is_available():
    checkpoint = torch.load(PATH)
  else:
    checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
              
  critic.load_state_dict(checkpoint['state_dict_critic'])
  encoder.load_state_dict(checkpoint['state_dict_encoder'])
  decoder1.load_state_dict(checkpoint['state_dict_decoder1'])
  decoder2.load_state_dict(checkpoint['state_dict_decoder2'])
  en_de_optimizer.load_state_dict(checkpoint['en_de_optimizer'])
  cr_optimizer.load_state_dict(checkpoint['cr_optimizer'])
  metrics=checkpoint['metrics']
  ep=checkpoint['train_epoch']
  date=checkpoint['date']
  critic.train(mode=False)
  encoder.train(mode=False)
  decoder1.train(mode=False)
  decoder2.train(mode=False)
  print('GAN loaded')
  # print(critic)
  # print(encoder)
  # print(decoder1)
  # print(decoder2)
  # print(en_de_optimizer)
  # print(cr_optimizer)
  # print(date)

  # test for single audio
  cover, path, hop_length = valid_set[129]

  _, H, W = cover.size()
  cover = cover[None].to(device)

  text1 = "Lorem ipsum"
  text2 = "Fake message"
  payload1 = make_payload(W, H, data_depth, text1)
  payload2 = make_payload(W, H, data_depth, text2)
  payload1 = payload1.to(device)
  payload2 = payload2.to(device)
  generated = test_audio(encoder,decoder1,decoder2,data_depth,epochs,cover,payload1,payload2)

  text_return_1 = make_message(generated, decoder1)
  text_return_2 = make_message(generated, decoder2)
  print('Message1 found: ',text_return_1)
  print('Message2 found: ',text_return_2)