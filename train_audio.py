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
PATH='results\model\DenseEncoder_DenseDecoder_DenseDecoder_+0.721_+0.721_2023-01-12_20-19-13.dat'



def save_model(encoder,decoder1,decoder2,critic,en_de_optimizer,cr_optimizer,metrics,ep):
    now = datetime.datetime.now()
    cover_score1 = metrics['val.decoder_acc1'][-1]
    cover_score2 = metrics['val.decoder_acc2'][-1]
    name = "%s_%s_%s_%+.3f_%+.3f_%s.dat" % (encoder.name,decoder1.name,decoder2.name,cover_score1,cover_score2,
                                   now.strftime("%Y-%m-%d_%H-%M-%S"))
    fname = os.path.join('.', 'results\\model', name)
    states = {
            'state_dict_critic': critic.state_dict(),
            'state_dict_encoder': encoder.state_dict(),
            'state_dict_decoder1': decoder1.state_dict(),
            'state_dict_decoder2': decoder2.state_dict(),
            'en_de_optimizer': en_de_optimizer.state_dict(),
            'cr_optimizer': cr_optimizer.state_dict(),
            'metrics': metrics,
            'train_epoch': ep,
            'date': now.strftime("%Y-%m-%d_%H:%M:%S"),
    }
    torch.save(states, fname)


def fit_gan(encoder,decoder1,decoder2,critic,en_de_optimizer,cr_optimizer,metrics,train_loader,valid_loader):
      iter_train_critic=0
      iter_train_enc_dec=0
      iter_valid=0
      writer = SummaryWriter()
      for ep in range(epochs):
        print("Epoch %d" %(ep+1))
        for cover, *rest in tqdm(train_loader):
            if torch.cuda.is_available():
              cover = cover.cuda()
            iter_train_critic+=1
            gc.collect()
            cover = cover.to(device)
            N, _, H, W = cover.size()
            # sampled from the discrete uniform ditribution over 0 to 2
            payload1 = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            payload2 = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            generated = encoder.forward(cover, payload1, payload2)
            cover_score = torch.mean(critic.forward(cover))
            generated_score = torch.mean(critic.forward(generated))

            cr_optimizer.zero_grad()
            (cover_score - generated_score).backward(retain_graph=False)
            cr_optimizer.step()
            
            for p in critic.parameters():
                p.data.clamp_(-0.1, 0.1)
            writer.add_scalar('cover_score/train', cover_score.item(), iter_train_critic)
            writer.add_scalar('generated_score/train', generated_score.item(), iter_train_critic)

            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.generated_score'].append(generated_score.item())
              
            for tag, value in critic.named_parameters():
              tag = tag.replace('.', '/')
              writer.add_histogram('critic/'+tag, value.data.cpu().numpy(), iter_train_critic)
              writer.add_histogram('critic/'+tag+'/grad', value.grad.data.cpu().numpy(), iter_train_critic)
            
        for cover, *rest in tqdm(train_loader):
            if torch.cuda.is_available():
              cover = cover.cuda()
            iter_train_enc_dec+=1
            gc.collect()
            cover = cover.to(device)
            N, _, H, W = cover.size()
            # sampled from the discrete uniform distribution over 0 to 2
            payload1 = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            payload2 = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            generated = encoder.forward(cover, payload1, payload2)
            decoded1 = decoder1.forward(generated)
            decoded2 = decoder2.forward(generated)
            encoder_mse = mse_loss(generated, cover)
            decoder_loss1 = binary_cross_entropy_with_logits(decoded1, payload1)
            decoder_acc1 = (decoded1 >= 0.0).eq(
                payload1 >= 0.5).sum().float() / payload1.numel()
            decoder_loss2 = binary_cross_entropy_with_logits(decoded2, payload2)
            decoder_acc2 = (decoded2 >= 0.0).eq(
                payload2 >= 0.5).sum().float() / payload2.numel()
            generated_score = torch.mean(critic.forward(generated))
            

            en_de_optimizer.zero_grad()
            (100 * encoder_mse + decoder_loss1 + decoder_loss2 + generated_score).backward()  # Why 100?
            en_de_optimizer.step()

            writer.add_scalar('encoder_mse/train', encoder_mse.item(), iter_train_enc_dec)
            writer.add_scalar('decoder_loss1/train', decoder_loss1.item(), iter_train_enc_dec)
            writer.add_scalar('decoder_loss2/train', decoder_loss2.item(), iter_train_enc_dec)
            writer.add_scalar('decoder_acc1/train', decoder_acc1.item(), iter_train_enc_dec)
            writer.add_scalar('decoder_acc2/train', decoder_acc2.item(), iter_train_enc_dec)

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss1'].append(decoder_loss1.item())
            metrics['train.decoder_loss2'].append(decoder_loss2.item())
            metrics['train.decoder_acc1'].append(decoder_acc1.item())
            metrics['train.decoder_acc2'].append(decoder_acc2.item())

            for tag, value in encoder.named_parameters():
              tag = tag.replace('.', '/')
              writer.add_histogram('encoder/'+tag, value.data.cpu().numpy(), iter_train_enc_dec)
              writer.add_histogram('encoder/'+tag+'/grad', value.grad.data.cpu().numpy(), iter_train_enc_dec)

            for tag, value in decoder1.named_parameters():
              tag = tag.replace('.', '/')
              writer.add_histogram('decoder1/'+tag, value.data.cpu().numpy(), iter_train_enc_dec)
              writer.add_histogram('decoder1/'+tag+'/grad', value.grad.data.cpu().numpy(), iter_train_enc_dec)
            
            for tag, value in decoder2.named_parameters():
              tag = tag.replace('.', '/')
              writer.add_histogram('decoder2/'+tag, value.data.cpu().numpy(), iter_train_enc_dec)
              writer.add_histogram('decoder2/'+tag+'/grad', value.grad.data.cpu().numpy(), iter_train_enc_dec)
            
        for cover, *rest in tqdm(valid_loader):
            if torch.cuda.is_available():
              cover = cover.cuda()
            iter_valid+=1
            gc.collect()
            cover = cover.to(device)
            
            N, _, H, W = cover.size()
            # sampled from the discrete uniform distribution over 0 to 2
            payload1 = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            payload2 = torch.zeros((N, data_depth, H, W),
                                  device=device).random_(0, 2)
            generated = encoder.forward(cover, payload1, payload2)

            decoded1 = decoder1.forward(generated)
            decoded2 = decoder2.forward(generated)

            encoder_mse = mse_loss(generated, cover)
            decoder_loss1 = binary_cross_entropy_with_logits(decoded1, payload1)
            decoder_acc1 = (decoded1 >= 0.0).eq(
                payload1 >= 0.5).sum().float() / payload1.numel()
            decoder_loss2 = binary_cross_entropy_with_logits(decoded2, payload2)
            decoder_acc2 = (decoded2 >= 0.0).eq(
                payload2 >= 0.5).sum().float() / payload2.numel()
            generated_score = torch.mean(critic.forward(generated))
            cover_score = torch.mean(critic.forward(cover))

            ssim_=ssim(cover, generated)
            psnr_=10 * torch.log10(4 / encoder_mse)
            bbp_1=data_depth * (2 * decoder_acc1.item() - 1)
            bbp_2=data_depth * (2 * decoder_acc2.item() - 1)

            writer.add_scalar('encoder_mse/test', encoder_mse.item(), iter_valid)
            writer.add_scalar('decoder_loss1/test', decoder_loss1.item(), iter_valid)
            writer.add_scalar('decoder_loss2/test', decoder_loss2.item(), iter_valid)
            writer.add_scalar('decoder_acc1/test', decoder_acc1.item(), iter_valid)
            writer.add_scalar('decoder_acc2/test', decoder_acc2.item(), iter_valid)
            writer.add_scalar('cover_score/test', cover_score.item(), iter_valid)
            writer.add_scalar('generated_score/test', generated_score.item(), iter_valid)
            writer.add_scalar('ssim/test', ssim_.item(), iter_valid)
            writer.add_scalar('psnr/test', psnr_.item(), iter_valid)
            writer.add_scalar('bbp1/test',bbp_1, iter_valid)
            writer.add_scalar('bbp2/test',bbp_2, iter_valid)

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss1'].append(decoder_loss1.item())
            metrics['val.decoder_loss2'].append(decoder_loss2.item())
            metrics['val.decoder_acc1'].append(decoder_acc1.item())
            metrics['val.decoder_acc2'].append(decoder_acc2.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            metrics['val.ssim'].append(ssim_.item())
            metrics['val.psnr'].append(psnr_.item())
            metrics['val.bpp1'].append(bbp_1)
            metrics['val.bpp2'].append(bbp_2)

        writer.add_image('cover/test', torch.tensor(np.abs(cover[:,0,:,:].cpu().detach().numpy()+1j*cover[:,1,:,:].cpu().detach().numpy())).unsqueeze(0), ep, dataformats='CNHW')
        writer.add_image('generated/test', torch.tensor(np.abs(generated[:,0,:,:].cpu().detach().numpy()+1j*generated[:,1,:,:].cpu().detach().numpy())).unsqueeze(0), ep, dataformats='CNHW')
        writer.add_image('payload1/test', torch.tensor(np.abs((cover[:,0,:,:].cpu().detach().numpy()+1j*cover[:,1,:,:].cpu().detach().numpy())-(generated[:,0,:,:].cpu().detach().numpy()+1j*generated[:,1,:,:].cpu().detach().numpy()))).unsqueeze(0), ep, dataformats='CNHW')
        writer.add_image('payload2/test', torch.tensor(np.abs((cover[:,0,:,:].cpu().detach().numpy()+1j*cover[:,1,:,:].cpu().detach().numpy())-(generated[:,0,:,:].cpu().detach().numpy()+1j*generated[:,1,:,:].cpu().detach().numpy()))).unsqueeze(0), ep, dataformats='CNHW')
        

        print('encoder_mse: %.3f - decoder_loss: %.3f %.3f - decoder_acc: %.3f %.3f - cover_score: %.3f - generated_score: %.3f - ssim: %.3f - psnr: %.3f - bpp: %.3f %.3f'
          %(encoder_mse.item(),decoder_loss1.item(),decoder_loss2.item(),decoder_acc1.item(),decoder_acc2.item(),cover_score.item(),generated_score.item(), ssim_.item(),psnr_.item(),bbp_1, bbp_2))
      
        save_model(encoder,decoder1,decoder2,critic,en_de_optimizer,cr_optimizer,metrics,ep)
        
      writer.add_graph(encoder,(cover, payload1))
      writer.add_graph(encoder,(cover, payload2))
      writer.add_graph(decoder1,generated)
      writer.add_graph(decoder2,generated)
      writer.add_graph(critic,generated)
      writer.close()


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
        'val.decoder_loss1',
        'val.decoder_loss2',
        'val.decoder_acc1',
        'val.decoder_acc2',
        'val.cover_score',
        'val.generated_score',
        'val.ssim',
        'val.psnr',
        'val.bpp1',
        'val.bpp2',
        'train.encoder_mse',
        'train.decoder_loss1',
        'train.decoder_loss2',
        'train.decoder_acc1',
        'train.decoder_acc2',
        'train.cover_score',
        'train.generated_score',
  ]

  writer = SummaryWriter(log_dir)
  print('Tensorboard logs stored in: ',writer.get_logdir())

  #load datasets
  print('audio')
  data_dir="FSDKaggle2018"#"genres"#"music_speech" # directory to audio
  channels_size=2
  transform = transforms.Compose([transforms.Lambda(lambda wav: aud_to_melspectro(wav))])
  train_set = AudioToImageFolder(os.path.join(
        data_dir, "test\\"), transform=transform)
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

  # start train
  fit_gan(encoder,decoder1,decoder2,critic,en_de_optimizer,cr_optimizer,metrics,train_loader,valid_loader)