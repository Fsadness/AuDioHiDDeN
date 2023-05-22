import librosa
import librosa.display 
import matplotlib.pyplot as plt
from IPython.display import clear_output


def plot(name, train_epoch, values, path, save):
    clear_output(wait=True)
    plt.close('all')
    fig = plt.figure()
    fig = plt.ion()
    fig = plt.subplot(1, 1, 1)
    fig = plt.title('epoch: %s -> %s: %s' % (train_epoch, name, values[-1]))
    fig = plt.ylabel(name)
    fig = plt.xlabel('validation_set')
    fig = plt.plot(values)
    fig = plt.grid()
    get_fig = plt.gcf()
    fig = plt.draw()  # draw the plot
    fig = plt.pause(1)  # show it for 1 second
#     if save:
#         now = datetime.datetime.now()
#         get_fig.savefig('%s/%s_%.3f_%d_%s.png' %
#                         (path, name, train_epoch, values[-1], now.strftime("%Y-%m-%d_%H:%M:%S")))

def plot_audio(mel):
  fig, ax = plt.subplots()
  img = librosa.display.specshow(mel, x_axis='time', fmin=0,fmax=6000,
                          y_axis='mel', sr=6000, ax=ax)
  fig.colorbar(img, ax=ax,format='%+2.0f dB')