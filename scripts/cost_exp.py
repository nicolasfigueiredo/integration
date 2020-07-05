import librosa
import timeit

def test():
    """Stupid test function"""
    L = []
    for i in range(100):
        L.append(i)

def setup():
	import librosa
	y, sr = librosa.load('../data/example.wav', sr=44100)

if __name__ == '__main__':
	print(timeit.timeit("librosa.stft(y, n_fft=1024)", setup=setup))