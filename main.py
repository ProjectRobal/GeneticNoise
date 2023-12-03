import pyaudio
import wave
import librosa
import numpy as np
import os

AUDIO_FREQ=44100 # audio frequency

AUDIO_LENGTH=3 # in seconds

AUDIO_AMPLITUDE=int((2**16 - 1) / 2)

CROSSOVER_CHUNK_NUM=5

TRACKS_NUMBERS=10

DIRECTORY="tracks"

class Music:
    def __init__(self,length:float=None,arr:np.ndarray=None,path:str=None) -> None:
        self.reward:int=0.0

        if path is not None:
            self.open(path)
            return

        if arr is not None:
            self.samples:np.ndarray=arr
            return

        if length is not None:
            self.samples:np.ndarray=np.zeros(length*AUDIO_FREQ,np.float32)
            return
    
    def open(self,path:str):
        self.samples,_=librosa.load(path,sr=AUDIO_FREQ,mono=True,dtype=np.float32)

    def generate(self):
        self.samples[:]=np.random.uniform(0.0,1.0,len(self.samples))[:]

    def applyReward(self,reward:int):
        self.reward=reward

    @property
    def length(self)->int:
        return len(self.samples)
    
    def save(self,name:str):
        wav:wave.Wave_write=wave.open(DIRECTORY+"/"+name,"wb")

        wav.setframerate(AUDIO_FREQ)
        wav.setsampwidth(2)
        wav.setnchannels(1)

        wav.writeframes((self.samples*AUDIO_AMPLITUDE).astype(np.int16).tobytes())

        wav.close()


def crossover(m1:Music,m2:Music)->tuple[Music,Music]:
    chunk_size:int=np.min([m1.length,m2.length])/CROSSOVER_CHUNK_NUM

    chunks_m1=np.split(m1.samples,chunk_size)
    chunks_m2=np.split(m2.samples,chunk_size)

    size=np.min([len(chunks_m1),len(chunks_m2)])

    childA=np.zeros(int(size*CROSSOVER_CHUNK_NUM),np.float32)
    childB=np.zeros(int(size*CROSSOVER_CHUNK_NUM),np.float32)


    for i in range(size-1):
        childA[int(i*CROSSOVER_CHUNK_NUM):int((i+1)*CROSSOVER_CHUNK_NUM)]=chunks_m1[i]*(i%2) + chunks_m2[i]*((i+1)%2)
        childB[int(i*CROSSOVER_CHUNK_NUM):int((i+1)*CROSSOVER_CHUNK_NUM)]=chunks_m2[i]*(i%2) + chunks_m1[i]*((i+1)%2)


    return (Music(arr=childA),Music(arr=childB))

def mutate(m1:Music):

    for samp in m1.samples:
        if np.random.random(1)<0.1:
            samp+=np.random.random(1)

if not os.path.exists("./"+DIRECTORY):
    os.mkdir("./"+DIRECTORY)    

audio=pyaudio.PyAudio()

speaker=audio.open(AUDIO_FREQ,1,pyaudio.paInt16,output=True)

def play_music(music:Music):
    to_play=music.samples*AUDIO_AMPLITUDE
    to_play=to_play.astype(np.int16)
    speaker.write(to_play.tobytes())

def evolve(musics:list[Music]):
    musics=sorted(musics,key=lambda x: x.reward,reverse=True)

    # remove worst performing population
    musics=musics[0:int(len(musics)/2)]

    childs:list[Music]=[]

    for i in range(0,len(musics),2):
        if i+1 < len(musics):
            (ch1,ch2)=crossover(musics[i],musics[i+1])

            childs.append(ch1)
            childs.append(ch2)
    
    if len(childs)>TRACKS_NUMBERS:

        for i in range(TRACKS_NUMBERS-len(childs)):
            childs.append(Music(length=10))
            childs[-1].generate()

    for ch in childs:
        mutate(ch)

    return childs


musics=[]

track_dir=os.listdir("./"+DIRECTORY)

if len(track_dir)==0:

    for i in range(TRACKS_NUMBERS):
        musics.append(Music(AUDIO_LENGTH))
        musics[-1].generate()
        musics[-1].save("track"+str(i)+".wav")

else:

    for track in track_dir:
        music=Music(path="./"+DIRECTORY+"/"+track)

        musics.append(music)


while True:
    for i,music in enumerate(musics):
        print("Music: "+str(i))
        play_music(music)
        score=-1
        while score < 0:
            try:
                score=input("Score: ")
                if score == "exit":
                    exit(0)
                score=int(score)
            except:
                print("Wrong score selected")
        music.applyReward(score)
        
    musics=evolve(musics)

    for i,m in enumerate(musics):
        m.save("track"+str(i)+".wav")

    print("New batch:")
