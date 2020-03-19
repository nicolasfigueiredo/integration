import mido
import numpy as np
import math
from skimage.transform import resize

#timestamp: pega 30 segundos do meio do piano roll
def midi_to_piano_roll_augmented(midi_file, timestamp):
    ticks_in_track = np.ceil(mido.second2tick(midi_file.length, tempo=500000, ticks_per_beat=midi_file.ticks_per_beat))
    decay_per_tick = 28 / mido.second2tick(1, tempo=500000, ticks_per_beat=midi_file.ticks_per_beat)
    
    piano_roll = np.zeros([128, int(ticks_in_track)])   # fazer piano roll com eixo x em ticks e y em midi note number
    # linha 127 é a do pedal!!

    time = 0
    pedal_flag = 0           # 1 if sustain pedal is pressed, 0 otherwise

    # faz mapa de note ons e note offs
    for msg in midi_file.tracks[1]:
        time += msg.time
        if msg.type == 'note_on':
            if msg.velocity > 0:
                piano_roll[msg.note][time] = msg.velocity
            else: # note-off
                piano_roll[msg.note][time] = -1
        elif msg.type == 'control_change':
            if pedal_flag == 0 and msg.value>0: # o pedal manda diversas mensagens durante uma única pressionada
                piano_roll[127][time] = 1
                pedal_flag = 1
            elif pedal_flag == 1 and msg.value == 0:
                piano_roll[127][time] = 1
                pedal_flag = 0
                
    # faz ''linha do tempo'' do pedal (trata os note ons e offs, criando um piano roll para o pedal)
    pedal_flag = 0
    for j in range(piano_roll.shape[1]):
        if piano_roll[127,j] == 1:
            pedal_flag = int(not pedal_flag)
        piano_roll[127,j] = pedal_flag
    
    keypress_flag = 0  # indica se tecla está pressionada
    note_vel = 0    # indica se nota está soando
    noteon_flag = 0
    pedal_flag = 0     # indica se pedal está pressionado
    
    for i in range(127):
        if np.sum(piano_roll[i, :] != 0):   # se tiver alguma nota nessa linha
            note_vel = 0
            noteon_flag = 0
            pedal_flag = 0
            keypress_flag = 0
            for j in range(piano_roll.shape[1]):
                if piano_roll[i,j] > 0:    # nota foi pressionada
                    keypress_flag = 1
                    note_vel = piano_roll[i,j]
                elif piano_roll[i,j] == -1:   # achou um note-off
                    keypress_flag = 0

                pedal_flag = piano_roll[127,j]
                noteon_flag = keypress_flag or (pedal_flag and noteon_flag) # nota soa quando tecla é pressionada 
                                                                            # ou quando já estava soando e pedal está pressionado
                if noteon_flag:
                    piano_roll[i,j] = note_vel
                
                note_vel -= decay_per_tick
                if note_vel < 0:
                    noteon_flag = 0
                    note_vel = 0
                
    start_time = int(mido.second2tick(timestamp, tempo=500000, ticks_per_beat=midi_file.ticks_per_beat))
    stop_time  = int(mido.second2tick(timestamp+30, tempo=500000, ticks_per_beat=midi_file.ticks_per_beat))
    return piano_roll[:127,start_time:stop_time]

def midi_to_piano_roll(midi_file, timestamp):
    ticks_in_track = np.ceil(mido.second2tick(midi_file.length, tempo=500000, ticks_per_beat=midi_file.ticks_per_beat))
    
    piano_roll = np.zeros([128, int(ticks_in_track)])   # fazer piano roll com eixo x em ticks e y em midi note number
    # linha 127 é a do pedal!!

    time = 0
    pedal_flag = 0           # 1 if sustain pedal is pressed, 0 otherwise

    # faz mapa de note ons e note offs
    for msg in midi_file.tracks[1]:
        time += msg.time
        if msg.type == 'note_on':
            piano_roll[msg.note][time] = 1
        elif msg.type == 'control_change':
            if pedal_flag == 0 and msg.value>0: # o pedal manda diversas mensagens durante uma única pressionada
                piano_roll[127][time] = 1
                pedal_flag = 1
            elif pedal_flag == 1 and msg.value == 0:
                piano_roll[127][time] = 1
                pedal_flag = 0
                
    # faz ''linha do tempo'' do pedal (trata os note ons e offs, criando um piano roll para o pedal)
    pedal_flag = 0
    for j in range(piano_roll.shape[1]):
        if piano_roll[127,j] == 1:
            pedal_flag = int(not pedal_flag)
        piano_roll[127,j] = pedal_flag
    
    keypress_flag = 0  # indica se tecla está pressionada
    noteon_flag = 0    # indica se nota está soando
    pedal_flag = 0     # indica se pedal está pressionado
    
    for i in range(127):
        if np.sum(piano_roll[i, :] != 0):   # se tiver alguma nota nessa linha
            noteon_flag = 0
            pedal_flag = 0
            keypress_flag = 0
            for j in range(piano_roll.shape[1]):
                if piano_roll[i,j] == 1 and keypress_flag == 0:    # nota foi pressionada
                    keypress_flag = 1
                elif piano_roll[i,j] == 1 and keypress_flag:   # achou um note-off
                    keypress_flag = 0

                pedal_flag = piano_roll[127,j]
                noteon_flag = keypress_flag or (pedal_flag and noteon_flag) # nota soa quando tecla é pressionada 
                                                                            # ou quando já estava soando e pedal está pressionado
                piano_roll[i,j] = noteon_flag
                
    start_time = int(mido.second2tick(timestamp, tempo=500000, ticks_per_beat=midi_file.ticks_per_beat))
    stop_time  = int(mido.second2tick(timestamp+30, tempo=500000, ticks_per_beat=midi_file.ticks_per_beat))
    return piano_roll[:127,start_time:stop_time]
    
# LEMBRAR DO ''MASCARAMENTO'': SELECIONAR EVENTO MÁXIMO QUE CAI NO BIN

def filter_negatives(x):
    if x < 0:
        return 0
    else:
        return x

f_neg = np.vectorize(filter_negatives)

def compose_sound_event_map_aug(piano_roll, n_harmonics=7):
    new_piano_roll = np.zeros([piano_roll.shape[0]+n_harmonics*12, piano_roll.shape[1]])
    
    for i in range(127):
<<<<<<< HEAD
        new_piano_roll[i] = np.max([new_piano_roll[i], piano_roll[i]], axis=0) # isso pq overtones podem cair em cima de linhas que tenham notas
        for j in range(1,n_harmonics+1):
            partial = piano_roll[i] - 7.86 * j  # atenuação de velocity por harmônico
            new_piano_roll[i+12*j] = np.max([new_piano_roll[i+12*j], partial], axis=0)            

                    # if note_vel < 0:
                    # noteon_flag = 0
                    # note_vel = 0
=======
        new_piano_roll[i] = np.max([new_piano_roll[i], piano_roll[i]], axis=0)
        for j in range(1,n_harmonics+1):
            partial = piano_roll[i] - 7.86 * j  # atenuação por harmônico
            new_piano_roll[i+12*j] = np.max([new_piano_roll[i+12*j], partial], axis=0)
            
>>>>>>> 159f00e833569a016ac0b637559b80166a4d0aea
        
    return f_neg(new_piano_roll) / 127

def compose_sound_event_map(piano_roll, n_harmonics=7):
    new_piano_roll = np.zeros([piano_roll.shape[0]+n_harmonics*12, piano_roll.shape[1]])
    
    for i in range(127):
        new_piano_roll[i] = new_piano_roll[i] + piano_roll[i]
        for j in range(1,n_harmonics+1):
            new_piano_roll[i+12*j] = new_piano_roll[i+12*j] + piano_roll[i]
        
    return (new_piano_roll > 0).astype(int)

def fft_frequencies(sr=44100, n_fft=2048):
    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
    endpoint=True)

def midi_to_hz(midi_note_number):
    
    g = 2**(1/12)
    return 440*g**(midi_note_number-69)

def build_midi_freqs(number_midi_notes):
    midi_freqs = []
    for i in range(number_midi_notes):
        midi_freqs.append(midi_to_hz(i))
    return midi_freqs

# Acha o indice do elemento de array mais perto de value
def find_nearest_idx(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

# Passa dos eixos nota midi x sound ticks para freq. bin x time frame, de acordo com um n_fft e hop_size definidos
def sound_event_to_tfp(sound_event_map, num_time_frames, n_fft=2048):
    midi_freqs = build_midi_freqs(sound_event_map.shape[0])
    fft_freqs = fft_frequencies(n_fft=n_fft)
    tfp_pianoroll = np.zeros([fft_freqs.shape[0] , sound_event_map.shape[1]])
    
    for i in range(len(midi_freqs)):
        idx = find_nearest_idx(fft_freqs, midi_freqs[i])
        tfp_pianoroll[idx] = sound_event_map[i]
        
    tfp_pianoroll = resize(tfp_pianoroll, [tfp_pianoroll.shape[0],num_time_frames])
    
    return tfp_pianoroll