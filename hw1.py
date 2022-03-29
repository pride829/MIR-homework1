import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

path = "./HW1 Datasets/GTZAN/GTZAN/wav"

genres_list = ['disco', 'reggae', 'pop', 'rock',
               'metal', 'jazz', 'blues', 'hiphop', 'country']

correct_accumulate_dict = {'disco': 0, 'reggae': 0, 'pop': 0, 'rock': 0,
                           'metal': 0, 'jazz': 0, 'blues': 0, 'hiphop': 0, 'country': 0}

file_num_dict = {'disco': 0, 'reggae': 0, 'pop': 0, 'rock': 0,
                 'metal': 0, 'jazz': 0, 'blues': 0, 'hiphop': 0, 'country': 0}

# -- 1) binary-valued template matching
# -- 2) K-S template matching
# -- 3) harmonic template matching (you may try 𝛼 = 0.9).

# -- Using Cicular Shifting to generate another tone
binary_C_major = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
binary_C_minor = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
# -- C, C#, D, D#, E, F, F#, G, G#, A, A#, B
KS_C_major = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
KS_C_minor = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]

for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith((".wav")):
            print("============")

            txt_file_path = name.replace(".wav", ".lerch.txt")
            txt_file_path = txt_file_path.replace("wav", "key")
            print("txt_file_path = ", txt_file_path)
            genres = txt_file_path.split(".")[0]
            print("genres", genres)

            file_num_dict[genres] = file_num_dict[genres] + 1

            y, sr = librosa.load(
                r"./HW1 Datasets/GTZAN/GTZAN/wav/" + genres + "/" + name)

            print(y.shape)
            print(sr)

            plt.plot(y)

            # -- STFT chromagram
            chroma_stft = librosa.feature.chroma_stft(
                y=y, sr=sr, n_chroma=12, n_fft=4096)
            # -- CQT chromagram
            chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
            # -- Chroma Energy Normalized chromagram
            chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
            print(chroma_cens.shape)  # 12 dim pitch, 1293 frames

            # -- Chromagram visualization
            # plt.figure(figsize=(15, 15))
            # plt.subplot(3, 1, 1)
            # librosa.display.specshow(chroma_stft, y_axis='chroma')
            # plt.title('chroma_stft')
            # plt.colorbar()
            # plt.subplot(3, 1, 2)
            # librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
            # plt.title('chroma_cqt')
            # plt.colorbar()
            # plt.subplot(3, 1, 3)
            # librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time')
            # plt.title('chroma_cens')
            # plt.colorbar()
            # plt.tight_layout()

            # plt.show()

            # compute x = 1/N sigma(i=1, N){Zi}
            avg_chromagram = []
            frame_n = chroma_stft.shape[1]
            for pitchs in range(chroma_stft.shape[0]):
                accumulate_value = 0
                for frames in range(chroma_stft.shape[1]):
                    accumulate_value += chroma_stft[pitchs][frames]
                avg_chromagram.append(accumulate_value/frame_n)

            print(avg_chromagram)

            inner_product = []

            # Major
            for i in range(12):
                #print(np.roll(binary_C_major, i))
                inner_product.append(
                    np.dot(avg_chromagram, np.roll(binary_C_major, i)))
            # minor
            for i in range(12):
                #print(np.roll(binary_C_minor, i))
                inner_product.append(
                    np.dot(avg_chromagram, np.roll(binary_C_minor, i)))

            # print(inner_product)
            maxindex = np.argmax(inner_product)
            # print(maxindex)

            scale = ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']

            annotation_major_scale = ['A', 'A#', 'B', 'C', 'C#',
                                      'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
            annotation_minor_scale = ['a', 'a#', 'b', 'c', 'c#',
                                      'd', 'd#', 'e', 'f', 'f#', 'g', 'g#']

            if maxindex / 12 < 1:
                print(scale[maxindex] + " Major")
            if maxindex / 12 == 1:
                print(scale[maxindex] + " minor")

            moving_scale = {0: '15', 1: '16', 2: '17', 3: '18', 4: '19', 5: '20',
                            6: '21', 7: '22', 8: '23', 9: '0', 10: '1', 11: '2',
                            12: '3', 13: '4', 14: '5', 15: '6', 16: '7', 17: '8',
                            18: '9', 19: '10', 20: '11', 21: '12', 22: '13', 23: '14',
                            }

            print("prediction", moving_scale[maxindex])

            with open("./HW1 Datasets/GTZAN/GTZAN/key/" + genres + "/" + txt_file_path) as f:
                answer = f.readline().rstrip()
            print("answer", answer)

            print(moving_scale[maxindex] == answer)
            if (moving_scale[maxindex] == answer):
                correct_accumulate_dict[genres] = correct_accumulate_dict[genres] + 1
                #acc += 1

print(correct_accumulate_dict)
print(file_num_dict)

for i in range(len(correct_accumulate_dict)):
    print(genres_list[i], " Avg Accuracy = ", correct_accumulate_dict[genres_list[i]] /
          file_num_dict[genres_list[i]])
