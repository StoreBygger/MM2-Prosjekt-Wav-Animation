import numpy as np
import matplotlib

matplotlib.use("QtAgg")  # må stå FØR pyplot importeres
import matplotlib.pyplot as plt
from numpy.fft import fftfreq
from scipy.io import wavfile
from scipy.fft import fft, ifft
from matplotlib.animation import FuncAnimation
import subprocess as sp
import progressbar
import os


def animate_func(input_file: str, output_file: str, fps: int):

    # Read input file
    sample_rate, wav = wavfile.read(input_file)
    num_samples = wav.shape[0]

    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    else:
        wav = wav.astype(np.float32)

    # Hvis to kanaler, legg dem sammen og normaliser
    if len(wav.shape) == 2:
        wav = np.mean(wav, axis=1)  # legg sammen
    spf = int(
        round(sample_rate / fps)
    )  # samples per frame, hvor stor mange samples fft-en skal ta for seg
    T = 1.0 / sample_rate  # Tidsmellomrom mellom samples
    N = int(round(spf))

    total_frames = max(
        0, (len(wav) - spf) // spf + 1
    )  # Hvor mange frames / bilder det er totalt

    print(
        f"INPUT FILE:{input_file}\nOUTPUT FILE:{output_file}\nSAMPLE RATE: {sample_rate}\nNUM SAMPLES: {num_samples}\nTOTAL FRAMES: {total_frames}\nSPF / N: {spf}\nT: {T}\n"
    )

    # Progressbar for å se status av rendering
    bar = progressbar.ProgressBar(maxval=total_frames)
    print("PROGRESS FOURIER TRANSFORM:")
    bar.start()

    # Lage liste for verdier langs x-aksen
    xf = fftfreq(N, T)[: N // 2]

    # Regne ut ny fft for hvert bilde
    def fft_wav(n):
        n0 = int(n) * spf  # Start sample -n
        n1 = n0 + spf  # slutt sample n

        fftwav = wav[n0:n1]  # Kutte wav-filen fra n0 til n1

        if len(fftwav) < N:
            fftwav = np.pad(seq, (0, N - len(fftwav)))
        yf = fft(fftwav)  # fft av denkuttet den
        yf = 2.0 / N * np.abs(yf[0 : N // 2])  # Fjerne negative verdier
        return yf

    # Lag figur til plot
    fig = plt.figure()

    # precompute maks y
    ymax = 1e-9
    for i in range(total_frames):
        y = fft_wav(i)  # samme som i animate
        ymax = max(ymax, np.max(y))

    # Sett på akse - begrensninger
    axis = plt.axes(xlim=(0, 4000), ylim=(0, 1.1 * ymax))
    axis.set_xlabel("Frekvens [Hz]")
    axis.set_ylabel("Amplitude")

    (line,) = axis.plot([], [], lw=3)

    # initialiser animasjon linje
    def init():
        line.set_data([], [])
        return (line,)

    # funkjson for hver frame av animasjonen
    def animate(i):
        x = xf

        y = fft_wav(i)
        line.set_data(x, y)

        bar.update(i)  # Oppdattere progressbaren
        return (line,)

    # animer graf

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=total_frames,
        interval=1000 / fps,
        blit=True,
    )

    # Lag temp-fil og lagre graf på den
    temp_file = "./temp-fft-no-sound.mp4"
    anim.save(temp_file, writer="ffmpeg", fps=fps)
    bar.finish()
    print("FINISHED FOURIER TRANSFORM, ADDING AUDIO NOW")
    # Legg til lydfilen over animasjons-videoen
    sp.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            temp_file,
            "-i",
            input_file,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_file,
        ],
        stdout=sp.DEVNULL,
        stderr=None,
    )
    print("FINISHED ADDING AUDIO")
    os.remove(temp_file)


def main():
    wav_file = "./seven-nation-army.wav"
    out_file = "./seven-nation-army-fft-60fps-nojmp.mp4"
    animate_func(wav_file, out_file, 60)


if __name__ == "__main__":
    main()
