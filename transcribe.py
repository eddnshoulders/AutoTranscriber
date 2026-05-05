import sys
import os
import time
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from faster_whisper import WhisperModel

def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("-cpu", "-gpu"):
        print("Usage: python transcribe.py [-cpu|-gpu] <audio_file>")
        sys.exit(1)

    device_option = sys.argv[1]
    audio_path = sys.argv[2]

    if not os.path.isfile(audio_path):
        print(f"Error: File '{audio_path}' does not exist.")
        sys.exit(1)

    device, compute_type = ("cpu", "int8") if device_option == "-cpu" else ("cuda", "float16")

    start_time = time.time()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Transcribing '{audio_path}' using device: {device}")

    if device == "cuda":
        model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    else:
        model = WhisperModel("distil-large-v2", device=device, compute_type=compute_type, cpu_threads=32, num_workers=8)

    segments, info = model.transcribe(audio_path, language="en")

    output_file = os.path.splitext(audio_path)[0] + "_transcript.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Detected language: {info.language}\n\n")
        for segment in segments:
            f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Transcription saved to: {output_file}")

if __name__ == "__main__":
    main()
