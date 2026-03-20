# Guitar Note Identifier

Input:
- mono or stereo WAV guitar recording

Output:
- estimated pitches over time
- note names over time
- candidate guitar fretboard positions
- static fretboard visualization

## Install

````bash
pip install -r requirements.txt
````

## Run

````bash
python app.py path/to/guitar.wav
````

Optional arguments:

````bash
python app.py path/to/guitar.wav --frame-size 2048 --hop-size 256 --fmin 80 --fmax 1200 --topk 3
````
