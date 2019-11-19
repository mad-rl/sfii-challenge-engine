FROM ubuntu:18.04

WORKDIR .

RUN apt update
RUN apt -y install python3.7 python3-pip git python-opengl xvfb ffmpeg
RUN python3.7 -m pip install torch torchvision gym-retro numpy Pillow

COPY . .

ENV ROM_PATH ./roms/
ENV PYTHONPATH ./src/
ENV GAME_FOLDER ./usr/local/lib/python3.7/dist-packages/retro/data/stable/StreetFighterIISpecialChampionEdition-Genesis/

RUN chmod +x start.sh

CMD [ "./start.sh" ]