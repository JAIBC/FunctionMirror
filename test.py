import streamlit as st
import time
import random
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import webbrowser
import requests


model = tf.keras.models.load_model("model.h5")
label_mapping = {0: "Rock", 1: "Paper", 2: "Scissors"}

cap = cv2.VideoCapture(0)

def main_screen():
    st.markdown("<div align='center'><h1>Function Mirror</h1></div>", unsafe_allow_html=True)
    if st.button("가위바위보"):
        st.session_state["screen"] = "game_screen"
    if st.button("양치"):
        countdown(180)

def predict_move(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = frame.resize((224, 224))
    frame = np.array(frame) / 255.0
    frame = np.expand_dims(frame, axis=0)
    prediction = model.predict(frame)
    predicted_index = np.argmax(prediction)
    predicted_move = label_mapping[predicted_index]
    return predicted_move

def game_screen():
    st.title("가위, 바위, 보")

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    stframe = st.empty()

    options = ["Rock", "Paper", "Scissors"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        move = predict_move(frame)

        stframe.image(frame, channels="BGR")
        st.write("당신이 낸 것 :", move)

        if move in options:
            if st.button("시작"):
                computer_choice = random.choice(options)
                result = determine_winner(move, computer_choice)
                st.write("컴퓨터 :", computer_choice)
                st.write("결과 :", result)
                break

    cap.release()


def determine_winner(player_choice, computer_choice):
    if player_choice == computer_choice:
        return "비김!!"
    elif (
        (player_choice == "Rock" and computer_choice == "Scissors") or
        (player_choice == "Paper" and computer_choice == "Rock") or
        (player_choice == "Scissors" and computer_choice == "Paper")
    ):
        return "승리!"
    else:
        return "패배!"

def main():
    st.set_page_config(page_title="Function Mirror", page_icon="🎮")
    if "screen" not in st.session_state:
        st.session_state["screen"] = "main_screen"

    if st.session_state["screen"] == "main_screen":
        main_screen()
    elif st.session_state["screen"] == "game_screen":
        game_screen()
    


def countdown(duration):
    st.write("양치 시작!")
    remaining_time = st.empty()
    while duration >= 0:
        mins, secs = divmod(duration, 60)
        time_format = "{:02d}:{:02d}".format(mins, secs)
        remaining_time.text(time_format)
        time.sleep(1)
        duration -= 1
    st.write("양치 끝!")

if __name__ == "__main__":
    main()
