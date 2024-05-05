import json
import os

import nussl

def iemocap_analysis():
    text_transcriptions = {}
    emotions = {
        "ang": 0,
        "hap": 0,
        "exc": 0,
        "sad": 0,
        "fru": 0,
        "fea": 0,
        "sur": 0,
        "neu": 0,
        "dis": 0
    }
    iemocap_emotion_exclude = ["hap", "xxx", "oth", "fea", "sur", "dis"]
    transcriptions = os.listdir("IEMOCAP/transcription")
    for f in transcriptions:
        transcription = open("IEMOCAP/transcription/" + f).readlines()
        for line in transcription:
            inf = line.split(":")
            if inf[0] not in text_transcriptions:
                text_transcriptions[inf[0].split()[0]] = inf[1]

    emotion_files = os.listdir("IEMOCAP/emotions")
    iemocap_data = []
    for f in emotion_files:
        dialogue = open("IEMOCAP/emotions/" + f).readlines()
        relevant_lines = []
        for line in dialogue:
            if line[0] == "[":
                relevant_lines.append(line)
        for relevant_line in relevant_lines:
            emotion_record = relevant_line.split("\t")
            if emotion_record[2] not in iemocap_emotion_exclude:
                filename = emotion_record[1] + ".wav"
                emotion = emotion_record[2]
                emotions[emotion] += 1
                text = text_transcriptions[emotion_record[1]].strip()
                iemocap_data.append({"filename": filename, "emotion": emotion, "text": text})

    print("done")



def audio_analysis():
    train_folder = "train"
    val_folder = "val"
    test_folder = "test"
    trains = os.listdir(train_folder)
    vals = os.listdir(val_folder)
    tests = os.listdir(test_folder)
    counter = 0
    timer = 0.0
    for f in trains:
        counter += 1
        signal1 = nussl.AudioSignal("train/" + f)
        timer += signal1.signal_duration
    for f in vals:
        counter += 1
        signal1 = nussl.AudioSignal("val/" + f)
        timer += signal1.signal_duration
    for f in tests:
        counter += 1
        signal1 = nussl.AudioSignal("test/" + f)
        timer += signal1.signal_duration

    print("a")

def main():
    test_folder = "test"
    tests = os.listdir(test_folder)
    f = open("data/Subtask_2_2_train.json")
    data = json.load(f)
    analysis = {"utterances": 0, "dialogues": 0, "pairs": 0, "utt_cause": 0, "self_cause": 0, "future_cause": 0,
                "only_self_cause": 0, "only_different_cause": 0, "neutral_cause": 0, "non_cause_emotions": 0}
    emotions = {"surprise": 0, "joy": 0, "sadness": 0, "neutral": 0, "disgust": 0, "anger": 0, "fear": 0}
    speakers = {}
    for conv in data:
        analysis["utterances"] += len(conv["conversation"])
        analysis["dialogues"] += 1
        analysis["pairs"] += len(conv["emotion-cause_pairs"])
        non_neutral_emotions = []
        for utt in conv["conversation"]:
            emotions[utt["emotion"]] += 1
            if utt["emotion"] != "neutral":
                non_neutral_emotions.append(utt["utterance_ID"])
            if utt["speaker"] not in speakers:
                speakers[utt["speaker"]] = 0
            speakers[utt["speaker"]] += 1

        last = "0"
        conv_emotions = {}
        for emotion_pair in conv["emotion-cause_pairs"]:
            utterance_id = emotion_pair[0].split("_")[0]
            if int(utterance_id) in non_neutral_emotions:
                non_neutral_emotions.remove(int(utterance_id))
            if emotion_pair[0].split("_")[1] == "neutral":
                analysis["neutral_cause"] += 1
            if utterance_id not in conv_emotions:
                conv_emotions[utterance_id] = []
            conv_emotions[utterance_id].append(emotion_pair[1])
            if last != utterance_id:
                analysis["utt_cause"] += 1
                last = utterance_id

            if utterance_id == emotion_pair[1]:
                analysis["self_cause"] += 1

            if int(utterance_id) < int(emotion_pair[1]):
                analysis["future_cause"] += 1

        analysis["non_cause_emotions"] += len(non_neutral_emotions)
        for conv_emotion in conv_emotions:
            if len(conv_emotions[conv_emotion]) == 1 and conv_emotions[conv_emotion][0] == conv_emotion:
                analysis["only_self_cause"] += 1
            different = True
            for cause in conv_emotions[conv_emotion]:
                if cause == conv_emotion:
                    different = False
            if different:
                analysis["only_different_cause"] += 1
    print("a")





if __name__ == "__main__":
    iemocap_analysis()
    # audio_analysis()
    # main()