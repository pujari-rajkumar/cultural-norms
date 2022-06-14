# all datasets
from typing import Tuple
from multimethod import Dict
import torch
from torch.utils.data import Dataset

import utils

class MPDD(Dataset):
    def __init__(self, dialogs: dict, context:bool =False, sep_token:str ='[SEP]') -> None:
        super().__init__()
        self.context = context
        self.sep_token = sep_token
        dialog_speakers = {}
        self.emotions = ['angry', 'surprise', 'sadness', 'happiness', 'neutral', 'fear', 'disgust']
        self.emotion_to_id = {e:i for i, e in enumerate(self.emotions)}
        self.id_to_emotion = {v:k for k, v in self.emotion_to_id.items()}
        self.num_emotions = len(self.emotions)
        for key, dialog in dialogs.items():
            sent_speakers = dialog_speakers[key] = []
            for utter_id, utter in enumerate(dialog):
                speaker = utter['speaker']
                emotion = utter['emotion']
                emotion_id = self.emotion_to_id.get(emotion, self.emotion_to_id['neutral'])
                utterance = utils.tokenizer(utter['utterance'])
                for sent in utterance.sentences:
                    sent_speaker = {}
                    tokens = [token.text for token in sent.tokens]
                    sent = " ".join(tokens)
                    sent_speaker['utter_id'] = utter_id
                    sent_speaker['sent'] = sent
                    sent_speaker['speaker'] = speaker
                    sent_speaker['emotion_id'] = emotion_id
                    sent_speakers.append(sent_speaker)

        self.data_idx = 0
        self.data = {}
        for key, sent_speakers in dialog_speakers.items():
            for i, sent_speaker in enumerate(sent_speakers[:-1]):
                utter_id = sent_speaker['utter_id']
                sent = sent_speaker['sent']
                emotion_id = sent_speaker['emotion_id']
                speaker = sent_speaker['speaker']
                next_speaker = sent_speakers[i + 1]['speaker']
                next_emotion_id = sent_speakers[i + 1]['emotion_id']
                if speaker is not None and speaker != next_speaker:
                    label = 1
                else:
                    label = 0
                
                if emotion_id != next_emotion_id:
                    emotion_label = 1
                else:
                    emotion_label = 0

                if self.context and i > 0:
                    prev_sent = sent_speakers[i - 1]['sent']
                    input_sent = f"{sent} {self.sep_token} {prev_sent}"
                else:
                    input_sent = sent
                
                data_i = self.data[self.data_idx] = {}
                data_i['dialog_id'] = int(key)
                data_i['utter_id'] = utter_id

                data_i['sent'] = input_sent

                data_i['emotion_id'] = emotion_id
                
                data_i['emotion_label'] = emotion_label
                data_i['label'] = label
                self.data_idx += 1


    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Dict[int, dict]:
        return self.data[index]