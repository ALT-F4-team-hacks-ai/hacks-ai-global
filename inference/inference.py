import pandas as pd
import numpy as np
import whisper
import os
import torch
import re

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from transformers import (
    RobertaForTokenClassification,
    RobertaTokenizer,
    BertTokenizer,
    BertForTokenClassification,
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_whisper = whisper.load_model("base")

def get_termins(text):

    tokenizer_ru_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    model_ru_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

    tokenizer_en_ru = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    model_en_ru = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sentences = text.replace('! ', '. ').replace('? ', '. ').split('. ')

    sentences_new = []

    for i in range(len(sentences)):
      if len(sentences[i])>=512:
        for j in range(len(sentences[i])//512+1):
          sentences_new.append(sentences[i][512*j:512*(j+1)])
      else:
        sentences_new.append(sentences[i])

    sentences = sentences_new.copy()

    
    for i in range(len(sentences)):
        sentences[i] = re.sub("\xa0", " ", sentences[i])

    sentences_en = []

    for article in sentences:

        inputs = tokenizer_ru_en(article, return_tensors="pt")
        translated_tokens = model_ru_en.generate(**inputs, max_length=512)

        sentences_en.append(
            tokenizer_ru_en.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        )

    for i in range(len(sentences_en)):
        sentence = sentences_en[i]

        for punc in [".", ","]:
            for pattern in [rf"([^0-9{punc} ])(\{punc})", rf"(\{punc})([^0-9{punc} ])"]:
                # while re.search(pattern, sentence):
                sentence = re.sub(pattern, r"\1 \2", sentence)

        for punc in list("!?()-"):
            for pattern in [rf"([^{punc}])(\{punc})", rf"(\{punc})([^{punc}])"]:
                # while re.search(pattern, sentence):
                sentence = re.sub(pattern, r"\1 \2", sentence)
        for punc in list(".!?()-"):
            while re.search(rf"(\{punc}) (\{punc})", sentence):
                sentence = re.sub(rf"(\{punc}) (\{punc})", r"\1\2", sentence)

        sentence = re.sub("\xa0", " ", sentence)
        sentences_en[i] = sentence

    with open("data_file.txt", "w") as f:
        for sentence in sentences_en:
            for word in sentence.split():
                f.write(word + "\t" + "O" + "\n")
            f.write("\n")

    classes = ["O", "B-Term", "I-Term", "B-Def", "I-Def"]
    num_classes = len(classes)

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512
    mdodel = BertForTokenClassification.from_pretrained(
        model_name, num_labels=num_classes
    )
    model.load_state_dict(
        torch.load("/content/8_weights", map_location=device)["model_state_dict"]   # Путь в модели
    )

        


    class NERDataset(Dataset):
        def __init__(self, file_path):
            self.sentences, self.labels = self.read_data(file_path)
            self.label2id = dict(zip(classes, list(range(len(classes)))))
            self.id2label = {v: k for k, v in self.label2id.items()}

        def read_data(self, file_path):
            self.label2id = dict(zip(classes, list(range(len(classes)))))
            sentences = []
            labels = []

            with open(file_path, 'r', encoding='utf-8') as file:
                current_sentence = []
                current_labels = []

                for line in file:
                    if line.strip() == '':
                        if current_sentence:
                            encodings = tokenizer(current_sentence, truncation=True, padding='max_length', is_split_into_words=True)
                            current_labels = current_labels + ['O'] * (tokenizer.model_max_length - len(current_labels))
                            encodings['input_ids'] = torch.tensor(encodings['input_ids'])
                            encodings['attention_mask'] = torch.tensor(encodings['attention_mask'])
                            sentences.append(encodings)
                            current_labels = torch.tensor([self.label2id[label] for label in current_labels])
                            labels.append(current_labels)


                            current_sentence = []
                            current_labels = []
                    else:
                        word, label = line.strip().split('\t')
                        current_sentence.append(word)
                        current_labels.append(label)

                if current_sentence:
                    encodings = tokenizer(current_sentence, truncation=True, padding='max_length', is_split_into_words=True)
                    current_labels = current_labels + ['O'] * (tokenizer.model_max_length - len(current_labels))
                    encodings['input_ids'] = torch.tensor(encodings['input_ids'])
                    encodings['attention_mask'] = torch.tensor(encodings['attention_mask'])
                    sentences.append(encodings)
                    current_labels = torch.tensor([self.label2id[label] for label in current_labels])
                    labels.append(current_labels)


            return sentences, labels

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            words = self.sentences[idx]
            labels = self.labels[idx]

            # Convert labels to numerical values
            # label_ids = [self.label2id[label] for label in labels]
            label_ids = labels

            return {'input_ids': words['input_ids'], 'attention_mask': words['attention_mask'] ,'labels': label_ids}
            return words, label_ids


    # Create dataset
    file_path = 'data_file.txt'
    dataset = NERDataset(file_path)
    
    
    
    label2id = dict(zip(classes, list(range(len(classes)))))
    id2label = {v: k for k, v in label2id.items()}


    batch_size = 64
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    

    
    def process_preds(pred_values):

      for i in range(pred_values.shape[0]-1):
        if pred_values[i]==pred_values[i+1]==1:
          pred_values[i+1] = 2
        elif pred_values[i]==pred_values[i+1]==3:
          pred_values[i+1] = 4

      is_open = 0
      ind_open = 0
      if 1 in pred_values and 2 in pred_values:
        for i in range(pred_values.shape[0]-1):
          if pred_values[i] == 1 and pred_values[i+1] != 2:
            is_open = 1
            ind_open = i
          elif pred_values[i] == 1 and pred_values[i+1] == 2:
            is_open = 0
          elif pred_values[i] == 2 and is_open:
            pred_values[ind_open:i] = 2
            is_open = 0

      is_open = 0
      ind_open = 0
      if 3 in pred_values and 4 in pred_values:
        for i in range(pred_values.shape[0]-1):
          if pred_values[i] == 3 and pred_values[i+1] != 4:
            is_open = 3
            ind_open = i
          elif pred_values[i] == 3 and pred_values[i+1] == 4:
            is_open = 0
          elif pred_values[i] == 4 and is_open:
            pred_values[ind_open:i] = 4
            is_open = 0


      for i in range(pred_values.shape[0]-1):
        if pred_values[i] not in [1, 2] and pred_values[i+1]==2:
          pred_values[i+1] = 1

        elif pred_values[i] not in [3, 4] and pred_values[i+1]==4:
          pred_values[i+1] = 3

      return pred_values

    glossary = dict()

    model = model.eval().to(device)

    for i, batch in enumerate(dataloader):

        with torch.no_grad():

            batch = { k: v.to(device) for k, v in batch.items() }

            outputs = model(**batch)

        s_lengths = batch['attention_mask'].sum(dim=1)

        for idx, length in enumerate(s_lengths):

            pred_values = torch.argmax(outputs[1], dim=2)[idx][:length]

            # print(pred_values)

            pred_values = process_preds(pred_values)

            # print(pred_values)

            sentence = sentences_en[batch_size*i+idx]

            if pred_values.sum().item()!=0:
              term, definition = '', ''
              for word, pred in zip(sentence.split()[:length], pred_values):
                if pred == 1:
                  term = word
                elif pred == 2:
                  term += ' '+ word
                elif pred == 3:
                  definition = word
                elif pred == 4:
                  definition += ' ' + word
                else:
                  if definition and term:
                    glossary[term] = definition
                    term, definition = '', ''

              if definition and term:
                    glossary[term] = definition
                    term, definition = '', ''






    # final_glossary = dict()
    final_glossary = []


    for term, definition in glossary.items():


            inputs = tokenizer_en_ru(f"'{term}': '{definition}'", return_tensors="pt")

            translated_tokens = model_en_ru.generate(
              **inputs, max_length=512
            )

            final_glossary.append(tokenizer_en_ru.batch_decode(translated_tokens, skip_special_tokens=True)[0])

    return final_glossary


def inference_termins(filename):
  #1. транскрибируем текст
  result = model_whisper.transcribe(filename)
  result_text = result["text"]
  #2. выделяем термины
  termins = get_termins(result_text)
  new_termins = []
  for termin in termins: 
    termin = re.sub("'", "", termin)
    termin = re.sub("\"", "", termin)
    new_termins.append(termin)

  final_results_dict = {"File" : [], "Term" : []}
  for i in range(len(new_termins)):
    termin = new_termins[i].split(":")[0]
    final_results_dict["File"].append(filename)
    final_results_dict["Term"].append(termin)

  return pd.DataFrame.from_dict(final_results_dict)

filename = ""
audi="data/" + filename
result_df = inference_termins(audi)
result_df["File"] = filename
result_df.to_csv(f"result/{filename[:-4]}.csv", index=False)
