import spacy
import json
import yaml
import json
import random
import os
import glob
import pandas as pd
import datetime
import argparse
import pathlib
import requests
import pytesseract
import multiprocessing
import gunicorn.app.base
from PIL import Image
from spacy.lang.en import English
from spacy.util import minibatch
from flask import Flask, request

class Identifier:
    def __init__(self):
        with open('identifier_config.yml') as config_file:
            config = yaml.load(config_file, Loader=yaml.SafeLoader)
            self.epochs = int(config['config']['epochs'])
            self.training_data = str(config['config']['training_data_file'])
            self.label1 = str(config['config']['labels']['label1'])
            self.label2 = str(config['config']['labels']['label2'])
            self.label3 = str(config['config']['labels']['label3'])
            self.label4 = str(config['config']['labels']['label4'])
            self.label5 = str(config['config']['labels']['label5'])
            self.label6 = str(config['config']['labels']['label6'])

    def train(self):
        if os.path.isfile(self.training_data):
            data = pd.read_csv(self.training_data)
        else:
            try:
                print("Error: training_data.csv missing. Creating training_data.csv")
                td = open("training_data.csv","w+")
                td.write('label,"text"')
                td.close()
                raise SystemExit
            except (IOError, OSError):
                print('Error: traning_data.csv missing. Failed to create training_data.csv')
                raise SystemExit

        nlp = English()  # use directly
        nlp = spacy.blank("en")  # blank instance

        # Create the TextCategorizer with exclusive classes and "bow" architecture
        textcat = nlp.create_pipe(
                      "textcat",
                      config={
                        "exclusive_classes": True,
                        "architecture": "bow"})

        # Add the TextCategorizer to the empty model
        nlp.add_pipe(textcat)

        # Add labels to text classifier
        textcat.add_label(self.label1)
        textcat.add_label(self.label2)
        textcat.add_label(self.label3)
        textcat.add_label(self.label4)
        textcat.add_label(self.label5)
        textcat.add_label(self.label6)

        train_texts = data['text'].values
        train_labels = [{'cats': {self.label1: label == self.label1,
                                  self.label2: label == self.label2,
                                  self.label3: label == self.label3,
                                  self.label4: label == self.label4,
                                  self.label5: label == self.label5,
                                  self.label6: label == self.label6}}
                        for label in data['label']]

        train_data = list(zip(train_texts, train_labels))

        random.seed(1)
        spacy.util.fix_random_seed(1)
        optimizer = nlp.begin_training()
        drop = 0.25
        losses = {}

        for epoch in range(self.epochs):
            print('Epoch: ' + str(epoch + 1) + '/' + str(self.epochs) + ' ' + str(100*round((epoch + 1)/self.epochs, 4)) + '%', flush=True)
            random.shuffle(train_data)
            # Create the batch generator with batch size = 8
            batches = minibatch(train_data, size=8)
            # Iterate through minibatches
            for batch in batches:
                # Each batch is a list of (text, label) but we need to
                # send separate lists for texts and labels to update().
                # This is a quick way to split a list of tuples into lists
                texts, labels = zip(*batch)
                nlp.update(texts, labels, drop=drop, sgd=optimizer, losses=losses)

        if not os.path.isdir('models'):
            print('Models directory does not exist, creating folder')
            try:
                os.mkdir('models')
            except:
                print('Failed to create models directory')
                raise SystemExit(4)
        print('Saving model')
        nlp.to_disk(os.path.dirname(os.path.realpath(__file__)) + "/models/" + str(datetime.datetime.now()).split('.')[0])
        raise SystemExit

    def find_newest_model(self):
        if not os.path.isdir('models'):
            print('Models directory does not exist, creating folder')
            try:
                os.mkdir('models')
            except:
                print('Failed to create models directory')
                raise SystemExit(4)
        newest_model_time = 0
        newest_model_path = ""
        # finds newest spacy model in subdirectory "models"
        for x in range(len(glob.glob(os.path.join("models", '*')))):
            time = os.path.getmtime(glob.glob(os.path.join("models", '*'))[x])
            name = glob.glob(os.path.join("models", '*'))[x]
            if time > newest_model_time:
                newest_model_time = time
                newest_model_path = name
        if os.path.isdir(newest_model_path):
            return newest_model_path
        else:
            print('No model available. Train one by running ocr_identifier.py with the -t flag (ocr_identifier -t)')
            raise SystemExit(4)

    def ocr(self, path):
        img_original = Image.open(path)
        width, height = img_original.size
        ratio = 3000/width
        new_width = 3000
        new_height = height*ratio
        img_original.thumbnail((new_width, new_height))
        img_original.save(path)
        img_original.close()
        img = Image.open(path).convert('L')
        text = pytesseract.image_to_string(img, lang = 'swe')
        text = text[:-3]
        new_text = ""
        lines = text.split("\n")
        for x in range(len(lines)):
            if lines[x] == '':
                lines[x] = '\n'
            new_text = new_text + ' ' + lines[x]
        input = new_text.split('\n')
        return input

    def json_conversion(self, labels):
        return json.dumps(labels, ensure_ascii=False)
        
    def identify(self, input):
        nlp = spacy.load(identifier.find_newest_model())

        label1 = {}
        label2 = {}
        label3 = {}
        label4 = {}
        label5 = {}
        label6 = {}

        for y in range(len(input)):
            docs = [nlp.tokenizer(input[y])]

            # Use textcat to get the scores for each doc
            textcat = nlp.get_pipe('textcat')
            scores, _ = textcat.predict(docs)

            # From the scores, find the label with the highest score/probability
            predicted_labels = scores.argmax(axis=1)
            for label in predicted_labels:
                if self.label1 in textcat.labels[label]:
                    input[y] = input[y].replace('"', '')
                    label1[str(self.label1) + str(y)] = input[y]
                elif self.label2 in textcat.labels[label]:
                    input[y] = input[y].replace('"', '')
                    label2[str(self.label2) + str(y)] = input[y]
                elif self.label3 in textcat.labels[label]:
                    input[y] = input[y].replace('"', '')
                    label3[str(self.label3) + str(y)] = input[y]
                elif self.label4 in textcat.labels[label]:
                    input[y] = input[y].replace('"', '')
                    label4[str(self.label4) + str(y)] = input[y]
                elif self.label5 in textcat.labels[label]:
                    input[y] = input[y].replace('"', '')
                    label5[str(self.label5) + str(y)] = input[y]
                elif self.label6 in textcat.labels[label]:
                    input[y] = input[y].replace('"', '')
                    label6[str(self.label6) + str(y)] = input[y]
                labels = label1.copy()
                labels.update(label2)
                labels.update(label3)
                labels.update(label4)
                labels.update(label5)
                labels.update(label6)
        return identifier.json_conversion(labels)

def number_of_workers():
    return (multiprocessing.cpu_count() * 2) + 1

def port():
    if os.path.isfile('docker-compose.yml'):
        try:
            with open('docker-compose.yml') as docker_compose_file:
                docker_compose_config = yaml.load(docker_compose_file, Loader=yaml.SafeLoader)
                return str(docker_compose_config['services']['ocr']['ports'][0].split(':')[1])
        except:
            try:
                with open('identifier_config.yml') as config_file:
                    config_file = yaml.load(config_file, Loader=yaml.SafeLoader)
                    return str(config_file['config']['port'])
            except:
                print('Error: Port not set. Set Port in docker-compose.yml')
                print('If you do not intend to use docker-compose set port the in identifier_config.yml')
                raise SystemExit
    else:
        try:
            with open('identifier_config.yml') as config_file:
                config_file = yaml.load(config_file, Loader=yaml.SafeLoader)
                return str(config_file['config']['port'])
        except:
            print('Error: Port not set. Set Port in docker-compose.yml')
            print('If you do not intend to use docker-compose set port the in identifier_config.yml')
            raise SystemExit

app = Flask(__name__)
@app.route('/', methods=['POST'])
def result():
    file_path = request.json['file_path']
    print('POST: {"file_path":"' + file_path + '"}')

    if os.path.isfile(file_path):
        return str(identifier.identify(identifier.ocr(str(file_path))))
    else:
        print(file_path + ' is not a file')
        return file_path + ' is not a file'

class StandaloneApplication(gunicorn.app.base.BaseApplication):

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == '__main__':
    options = {
        'bind': '%s:%s' % ('0.0.0.0', port()),
        'workers': number_of_workers(),
    }
    identifier = Identifier()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-t',
                           '--train',
                           dest='t_exists',
                           action='store_true',
                           help='train a new model using the traning data')

    if vars(argparser.parse_args())['t_exists'] is True:
        identifier.train()
    StandaloneApplication(app, options).run()
