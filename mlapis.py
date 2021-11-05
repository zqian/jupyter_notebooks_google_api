#!/usr/bin/env python
# coding: utf-8

# <h1> Using Machine Learning APIs </h1>
# 
# First, visit <a href="http://console.cloud.google.com/apis">API console</a>, choose "Credentials" on the left-hand menu.  Choose "Create Credentials" and generate an API key for your application. You should probably restrict it by IP address to prevent abuse, but for now, just  leave that field blank and delete the API key after trying out this demo.
# 
# Copy-paste your API Key here:

# In[1]:


# Use the chown command to change the ownership of repository to user
get_ipython().system('sudo chown -R jupyter:jupyter /home/jupyter/training-data-analyst')


# In[11]:


APIKEY=""  # Replace with your API key


# <b> Note: Make sure you generate an API Key and replace the value above. The sample key will not work.</b>
# 

# <h2> Invoke Translate API </h2>

# In[7]:


# Running the Translate API
from googleapiclient.discovery import build
service = build('translate', 'v2', developerKey=APIKEY)

# Use the service
inputs = ['is it really this easy?', 'amazing technology', 'wow']
outputs = service.translations().list(source='en', target='zh', q=inputs).execute()

# Print outputs
for input, output in zip(inputs, outputs['translations']):
  print("{0} -> {1}".format(input, output['translatedText']))


# <h2> Invoke Vision API </h2>
# 
# The Vision API can work off an image in Cloud Storage or embedded directly into a POST message. I'll use Cloud Storage and do OCR on this image: <img src="https://storage.googleapis.com/cloud-training-demos/vision/sign2.jpg" width="200" />.  That photograph is from http://www.publicdomainpictures.net/view-image.php?image=15842
# 

# In[25]:


# Running the Vision API
import base64
IMAGE="gs://cloud-training-demos/images/met/APS6880.jpg"
vservice = build('vision', 'v1', developerKey=APIKEY)
request = vservice.images().annotate(body={
        'requests': [{
                'image': {
                    'source': {
                        'gcs_image_uri': IMAGE
                    }
                },
                'features': [{
                    'type': 'FACE_DETECTION',
                    'maxResults': 3,
                }]
            }],
        })
responses = request.execute(num_retries=3)

# Let's output the `responses`
print(responses)

foreigntext = responses['responses'][0]['faceAnnotations'][0]['joyLikelihood']
foreignlang = responses['responses'][0]['faceAnnotations'][0]['sorrowLikelihood']

# Let's output the `foreignlang` and `foreigntext`
print(foreignlang, foreigntext)
# <h2> Translate sign </h2>

# In[12]:


inputs=[foreigntext]
outputs = service.translations().list(source=foreignlang, target='en', q=inputs).execute()

# Print the outputs
for input, output in zip(inputs, outputs['translations']):
  print("{0} -> {1}".format(input, output['translatedText']))


# <h2> Sentiment analysis with Language API </h2>
# 
# Let's evaluate the sentiment of some famous quotes using Google Cloud Natural Language API.

# In[13]:


# Evaluating the sentiment with Google Cloud Natural Language API
lservice = build('language', 'v1beta1', developerKey=APIKEY)
quotes = [
  'To succeed, you must have tremendous perseverance, tremendous will.',
  'It’s not that I’m so smart, it’s just that I stay with problems longer.',
  'Love is quivering happiness.',
  'Love is of all passions the strongest, for it attacks simultaneously the head, the heart, and the senses.',
  'What difference does it make to the dead, the orphans and the homeless, whether the mad destruction is wrought under the name of totalitarianism or in the holy name of liberty or democracy?',
  'When someone you love dies, and you’re not expecting it, you don’t lose her all at once; you lose her in pieces over a long time — the way the mail stops coming, and her scent fades from the pillows and even from the clothes in her closet and drawers. '
]
for quote in quotes:
# The `documents.analyzeSentiment` method analyzes the sentiment of the provided text.
  response = lservice.documents().analyzeSentiment(
    body={
      'document': {
         'type': 'PLAIN_TEXT',
         'content': quote
      }
    }).execute()
  polarity = response['documentSentiment']['polarity']
  magnitude = response['documentSentiment']['magnitude']

# Lets output the value of each `polarity`, `magnitude` and `quote`
  print('POLARITY=%s MAGNITUDE=%s for %s' % (polarity, magnitude, quote))


# <h2> Speech API </h2>
# 
# The Speech API can work on streaming data, audio content encoded and embedded directly into the POST message, or on a file on Cloud Storage. Here I'll pass in this <a href="https://storage.googleapis.com/cloud-training-demos/vision/audio.raw">audio file</a> in Cloud Storage.

# In[16]:


# Using the Speech API by passing audio file as an argument
sservice = build('speech', 'v1', developerKey=APIKEY)
# The `speech.recognize` method performs synchronous speech recognition.
# It receive results after all audio has been sent and processed.
response = sservice.speech().recognize(
    body={
        'config': {
            'languageCode' : 'en-US',
            'encoding': 'LINEAR16',
            'sampleRateHertz': 16000
        },
        'audio': {
            'uri': 'gs://cloud-training-demos/vision/audio.raw'
            }
        }).execute()

# Let's output the value of `response`
print(response)


# In[17]:


print(response['results'][0]['alternatives'][0]['transcript'])

# Let's output the value of `'Confidence`
print('Confidence=%f' % response['results'][0]['alternatives'][0]['confidence'])


# <h2> Clean up </h2>
# 
# Remember to delete the API key by visiting <a href="http://console.cloud.google.com/apis">API console</a>.
# 
# If necessary, commit all your notebooks to git.
# 
# If you are running Datalab on a Compute Engine VM or delegating to one, remember to stop or shut it down so that you are not charged.
# 

# ## Challenge Exercise
# 
# Here are a few portraits from the Metropolitan Museum of Art, New York (they are part of a [BigQuery public dataset](https://bigquery.cloud.google.com/dataset/bigquery-public-data:the_met) ):
# 
# * gs://cloud-training-demos/images/met/APS6880.jpg
# * gs://cloud-training-demos/images/met/DP205018.jpg
# * gs://cloud-training-demos/images/met/DP290402.jpg
# * gs://cloud-training-demos/images/met/DP700302.jpg
# 
# Use the Vision API to identify which of these images depict happy people and which ones depict unhappy people.
# 
# Hint (highlight to see): <p style="color:white">You will need to look for joyLikelihood and/or sorrowLikelihood from the response.</p>

# Copyright 2020 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# In[ ]:




